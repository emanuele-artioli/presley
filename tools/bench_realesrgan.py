#!/usr/bin/env python3
"""Is there a cheap speedup left in the fastest restorer we ship?

Referee 2's throughput objection is that even Real-ESRGAN, the cheapest
restorer in `tab:throughput`, runs at 3.80 fps against a 24 fps source. The
response is either a measured speedup or a scoped reframing, and this measures
whether the first is available.

**Pre-registered before running:** a speedup is reportable only if it is
>= 1.2x AND leaves the output within one JND of the baseline configuration. A
faster restorer that changes its output is a different restorer, not an
optimization. A null result is a publishable answer and closes the item.

Levers, in the order the plan expected them to pay:

1. **fp16** -- already the default (`_instantiate_realesrgan_upsampler` takes
   `fp32: bool = False`), so the published 3.80 fps is *already* the
   half-precision number. Measured here anyway, because "we already do this"
   is only credible with the fp32 comparison beside it.
2. **Tiling** -- `tile=0` processes the frame whole; `tile=400` splits it. At
   640x360 a 400px tile is nearly the whole frame, so the two should be close.
3. Batching frames would be the remaining lever and is NOT implemented here:
   `RealESRGANer.enhance` takes one image, so batching means changing the
   upstream inference path rather than passing a flag.

Arms are interleaved and repeated, so slow drift in GPU contention (this is a
shared host) hits every arm equally -- the ratio is the measurement, not the
absolute fps.

    python tools/bench_realesrgan.py --frames 24 --repeats 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

JND_PSNR = 0.5  # dB, the threshold used throughout the paper


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()

    # presley first, torch second, always: importing torch before sqlite3 is
    # fatal on this host (CXXABI_1.3.15), and presley/__init__ is what pulls
    # sqlite3 in at the right moment. See src/presley/__init__.py.
    from presley.restoration import _instantiate_realesrgan_upsampler

    import torch

    if not torch.cuda.is_available():
        print("error: no CUDA device; a throughput benchmark on CPU is meaningless",
              file=sys.stderr)
        return 1
    device = torch.device("cuda")

    # Deterministic synthetic content: the benchmark measures throughput, and
    # using real frames would add NFS load time to every arm equally but noisily.
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    frames = [np.clip(base.astype(np.int16) + rng.integers(-8, 9, base.shape), 0, 255)
              .astype(np.uint8) for _ in range(args.frames)]

    arms = {
        "fp16 tile=0 (shipped default)": dict(fp32=False, tile=0),
        "fp32 tile=0": dict(fp32=True, tile=0),
        "fp16 tile=400": dict(fp32=False, tile=400),
    }
    upsamplers = {
        name: _instantiate_realesrgan_upsampler(
            model_name="RealESRGAN_x4plus", device=device, denoise_strength=1.0, **kw)
        for name, kw in arms.items()
    }

    # Warm each arm once; the first call pays cudnn autotuning, not inference.
    outputs = {}
    for name, up in upsamplers.items():
        outputs[name], _ = up.enhance(frames[0], outscale=2)

    timings = {name: [] for name in arms}
    for _ in range(args.repeats):
        for name, up in upsamplers.items():          # interleaved on purpose
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for f in frames:
                up.enhance(f, outscale=2)
            torch.cuda.synchronize()
            timings[name].append(time.perf_counter() - t0)

    ref_name = "fp16 tile=0 (shipped default)"
    ref = statistics.median(timings[ref_name])
    print(f"\n{args.frames} frames at {args.width}x{args.height}, "
          f"{args.repeats} interleaved repeats, median of each arm\n")
    print(f"{'arm':32}{'s/pass':>9}{'fps':>8}{'speedup':>9}{'PSNR vs default':>17}")
    for name in arms:
        med = statistics.median(timings[name])
        q = psnr(outputs[ref_name], outputs[name])
        print(f"{name:32}{med:>9.2f}{args.frames / med:>8.2f}{ref / med:>9.2f}x"
              f"{('identical' if q == float('inf') else f'{q:.2f} dB'):>17}")

    best = max(arms, key=lambda n: ref / statistics.median(timings[n]))
    gain = ref / statistics.median(timings[best])
    print()
    if best == ref_name or gain < 1.2:
        print("VERDICT: no reportable speedup. The pre-registered bar was >= 1.2x;")
        print(f"the best alternative arm reaches {gain:.2f}x over what we already ship.")
        print("fp16 is already the shipped default, so the published throughput")
        print("figure is ALREADY the half-precision number -- the obvious lever was")
        print("pulled before the referee asked. Remaining headroom would need frame")
        print("batching, which is an upstream change to the inference path.")
    else:
        q = psnr(outputs[ref_name], outputs[best])
        ok = q == float("inf") or q > 40.0
        print(f"CANDIDATE: {best} at {gain:.2f}x, output PSNR {q:.2f} dB vs default.")
        print("Report only if the output difference is also within JND." if ok else
              "REJECT: the output changed materially; that is a different restorer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
