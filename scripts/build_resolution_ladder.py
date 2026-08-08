#!/usr/bin/env python3
"""Emit the run-file for the bitrate-vs-resolution ladder (W1f).

The article's rate results are almost all at 640x360, which makes "does the
bitrate saving survive at higher resolution?" a genuine open question rather
than a formality. This builds the run-file that answers it.

**Block size scales with resolution** (8/12/16/24 for 360p/540p/720p/1080p) so
every rung keeps an 80x45 block grid. Holding the block size fixed instead
would shrink each block to a ninth of its former share of the frame between the
bottom and top rung, changing selection granularity, side-channel size and the
codec's view of the degraded region all at once -- so a difference measured
that way would not be attributable to resolution. See
`prepare_resolution_ladder.py`, which warms the matching cache.

**The QP ladder is held fixed across resolutions.** This is sound because
BD-rate is computed *within* a (video, resolution) cell and never compares
quality across cells. It does **not** mean the rungs sit in matched operating
regimes -- measured on this corpus they do not, and the direction matters:
at fixed QP the higher rungs run at roughly half the bits per pixel (QP 43:
0.104 / 0.061 / 0.049 bpp at 360p / 720p / 1080p), i.e. *more* starved, which
is the regime that favours the method. Any growth of the saving with resolution
is therefore confounded with that drop and must be reported as such.

That was established rather than assumed, and it corrected the first version of
this script: the baseline arm runs on its own first, and `--check` reports the
realized regime before any GPU time goes into the restoration arm. The original
gate compared absolute PSNR across resolutions, which cannot establish anything
-- each rung is scored against its own downscaled reference. See `check`.

Two arms, matched in everything but the method under test:

  * `baselines`   -- pristine SVT-AV1 at the same QP, the rate anchor.
  * `presley_ai`  -- downsample + Real-ESRGAN, the canonical reported recipe
                     (shrink_amount 0.25, fg_protect, composite_output).

Usage:
    python scripts/build_resolution_ladder.py --arm baselines -o config/w1f_ladder_baselines.yaml
    python scripts/build_resolution_ladder.py --arm presley   -o config/w1f_ladder_presley.yaml
    python scripts/build_resolution_ladder.py --check results/
"""
from __future__ import annotations

import argparse
import sys

import yaml

VIDEOS = ("bear", "camel", "dog", "pigs", "tennis", "train")

# (width, height, block_size) -- every rung is an 80x45 block grid.
LADDER = (
    (640, 360, 8),
    (1280, 720, 16),
    (1920, 1080, 24),
)

# Four rungs spanning the starved range the article reports for SVT-AV1. Held
# fixed across resolutions (see the module docstring).
QPS = (43, 50, 55, 60)

CODEC = "svtav1"
PRESET = "8"


def baseline_entry(video: str, width: int, height: int, qp: int) -> dict:
    return {
        "component": "baselines",
        "video": video,
        "width": width,
        "height": height,
        "codec": CODEC,
        "codec_params": {"preset": PRESET, "qp": qp},
        "target_bitrate": 0,
    }


def presley_entry(video: str, width: int, height: int, block_size: int, qp: int) -> dict:
    """The canonical downsample+Real-ESRGAN recipe, resolution-parameterized."""
    return {
        "component": "presley_ai",
        "video": video,
        "width": width,
        "height": height,
        "block_size": block_size,
        "codec": CODEC,
        "codec_params": {"preset": PRESET, "qp": qp},
        "alpha": 0.5,
        "beta": 0.5,
        "shrink_amount": 0.25,
        "fg_protect": True,
        "composite_output": True,
        "degradation": "downsample",
        "restorer": "realesrgan",
        "restorer_params": {},
        "target_bitrate": 0,
    }


def build(arm: str) -> list:
    entries = []
    for video in VIDEOS:
        for width, height, block_size in LADDER:
            for qp in QPS:
                if arm == "baselines":
                    entries.append(baseline_entry(video, width, height, qp))
                else:
                    entries.append(presley_entry(video, width, height, block_size, qp))
    return entries


def check(results_dir: str) -> int:
    """Report the operating regime the baseline arm actually landed in.

    **Read the bits-per-pixel column, not the PSNR column**, and the reason is
    worth stating because the first version of this gate got it wrong.

    Absolute PSNR is not comparable across the rungs. Each resolution is scored
    against *its own* reference -- the source downscaled to that size -- so a
    1080p run at 35 dB and a 360p run at 31 dB are not measuring distance from
    the same thing. Downscaling concentrates detail: at 360p each pixel carries
    far more unique information, so the same QP scores worse against a harder
    reference. Reading that as "1080p is not starved" confuses the difficulty
    of the reference with the generosity of the bit budget.

    Bits per pixel is comparable, and it says the opposite. Measured on this
    corpus at fixed QP, the higher rungs are roughly TWICE as starved per pixel
    as 360p (QP 43: 0.104 / 0.061 / 0.049 bpp at 360p / 720p / 1080p).

    Two consequences, both of which belong in the article rather than in a
    footnote:

    * The fixed-QP ladder is sound for this experiment, because BD-rate is
      computed *within* a (video, resolution) cell and never compares PSNR
      across cells.
    * The regimes are NOT matched, and the mismatch runs in the direction that
      flatters the method: PRESLEY pays off when the codec is bit-starved, and
      the higher rungs are more starved. So if the saving grows with
      resolution, that growth is confounded with the bpp drop and may not be
      attributed to resolution alone.
    """
    import presley  # noqa: F401  sqlite3-before-torch
    from presley import db as _db
    from presley.runner import compute_experiment_hash

    rows = {}
    for width, height, _ in LADDER:
        for video in VIDEOS:
            for qp in QPS:
                cfg = baseline_entry(video, width, height, qp)
                data = _db.load_run(results_dir, compute_experiment_hash(cfg))
                if data is None:
                    continue
                psnr = ((data.get("metrics") or {}).get("overall") or {}).get("psnr_mean")
                rate = data.get("actual_bitrate_bps")
                fps = data.get("video_framerate")
                bpp = rate / (width * height * fps) if (rate and fps) else None
                rows.setdefault((width, height, qp), []).append((psnr, rate, bpp))

    if not rows:
        print("no baseline results yet -- run the baseline arm first")
        return 1

    print(f"{'resolution':>11} {'QP':>4} {'PSNR':>7} {'kbps':>9} {'bits/pixel':>11}  n")
    for (width, height, qp), vals in sorted(rows.items()):
        psnrs = [v[0] for v in vals if v[0]]
        rates = [v[1] / 1000 for v in vals if v[1]]
        bpps = [v[2] for v in vals if v[2]]
        if not psnrs:
            print(f"{width}x{height:<5} {qp:>4} {'--':>7} {'--':>9} {'--':>11}  "
                  f"{len(vals)} (no metrics yet)")
            continue
        print(f"{width}x{height:<5} {qp:>4} {sum(psnrs)/len(psnrs):>7.2f} "
              f"{sum(rates)/len(rates):>9.1f} {sum(bpps)/len(bpps):>11.4f}  {len(psnrs)}")

    missing = sum(1 for _, _, _ in
                  ((v, w, q) for (w, h, q), vals in rows.items() for v in vals if v[0] is None))
    if missing:
        print(f"\n{missing} run(s) have no metrics yet -- evaluation still pending.")
    print("\nCompare rungs on bits/pixel, never on PSNR (see this function's docstring).\n"
          "Lower bpp = more starved. If bpp FALLS as resolution rises, the higher rungs\n"
          "sit in the regime that favours the method, and any growth of the saving with\n"
          "resolution is confounded with that -- report it, do not attribute it to\n"
          "resolution alone.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("baselines", "presley"))
    ap.add_argument("-o", "--out")
    ap.add_argument("--check", metavar="RESULTS_DIR",
                    help="report realized rate/quality span per resolution")
    args = ap.parse_args()

    if args.check:
        return check(args.check)
    if not args.arm or not args.out:
        ap.error("--arm and -o are required unless --check is given")

    entries = build(args.arm)
    with open(args.out, "w", encoding="utf-8") as fh:
        # Let the YAML writer escape: hand-quoting corrupted experiments.yaml once.
        yaml.safe_dump({"experiments": entries}, fh, sort_keys=True, default_flow_style=False)
    print(f"{args.out}: {len(entries)} entries "
          f"({len(VIDEOS)} videos x {len(LADDER)} resolutions x {len(QPS)} QPs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
