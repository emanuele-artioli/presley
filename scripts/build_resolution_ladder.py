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

**The QP ladder is held fixed across resolutions, deliberately.** Fixed QP
targets a roughly constant quality per pixel, so the same QP leaves the encoder
comparably starved at every resolution while the absolute bitrate scales with
pixel count -- which is exactly the comparison a BD-rate per (video,
resolution) wants. Recalibrating QP per rung would instead hold *bitrate*
roughly fixed and let quality vary, which answers a different question. The
assumption that the ladder still spans a starved range at 1080p is checked
rather than asserted: the baseline arm is emitted first and run on its own, and
`--check` reports the realized bitrate and quality span per resolution before
any GPU time is spent on the restoration arm.

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
    """Report the realized rate/quality span per resolution for the baseline arm.

    This is the gate the docstring promises: if the top rung is not actually
    starved at 1080p, the ladder measures a comfortable-bitrate regime where
    the method is already known to lose, and the result would be about the
    operating point rather than about resolution.
    """
    import presley  # noqa: F401  sqlite3-before-torch
    from presley import db as _db

    rows = {}
    for width, height, _ in LADDER:
        for video in VIDEOS:
            for qp in QPS:
                cfg = baseline_entry(video, width, height, qp)
                h = _db.compute_experiment_hash(cfg)
                data = _db.load_run(results_dir, h)
                if data is None:
                    continue
                psnr = ((data.get("metrics") or {}).get("overall") or {}).get("psnr_mean")
                rows.setdefault((width, height), []).append(
                    (video, qp, data.get("actual_bitrate_bps"), psnr))

    if not rows:
        print("no baseline results yet -- run the baseline arm first")
        return 1

    print(f"{'resolution':>12} {'n':>4} {'kbps min':>10} {'kbps max':>10} "
          f"{'PSNR min':>9} {'PSNR max':>9}")
    for (width, height), vals in sorted(rows.items()):
        rates = [v[2] / 1000 for v in vals if v[2]]
        psnrs = [v[3] for v in vals if v[3]]
        if not rates or not psnrs:
            continue
        print(f"{width}x{height:>6} {len(vals):>4} {min(rates):>10.1f} {max(rates):>10.1f} "
              f"{min(psnrs):>9.2f} {max(psnrs):>9.2f}")
    print("\nStarved means the baseline is visibly quality-limited at the top of the\n"
          "ladder. If PSNR max at 1080p sits far above 360p's, the same QP is not\n"
          "leaving the encoder comparably starved and the rungs need revisiting.")
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
