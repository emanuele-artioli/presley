#!/usr/bin/env python3
"""Warm the preprocessing cache for the bitrate-vs-resolution ladder (W1f).

The ladder asks one question -- does the bitrate saving survive at higher
resolution -- so everything except resolution has to be held constant, and the
non-obvious part is that **block size is not "everything else"**.

At 640x360 with block size 8 the importance map is an 80x45 grid. Holding the
block size fixed while raising the resolution shrinks each block to a ninth of
the frame area it covered before, which changes the selection granularity, the
side-channel size and the codec's view of the degraded region all at once --
so a difference measured that way is not attributable to resolution. Holding
the *grid* fixed instead is what isolates the axis under test:

    640x360  / bs 8   -> 80x45
    960x540  / bs 12  -> 80x45
    1280x720 / bs 16  -> 80x45
    1920x1080/ bs 24  -> 80x45

(The 1920x1080 bs24 entries already in `cache/` say an earlier session reached
the same conclusion.) The alternative reading -- fixed block size, varying
grid -- is a different experiment and is scoped out explicitly in the article
rather than silently averaged in.

Every source clip here is natively 1920x1080 or larger, so each rung is a
genuine downscale from the source rather than an upscale of a 360p rendition.
`india` and `color-run` are excluded for exactly that reason: they are 1280x720
natively and cannot honestly supply the top rung.

Nothing is computed twice: reference frames, EVCA complexity and UFO masks are
all cached on disk, so this script is re-entrant and safe to resume after an
SSH drop. It computes no experiment and writes nothing under `results/`.

Usage:
    python scripts/prepare_resolution_ladder.py --dry-run
    python scripts/prepare_resolution_ladder.py
"""
from __future__ import annotations

# `presley` first: it imports sqlite3 before torch, which is load-bearing on
# this host (see tests/test_import_order.py).
import presley  # noqa: F401  isort:skip

import argparse
import os
import sys
import time

from presley.preprocessing import get_reference_frames, get_removability_scores

# Six clips, all natively >=1920x1080. bear/camel/dog/pigs carry the ladder
# results the article already reports, so the resolution result attaches to the
# same content rather than to a fresh set; tennis and train add camera motion.
VIDEOS = ("bear", "camel", "dog", "pigs", "tennis", "train")

# (width, height, block_size) -- every rung is an 80x45 block grid.
LADDER = (
    (640, 360, 8),
    (960, 540, 12),
    (1280, 720, 16),
    (1920, 1080, 24),
)

# The operating point every reported presley_ai result uses.
ALPHA = 0.5
BETA = 0.5
MASK_SOURCE = "ufo"

# A silent hour is indistinguishable from a hang, so progress is printed per
# cell and the cadence is asserted by how small a cell is (minutes, not hours).
LOG_EVERY_SECONDS = 600


def cell_is_warm(video: str, width: int, height: int, block_size: int, cache_dir: str) -> bool:
    """True when this cell's removability scores are already on disk."""
    key_dir = os.path.join(cache_dir, f"{video}_{width}x{height}_bs{block_size}")
    return os.path.exists(os.path.join(key_dir, f"removability_a{ALPHA:.2f}_b{BETA:.2f}.npy"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--dry-run", action="store_true",
                    help="report which cells are cold, compute nothing")
    args = ap.parse_args()

    cells = [(v, w, h, bs) for v in VIDEOS for (w, h, bs) in LADDER]
    cold = [c for c in cells if not cell_is_warm(*c, cache_dir=args.cache_dir)]

    print(f"{len(cells)} cells, {len(cold)} cold, {len(cells) - len(cold)} already cached")
    for video, w, h, bs in cold:
        print(f"  cold: {video} {w}x{h} bs{bs}")
    if args.dry_run:
        return 0
    if not cold:
        print("nothing to do")
        return 0

    started = time.time()
    last_log = started
    for n, (video, width, height, block_size) in enumerate(cold, 1):
        t0 = time.time()
        print(f"[{n}/{len(cold)}] {video} {width}x{height} bs{block_size} ...", flush=True)
        try:
            # Frames first: EVCA and UFO both read the extracted reference
            # frames, and doing it explicitly makes the failure legible when a
            # clip's source is missing rather than surfacing inside EVCA.
            get_reference_frames(video, width, height, args.dataset_dir, args.cache_dir)
            get_removability_scores(video, width, height, block_size,
                                    ALPHA, BETA, args.dataset_dir, args.cache_dir,
                                    mask_source=MASK_SOURCE)
        except Exception as exc:  # one bad cell must not cost the whole warm-up
            print(f"  FAILED {video} {width}x{height} bs{block_size}: {exc!r}", flush=True)
            continue
        now = time.time()
        print(f"  done in {now - t0:.1f}s (elapsed {now - started:.0f}s)", flush=True)
        if now - last_log > LOG_EVERY_SECONDS:
            last_log = now

    remaining = [c for c in cells if not cell_is_warm(*c, cache_dir=args.cache_dir)]
    print(f"\nfinished in {time.time() - started:.0f}s; {len(remaining)} cell(s) still cold")
    for video, w, h, bs in remaining:
        print(f"  STILL COLD: {video} {w}x{h} bs{bs}")
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main())
