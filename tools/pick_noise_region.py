#!/usr/bin/env python
"""One-off helper used to pick the CROPS coordinates in
``tools/noise_mode_decision_analysis.py``.

Scans candidate 160x160 crops (block-aligned, stride 40px) against the
cached UFO masks for a video and reports the crop that is background in the
largest fraction of frames, prioritizing the worst-case (minimum-over-frames)
background fraction so the chosen region is background in *every* frame, not
just on average.

Usage:
    /home/itec/emanuele/.conda/envs/presley/bin/python tools/pick_noise_region.py
"""
import numpy as np
import cv2
from pathlib import Path

CACHE_DIR = "/home/itec/emanuele/presley/cache"


def best_crop(video: str, width: int = 640, height: int = 360, csize: int = 160, stride: int = 40):
    mdir = Path(CACHE_DIR) / f"{video}_{width}x{height}" / "ufo_masks"
    masks = np.array([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(mdir.glob("*.png"))])
    bg = masks < 127  # background = not-foreground
    print(video, "frames", masks.shape[0], "mean bg frac", bg.mean())

    best = None
    for y0 in range(0, height - csize + 1, stride):
        for x0 in range(0, width - csize + 1, stride):
            crop = bg[:, y0:y0 + csize, x0:x0 + csize]
            frac_per_frame = crop.reshape(crop.shape[0], -1).mean(axis=1)
            worst, mean = frac_per_frame.min(), frac_per_frame.mean()
            if best is None or (worst, mean) > (best[0], best[1]):
                best = (worst, mean, y0, x0)
    print(video, "best crop (worst_frac, mean_frac, y0, x0):", best)
    return best


if __name__ == "__main__":
    for v in ("bear", "camel"):
        best_crop(v)
