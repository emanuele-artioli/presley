#!/usr/bin/env python3
"""Per-video attribute audit for PRESLEY's explanatory-axis question.

Writes one CSV row per video describing the content properties hypothesised to
separate the videos where the bridge method frees bits from the ones where it
does not. The 2026-07-11 selection of india/tennis did this by hand and the
table was never persisted; this script is the reproducible replacement.

Attributes (all at the given resolution, over the cached reference frames and
UFO masks -- run the experiments first so the cache exists):

  fg_frac          mean fraction of pixels inside the UFO foreground mask
  fg_frac_std      its per-frame variability (a proxy for FG scale change)
  blobs            mean number of connected FG components >= min_blob_area
                   (fragmentation: many small blobs = scattered holes)
  hole_churn       mean fraction of BG pixels whose selected/unselected state
                   would flip between consecutive frames, approximated by the
                   frame-to-frame change of the FG mask's complement --
                   the axis the bmx-trees boundary case implicated
  motion_all       mean Farneback optical-flow magnitude, whole frame (px/frame)
  motion_fg        same, restricted to FG pixels
  motion_bg        same, restricted to BG pixels  (BG motion is what the
                   blackout/freeze transport has to re-encode)
  bg_texture       mean Sobel gradient magnitude in BG (BG spatial complexity:
                   cheap-to-code flat BG should benefit least from removal)
  bg_temporal_res  mean absolute inter-frame difference in BG (how much a
                   frozen/blacked-out BG actually differs from the real one)

Usage:
  python scripts/audit_videos.py --out scratch/video_attributes.csv
  python scripts/audit_videos.py --videos bear camel --width 640 --height 360
"""
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


def _load_frames(d: Path, limit=None):
    files = sorted(d.glob("*.png"))
    if limit:
        files = files[:limit]
    return [cv2.imread(str(f)) for f in files]


def _load_masks(d: Path, limit=None):
    files = sorted(d.glob("*.png"))
    if limit:
        files = files[:limit]
    return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]


def audit_video(video: str, width: int, height: int, cache_dir: str,
                min_blob_area: int = 64) -> dict:
    key = Path(cache_dir) / f"{video}_{width}x{height}"
    refs_dir, masks_dir = key / "reference_frames", key / "ufo_masks"
    if not refs_dir.is_dir() or not masks_dir.is_dir():
        return {"video": video, "status": "no cache (run an experiment first)"}

    frames, masks = _load_frames(refs_dir), _load_masks(masks_dir)
    n = min(len(frames), len(masks))
    if n < 2:
        return {"video": video, "status": f"only {n} cached frames"}
    frames, masks = frames[:n], masks[:n]

    fg = [m > 127 for m in masks]
    fg_frac = np.array([f.mean() for f in fg])

    blobs = []
    for f in fg:
        cnt, _, stats, _ = cv2.connectedComponentsWithStats(f.astype(np.uint8), 8)
        # label 0 is background; count only components above the noise floor
        blobs.append(int((stats[1:, cv2.CC_STAT_AREA] >= min_blob_area).sum()) if cnt > 1 else 0)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    mot_all, mot_fg, mot_bg, churn, bg_tempres = [], [], [], [], []
    for i in range(n - 1):
        flow = cv2.calcOpticalFlowFarneback(grays[i], grays[i + 1], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        f_cur, f_nxt = fg[i], fg[i + 1]
        bg_cur = ~f_cur
        mot_all.append(mag.mean())
        mot_fg.append(mag[f_cur].mean() if f_cur.any() else np.nan)
        mot_bg.append(mag[bg_cur].mean() if bg_cur.any() else np.nan)
        # mask instability: pixels changing FG/BG membership between frames
        churn.append(np.logical_xor(f_cur, f_nxt).mean())
        diff = cv2.absdiff(grays[i], grays[i + 1])
        bg_tempres.append(diff[bg_cur].mean() if bg_cur.any() else np.nan)

    bg_tex = []
    for g, f in zip(grays, fg):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        m = np.hypot(gx, gy)
        bg = ~f
        bg_tex.append(m[bg].mean() if bg.any() else np.nan)

    def f2(x):
        return round(float(x), 4)

    return {
        "video": video, "status": "ok", "frames": n,
        "fg_frac": f2(fg_frac.mean()), "fg_frac_std": f2(fg_frac.std()),
        "blobs": f2(np.mean(blobs)),
        "hole_churn": f2(np.mean(churn)),
        "motion_all": f2(np.nanmean(mot_all)),
        "motion_fg": f2(np.nanmean(mot_fg)),
        "motion_bg": f2(np.nanmean(mot_bg)),
        "bg_texture": f2(np.nanmean(bg_tex)),
        "bg_temporal_res": f2(np.nanmean(bg_tempres)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None,
                    help="default: every video with a cache dir at this resolution")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="scratch/video_attributes.csv")
    args = ap.parse_args()

    videos = args.videos
    if not videos:
        suffix = f"_{args.width}x{args.height}"
        videos = sorted(d.name[: -len(suffix)] for d in Path(args.cache_dir).iterdir()
                        if d.is_dir() and d.name.endswith(suffix))
    rows = []
    for v in videos:
        r = audit_video(v, args.width, args.height, args.cache_dir)
        rows.append(r)
        print(" ".join(f"{k}={r[k]}" for k in r))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["video", "status", "frames", "fg_frac", "fg_frac_std", "blobs", "hole_churn",
            "motion_all", "motion_fg", "motion_bg", "bg_texture", "bg_temporal_res"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
