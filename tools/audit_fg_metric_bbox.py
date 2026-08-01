#!/usr/bin/env python3
"""Re-derive the union-bbox vs true-masked FG-DISTS audit, with provenance.

The rebuttal to the "no subjective study" review comment leans on a fact about
our own tooling: the metric previously reported as *foreground* DISTS was a
union-bounding-box crop, which on some clips is most of the frame (`india` 100%,
`tennis` 58.6% against a true foreground of 4.0%). Correcting it to true masked
pooling moved results **against the direction that would have flattered us** --
the defective metric had been understating our own foreground result.

That audit was run once and recorded in the research log, but the old
`foreground.dists_mean` keys were deleted from `results/` by
`drop_unionbbox_keys` when the metric was corrected. So the numbers currently
have no reproducible provenance, and by this repo's own rule they may not appear
in reviewer-visible text. This recomputes the bbox variant from the stored
videos and compares it against the stored masked `dists_fg`, so the comparison
carries hashes again.

No re-encoding and nothing is written back into `results/`: this reads decoded
frames and reports.

    python tools/audit_fg_metric_bbox.py --limit 0        # whole corpus
    python tools/audit_fg_metric_bbox.py --video bear
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from presley import db as _db  # noqa: E402
from presley.encode_utils import load_frames_from_video  # noqa: E402
from presley.evaluation.cache import _get_masks_cached, _get_refs_cached  # noqa: E402
from presley.evaluation.masked import _fg_union_bbox  # noqa: E402
from presley.evaluation.perceptual import calculate_dists  # noqa: E402

# DISTS is lower-is-better, so a group's winner is its minimum.
BETTER = min


def bbox_dists(refs, decs, masks, width, height, device):
    """The OLD 'FG-DISTS': whole-frame DISTS computed on a union-bbox crop."""
    box = _fg_union_bbox(masks, width, height)
    if box is None:
        return None, None
    y1, y2, x1, x2 = box
    area = (y2 - y1) * (x2 - x1) / float(width * height)
    vals = calculate_dists([r[y1:y2, x1:x2] for r in refs],
                           [d[y1:y2, x1:x2] for d in decs], device)
    return float(np.mean(vals)), area


def collect(results_dir, cache_dir, dataset_dir, only_video, limit):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    entries = sorted(e for e in os.listdir(results_dir)
                     if not e.startswith(("_", ".")))
    for entry in entries:
        if limit and len(rows) >= limit:
            break
        try:
            data = _db.load_run(results_dir, entry)
        except (ValueError, OSError):
            continue
        if not data or data.get("invariant_failures"):
            continue  # never audit on a non-citable run
        metrics = data.get("metrics") or {}
        masked = (metrics.get("foreground") or {}).get("dists_fg")
        if masked is None:
            continue
        cfg = data["config"]
        video = cfg.get("video")
        if only_video and video != only_video:
            continue
        out = data.get("output_video")
        if not out or not os.path.exists(out):
            continue

        width, height = cfg["width"], cfg["height"]
        _, refs, _ = _get_refs_cached(video, width, height, dataset_dir, cache_dir)
        ref_dir = os.path.join(cache_dir, f"{video}_{width}x{height}", "reference_frames")
        ufo = _get_masks_cached(video, width, height, cfg.get("block_size", 8), ref_dir, cache_dir)
        decs = load_frames_from_video(out)
        n = min(len(refs), len(decs), len(ufo))
        if n == 0:
            continue
        masks = [ufo[i] > 127 for i in range(n)]
        bbox, area = bbox_dists(refs[:n], decs[:n], masks, width, height, device)
        if bbox is None:
            continue
        true_fg = float(np.mean([m.mean() for m in masks]))
        rows.append({
            "hash": entry, "video": video, "component": cfg.get("component"),
            "restorer": cfg.get("restorer") or cfg.get("inpainter") or "",
            "codec": cfg.get("codec"), "qp": (cfg.get("codec_params") or {}).get("qp"),
            "masked": float(masked), "bbox": bbox,
            "bbox_area_frac": area, "true_fg_frac": true_fg,
            "bitrate": data.get("actual_bitrate_bps"),
        })
        print(f"  {entry} {video:14} masked={masked:.4f} bbox={bbox:.4f} "
              f"bbox_area={area:.1%} true_fg={true_fg:.1%}", flush=True)
    return rows


def report(rows):
    if not rows:
        print("no auditable runs found", file=sys.stderr)
        return 1

    print(f"\n=== per video: is the masked metric WORSE (higher) than the bbox one? ===")
    print(f"{'video':16}{'n':>4}{'masked':>9}{'bbox':>9}{'delta%':>9}{'bbox area':>11}{'true FG':>9}")
    worse = 0
    per_video = collections.defaultdict(list)
    for r in rows:
        per_video[r["video"]].append(r)
    for video in sorted(per_video):
        rs = per_video[video]
        m = float(np.mean([r["masked"] for r in rs]))
        b = float(np.mean([r["bbox"] for r in rs]))
        worse += m > b
        print(f"{video:16}{len(rs):>4}{m:>9.4f}{b:>9.4f}{100*(m-b)/b:>8.1f}%"
              f"{np.mean([r['bbox_area_frac'] for r in rs]):>10.1%}"
              f"{np.mean([r['true_fg_frac'] for r in rs]):>9.1%}")
    print(f"\nmasked worse than bbox on {worse}/{len(per_video)} videos")

    # Winner flips, over groups matched on (video, codec, qp) with >1 component.
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["video"], r["codec"], r["qp"])].append(r)
    matched = {k: v for k, v in groups.items()
               if len({r["component"] for r in v}) > 1}
    flips, away = 0, 0
    print(f"\n=== winner flips across {len(matched)} matched groups ===")
    for key, rs in sorted(matched.items(), key=str):
        win_bbox = BETTER(rs, key=lambda r: r["bbox"])
        win_mask = BETTER(rs, key=lambda r: r["masked"])
        if win_bbox["component"] == win_mask["component"]:
            continue
        flips += 1
        moved_away = win_bbox["component"] == "baselines" and win_mask["component"] != "baselines"
        away += moved_away
        print(f"  {str(key):40} {win_bbox['component']:11} -> {win_mask['component']:11}"
              f"{'   (away from baselines)' if moved_away else ''}")
    print(f"\n{flips} of {len(matched)} matched groups change winner; "
          f"{away} of those move AWAY from baselines")
    print("\nA flip away from `baselines` means the defective bbox metric had been "
          "crediting\nthe pristine baseline for background the bridge degrades by "
          "design -- i.e. the\ncorrection costs us nothing and the honest metric is "
          "the stronger one.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    ap.add_argument("--dataset-dir", default=str(REPO_ROOT / "dataset"))
    ap.add_argument("--video", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    rows = collect(args.results_dir, args.cache_dir, args.dataset_dir,
                   args.video, args.limit)
    return report(rows)


if __name__ == "__main__":
    sys.exit(main())
