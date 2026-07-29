#!/usr/bin/env python3
"""Pick a maximally-diverse probe subset of videos by clustering content attributes.

Two facts from RESEARCH_LOG drive this. First, **no single scalar content
attribute explains the win/loss split** -- the pre-registered hypotheses
(hole_churn, bg_texture) both collapsed to noise out-of-sample at n=18, and the
best of 7 attributes against a max-FG-penalty target reached only rho=-0.40. So
hand-picking "one fast-camera, one slow-camera" video presumes an axis the data
says is not there. Second, a 6-video finding failed to replicate at n=18, so
single-video conclusions are how this project has previously been misled.

The defensible alternative is to treat the attribute vector as a whole: cluster
videos in standardized attribute space and take each cluster's medoid. Medoids
are real videos (unlike centroids), and the assignment radius reports honestly
how well the chosen subset covers the space.

Reads scratch/video_attributes.csv from scripts/audit_videos.py -- re-run that
first if videos have been added. Optionally joins the results index so the
report shows how much prior work each candidate already carries, since a medoid
with existing runs is cheaper to build on.

Usage:
    python tools/select_probe_videos.py -k 4
    python tools/select_probe_videos.py -k 5 --db results/index.db
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

REPO_ROOT = Path(__file__).resolve().parent.parent

# The content attributes audit_videos.py emits. 'frames' is excluded: clip
# length is a property of how we cut the dataset, not of the content.
ATTRS = ["fg_frac", "fg_frac_std", "blobs", "hole_churn", "motion_all",
         "motion_fg", "motion_bg", "bg_texture", "bg_temporal_res"]


def load_attributes(path: Path) -> tuple[list[str], np.ndarray]:
    rows = [r for r in csv.DictReader(path.open()) if r.get("status") == "ok"]
    if not rows:
        raise SystemExit(f"error: no rows with status=ok in {path}")
    missing = [a for a in ATTRS if a not in rows[0]]
    if missing:
        raise SystemExit(f"error: {path} is missing columns {missing} -- re-run scripts/audit_videos.py")

    videos = [r["video"] for r in rows]
    x = np.array([[float(r[a]) for a in ATTRS] for r in rows], dtype=np.float64)
    return videos, x


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score per attribute; attributes are on wildly different scales
    (bg_texture ~96, hole_churn ~0.005) so raw distances would be dominated by
    whichever happens to have the largest units."""
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - x.mean(axis=0)) / sd


def pick_medoids(z: np.ndarray, k: int) -> tuple[np.ndarray, list[int]]:
    """Ward-linkage clustering into k groups; medoid = the member minimizing
    total distance to its own cluster.

    NOTE: scipy's 'maxclust' criterion yields *at most* k clusters -- ties and
    coincident points can collapse it to fewer. It therefore returns as many
    medoids as clusters actually formed, which may be < k; callers must check
    rather than assume, or they would screen on a smaller subset than they
    asked for without noticing.
    """
    d = squareform(pdist(z))
    labels = fcluster(linkage(pdist(z), method="ward"), t=k, criterion="maxclust")

    medoids = []
    for c in sorted(set(labels)):
        members = np.flatnonzero(labels == c)
        within = d[np.ix_(members, members)].sum(axis=1)
        medoids.append(int(members[np.argmin(within)]))
    return labels, medoids


def run_counts(db: Path | None) -> dict[str, int]:
    if not db or not db.is_file():
        return {}
    conn = sqlite3.connect(db)
    counts = dict(conn.execute(
        "select config_video, count(*) from results group by 1").fetchall())
    conn.close()
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attributes", default=str(REPO_ROOT / "scratch" / "video_attributes.csv"))
    ap.add_argument("-k", type=int, default=4, help="number of probe videos (default 4)")
    ap.add_argument("--db", default=str(REPO_ROOT / "results" / "index.db"))
    args = ap.parse_args()

    videos, x = load_attributes(Path(args.attributes))
    if args.k < 1 or args.k > len(videos):
        raise SystemExit(f"error: -k must be between 1 and {len(videos)}")

    z = standardize(x)
    labels, medoids = pick_medoids(z, args.k)
    d = squareform(pdist(z))
    counts = run_counts(Path(args.db))

    print(f"{len(videos)} videos, {len(ATTRS)} attributes, k={args.k}\n")
    if len(medoids) < args.k:
        print(f"warning: only {len(medoids)} distinct clusters formed for k={args.k} "
              f"(coincident videos cannot be split further) -- the probe set is "
              f"smaller than requested\n", file=sys.stderr)
    print(f"{'probe video':>18} {'cluster':>8} {'members':>8} {'radius':>7} {'runs':>6}")
    for m in medoids:
        c = labels[m]
        members = np.flatnonzero(labels == c)
        radius = d[m, members].max()
        print(f"{videos[m]:>18} {c:>8} {len(members):>8} {radius:>7.2f} {counts.get(videos[m], 0):>6}")

    # How well the subset covers every video, including ones it did not pick.
    worst = max(min(d[i, m] for m in medoids) for i in range(len(videos)))
    print(f"\nworst-case distance from any video to its nearest probe: {worst:.2f} sd")
    print("(each attribute is z-scored, so this is in pooled standard deviations)")

    print("\ncluster membership:")
    for c in sorted(set(labels)):
        members = [videos[i] for i in np.flatnonzero(labels == c)]
        star = videos[[m for m in medoids if labels[m] == c][0]]
        print(f"  {c}: " + ", ".join(f"*{v}*" if v == star else v for v in members))

    print("\nProbe videos are marked *. Screen directional questions on one of")
    print("these; any conclusion that becomes a paper CLAIM re-runs on all of")
    print("them -- the n=6 -> n=18 collapse is why.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
