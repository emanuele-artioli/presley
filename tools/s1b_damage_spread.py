#!/usr/bin/env python3
"""S1b early-exit check: is there within-level damage spread to exploit?

Reads the npz emitted by tools/mine_block_damage.py, restricted to the S1b
uniform-level probe runs, and computes the one number the pre-registered
decision rule turns on (docs/WAVE1_FALSIFIERS.md, "S1b run log"):

    R = (p90-p10 of delta_psnr within k=3) / (mean delta(k=3) - mean delta(k=2))

R < 1.0, or an absolute within-level p90-p10 below 2.0 dB, kills the graded
direction with zero Arm-B runs: a perfect damage-aware sort would then buy less
than the cost of moving a block up one level, so there is nothing to exploit.

R is computed PER VIDEO and then aggregated, not pooled across videos. Pooling
would inflate the spread with between-video difficulty, which no reassignment
can exploit -- a level reassignment only ever moves blocks within one video.

Usage:
    python tools/s1b_damage_spread.py --damage /path/to/block_damage.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Probe hashes, from `presley-run --dry-run` on the S1b block of experiments.yaml.
# Kept explicit rather than re-derived: mine_block_damage.py does not carry the
# uniform-level key through into its output columns.
LEVEL_OF_RUN = {
    "01e9cb2228c06206": (2, "motorbike"),
    "dfa7aea43f509937": (2, "drift-straight"),
    "6ca87c33f33b642b": (2, "drift-turn"),
    "a916b27eb0135ade": (2, "color-run"),
    "f0fb80c5f996a1a9": (2, "dancing"),
    "dbe01c1d0fe8363b": (2, "dogs-jump"),
    "6437a590c56c629b": (2, "bike-packing"),
    "ba87f0fe5b901c66": (2, "bear"),
    "14287118971f9cbc": (3, "motorbike"),
    "9f8217f163b6aa88": (3, "drift-straight"),
    "762d42d2b8aeea9e": (3, "drift-turn"),
    "1aa5d44a194b21aa": (3, "color-run"),
    "6842e622a4144120": (3, "dancing"),
    "ab2c9c567229be90": (3, "dogs-jump"),
    "2cd884b123dc6ccf": (3, "bike-packing"),
    "56121ae83420b24d": (3, "bear"),
}

# Pre-registered gates. Changing these after seeing a number is exactly what
# the bound-before-believing rule forbids.
R_KILL = 1.0
SPREAD_FLOOR_DB = 2.0
R_ALARM = 30.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--damage", required=True, help="npz from tools/mine_block_damage.py")
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="only superblocks at least this degraded count (default 0.5)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    with np.load(args.damage, allow_pickle=False) as z:
        run = z["run"].astype(str)
        delta = z["delta_psnr"]
        frac = z["strength_frac"]

    keep = np.isin(run, list(LEVEL_OF_RUN)) & (frac >= args.min_frac) & np.isfinite(delta)
    if not keep.any():
        print("error: no S1b probe superblocks found -- was the miner re-run "
              "after indexing the probe results?")
        return 1
    run, delta = run[keep], delta[keep]

    levels = np.array([LEVEL_OF_RUN[r][0] for r in run])
    videos = np.array([LEVEL_OF_RUN[r][1] for r in run])

    per_video = {}
    print(f"{'video':>16} {'n(k2)':>7} {'n(k3)':>7} {'mean2':>7} {'mean3':>7} "
          f"{'gap':>6} {'p90-p10(k3)':>12} {'R':>7}")
    for v in sorted(set(videos.tolist())):
        m = videos == v
        d2, d3 = delta[m & (levels == 2)], delta[m & (levels == 3)]
        if d2.size == 0 or d3.size == 0:
            print(f"{v:>16}  missing one level -- skipped")
            continue
        mean2, mean3 = float(d2.mean()), float(d3.mean())
        gap = mean3 - mean2
        spread3 = float(np.percentile(d3, 90) - np.percentile(d3, 10))
        spread2 = float(np.percentile(d2, 90) - np.percentile(d2, 10))
        r = spread3 / gap if gap > 0 else float("nan")
        per_video[v] = dict(n2=int(d2.size), n3=int(d3.size), mean2=mean2, mean3=mean3,
                            gap=gap, spread_k2=spread2, spread_k3=spread3, R=r)
        print(f"{v:>16} {d2.size:>7,} {d3.size:>7,} {mean2:>7.2f} {mean3:>7.2f} "
              f"{gap:>6.2f} {spread3:>12.2f} {r:>7.2f}")

    if not per_video:
        print("error: no video had both levels")
        return 1

    rs = np.array([p["R"] for p in per_video.values()], dtype=float)
    gaps = np.array([p["gap"] for p in per_video.values()])
    spreads = np.array([p["spread_k3"] for p in per_video.values()])
    m2 = np.array([p["mean2"] for p in per_video.values()])
    m3 = np.array([p["mean3"] for p in per_video.values()])

    print("\n-- pre-registered checks --")
    print(f"mean delta_psnr k=2: {m2.mean():.2f} dB (bound 2-8, alarm <0 or >20)")
    print(f"mean delta_psnr k=3: {m3.mean():.2f} dB (bound 4-14, alarm <0 or >20)")
    print(f"k=3 worse than k=2 on {(gaps > 0).sum()}/{gaps.size} videos; "
          f"median gap {np.median(gaps):.2f} dB (bound >=1.0)")
    print(f"within-k=3 p90-p10: median {np.median(spreads):.2f} dB "
          f"(bound 5-12, kill floor <{SPREAD_FLOOR_DB})")
    print(f"R: median {np.nanmedian(rs):.2f}, per-video {np.round(rs, 2).tolist()}")
    print(f"   (kill if R < {R_KILL}; expected 3-10; ALARM if R > {R_ALARM})")

    r_med = float(np.nanmedian(rs))
    verdict = []
    if not (gaps > 0).all():
        verdict.append("ALARM: k=3 is not worse than k=2 on every video -- level "
                       "may not be reaching the pixels")
    if np.median(spreads) < SPREAD_FLOOR_DB:
        verdict.append("KILL: within-level spread below the 2.0 dB absolute floor")
    if r_med < R_KILL:
        verdict.append(f"KILL: median R = {r_med:.2f} < {R_KILL} -- no headroom for a "
                       "damage-aware reassignment; do not run Arm B")
    if r_med > R_ALARM:
        verdict.append(f"ALARM: median R = {r_med:.2f} > {R_ALARM}")
    if not verdict:
        verdict.append(f"PROCEED: median R = {r_med:.2f} clears the gate -- run the 8 "
                       "Arm-B confirmation runs")
    print()
    for line in verdict:
        print(line)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            dict(per_video=per_video, R_median=r_med, verdict=verdict), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
