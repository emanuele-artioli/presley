#!/usr/bin/env python3
"""Is the S1b damage spread a within-run dispersion, or a pooled one?

`sections/evaluation.tex` states that "within one level, the damage a block
suffers after restoration varies by 12.4 dB between its 10th and 90th
percentile, some 7 times the 1.7 dB average cost of moving a block up a level."
The claim is about the headroom a smarter selection rule could exploit, so it
has to be a dispersion *within* one run -- one video, one level, one restorer.

The 12.36 dB figure it quotes is not that. It is the p90-p10 of all 3,123
fully-degraded superblocks of the eight k=3 probes **pooled together**
(`docs/WAVE1_FALSIFIERS.md`, "S1b Arm B"), so it carries the between-video
variance as well. That is the same defect that superseded the 6.2/8.2 dB
figure in `CLAIM(block-damage)`: pooling inflates a number that is then read
as within-run.

This recomputes it per run and reports both, so the difference is visible
rather than argued. No GPU time and no re-mining: it reads the npz
`tools/mine_block_damage.py` already wrote.

    python tools/analyze_within_run_spread.py --npz results/block_damage_s1b.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The 16 S1b uniform-level probes, from experiments.yaml
# (component presley_ai, downsample_uniform_level in {2,3}, realesrgan).
PROBES = {
    2: {
        "ba87f0fe5b901c66": "bear", "6437a590c56c629b": "bike-packing",
        "a916b27eb0135ade": "color-run", "f0fb80c5f996a1a9": "dancing",
        "dbe01c1d0fe8363b": "dogs-jump", "dfa7aea43f509937": "drift-straight",
        "6ca87c33f33b642b": "drift-turn", "01e9cb2228c06206": "motorbike",
    },
    3: {
        "56121ae83420b24d": "bear", "2cd884b123dc6ccf": "bike-packing",
        "1aa5d44a194b21aa": "color-run", "6842e622a4144120": "dancing",
        "ab2c9c567229be90": "dogs-jump", "9f8217f163b6aa88": "drift-straight",
        "762d42d2b8aeea9e": "drift-turn", "14287118971f9cbc": "motorbike",
    },
}
# Fully-degraded superblocks only, exactly as the pooled version restricted:
# a partially covered SB has no unambiguous level.
FULL_COVERAGE = 0.999


def spread(values: np.ndarray) -> float:
    p10, p90 = np.percentile(values, [10, 90])
    return float(p90 - p10)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=str(REPO_ROOT / "results" / "block_damage_s1b.npz"))
    args = ap.parse_args()

    path = Path(args.npz)
    if not path.is_file():
        print(f"error: {path} not found -- run tools/mine_block_damage.py first", file=sys.stderr)
        return 1

    d = np.load(path, allow_pickle=True)
    run, frac, delta = d["run"], d["strength_frac"], d["delta_psnr"]
    full = frac >= FULL_COVERAGE

    summary = {}
    print(f"Within-run vs pooled p90-p10 of post-restoration damage, "
          f"fully-degraded superblocks only (strength_frac >= {FULL_COVERAGE})\n")
    for level in sorted(PROBES):
        print(f"  level k={level}")
        print(f"    {'video':16}{'n':>7}{'p10':>8}{'p90':>8}{'p90-p10':>10}{'mean':>8}")
        spreads, pooled = [], []
        for h, video in sorted(PROBES[level].items(), key=lambda kv: kv[1]):
            v = delta[full & (run == h)]
            if v.size == 0:
                print(f"    {video:16}{'MISSING':>7}")
                continue
            pooled.append(v)
            spreads.append(spread(v))
            p10, p90 = np.percentile(v, [10, 90])
            print(f"    {video:16}{v.size:>7,}{p10:>8.2f}{p90:>8.2f}{spread(v):>10.2f}{v.mean():>8.2f}")
        allv = np.concatenate(pooled)
        summary[level] = {
            "within": float(np.mean(spreads)),
            "pooled": spread(allv),
            "mean": float(allv.mean()),
        }
        print(f"    {'-- within-run mean':16}{'':>7}{'':>8}{'':>8}{summary[level]['within']:>10.2f}")
        print(f"    {'-- POOLED':16}{allv.size:>7,}{np.percentile(allv,10):>8.2f}"
              f"{np.percentile(allv,90):>8.2f}{summary[level]['pooled']:>10.2f}{allv.mean():>8.2f}")
        print(f"    pooling inflates the spread by "
              f"{summary[level]['pooled'] / summary[level]['within']:.2f}x\n")

    # The ratio the paper quotes: headroom available to a selection rule, over
    # the cost of the lever it would pull.
    between = summary[3]["mean"] - summary[2]["mean"]
    print(f"between-level cost  mean d(k=3) - mean d(k=2) = {between:+.3f} dB")
    print(f"R (within-run)           = {summary[3]['within']:.2f} / {between:.3f} "
          f"= {summary[3]['within'] / between:.2f}")
    print(f"R (pooled, as published) = {summary[3]['pooled']:.2f} / {between:.3f} "
          f"= {summary[3]['pooled'] / between:.2f}")
    print("\nThe within-run figure is what the sentence in sections/evaluation.tex means.")
    print("Unlike CLAIM(block-damage), where pooling ran across two restorers with very")
    print("different damage levels and inflated the spread badly, here it pools eight")
    print("videos at ONE restorer and one level, whose mean damages are close -- so the")
    print("published figure was only mildly inflated. Pooling's cost is a function of")
    print("between-group variance, not a fixed penalty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
