#!/usr/bin/env python3
"""Open question 0c: why is EVCA near-chance on drift-straight?

`CLAIM(f1-oracle-bits)` records that drift-straight breached its pre-registered
band low -- Spearman rho +0.082 against a 0.30-0.85 band, capture 0.510 against
its own random null of 0.436 -- while the other seven probe videos sit at
+0.61..+0.93. One explanation was already tested and refuted: its marginal-bit
distribution is not flat (CV 0.93, mid-pack) and rho vs CV across the eight
correlates at 0.031.

This reads the eight existing `probe_oracle_bits` runs -- no new GPU time -- and
mechanically applies three further tests, printing what each one settles:

  1. Is rho even distinguishable from zero? (bootstrap CI over superblocks)
  2. Is the PREDICTOR degenerate rather than the target? The refuted test looked
     at the spread of marginal bits; this looks at the spread of EVCA SC.
  3. Does the failure have spatial structure -- specifically, do the frame-edge
     columns cost more bits than EVCA thinks they should? That is the signature
     of camera motion carrying content in and out of frame: content with no
     temporal predictor is expensive to code but is not spatially complex, and
     EVCA's spatial SC cannot see it.

Verdicts are printed as REFUTED / SUPPORTED / UNSETTLED by the tool rather than
left to the reader, in the style of tools/analyze_f1_oracle.py.

    python tools/analyze_drift_straight_0c.py --results-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from presley.components.probe_oracle_bits import _rankdata, _spearman  # noqa: E402

# The eight F1 probe runs, from CLAIM(f1-oracle-bits) in sections/presley.tex.
RUNS = {
    "8925be20182b9967": "motorbike",
    "cdcc94a99f222a90": "drift-straight",
    "5d5a825d96490d1c": "drift-turn",
    "efaa2a75e3b4f05d": "color-run",
    "1fc429d1207c8dbb": "dancing",
    "b906d4cbb03fdbd1": "dogs-jump",
    "86e77a55a5c35e5a": "bike-packing",
    "7faf160138a5053b": "bear",
}
SUBJECT = "drift-straight"
BOOTSTRAP = 4000
SEED = 0
# How many columns at each side of the frame count as "edge". Two of ten, i.e.
# the outer 20% of the width, chosen before looking at the per-column numbers.
EDGE_COLS = 2


def load(results_dir: Path):
    out = {}
    for h, video in RUNS.items():
        path = results_dir / h / "result.json"
        if not path.is_file():
            sys.exit(f"missing {path} -- these are the F1 probe runs; do not re-run them to 'fix' this")
        data = json.loads(path.read_text())
        if data.get("invariant_failures"):
            sys.exit(f"{h} ({video}) is not citable: {data['invariant_failures']}")
        p = data["oracle_bits_probe"]
        out[video] = {
            "sc": np.asarray(p["evca_sc"], dtype=float),
            "mb": np.asarray(p["marginal_bits"], dtype=float),
            "rho": float(p["spearman_sc"]),
            "rho_blended": float(p["spearman_blended"]),
            "grid": tuple(p["sb_grid"]),
        }
    return out


def test_rho_vs_zero(runs) -> bool:
    """True when rho is distinguishable from a random predictor."""
    d = runs[SUBJECT]
    rng = np.random.default_rng(SEED)
    n = d["sc"].size
    boots = np.array([
        _spearman(d["sc"][idx], d["mb"][idx])
        for idx in (rng.integers(0, n, n) for _ in range(BOOTSTRAP))
    ])
    lo, hi = np.percentile(boots[np.isfinite(boots)], [2.5, 97.5])
    excludes_zero = not (lo < 0 < hi)
    print("\n1. Is rho distinguishable from zero at all?")
    print(f"   rho = {d['rho']:+.3f}, bootstrap 95% CI over {n} superblocks "
          f"[{lo:+.3f}, {hi:+.3f}]")
    print("   ->", "the CI EXCLUDES zero: weak but real skill"
          if excludes_zero else
          "the CI INCLUDES zero: on this clip EVCA SC is not distinguishable\n"
          "      from a random predictor. 'Near-chance' is the literal reading, and\n"
          "      any wording that treats +0.082 as a small positive effect is wrong.")
    return excludes_zero


def test_predictor_degeneracy(runs) -> bool:
    """True when a flat EVCA score explains the low rho across videos."""
    videos = list(runs)
    cv = np.array([runs[v]["sc"].std() / runs[v]["sc"].mean() for v in videos])
    rho = np.array([runs[v]["rho"] for v in videos])
    corr = _spearman(cv, rho)
    subject_rank = int((cv < cv[videos.index(SUBJECT)]).sum())
    print("\n2. Is the PREDICTOR degenerate rather than the target?")
    print(f"   {'video':16}{'EVCA SC CV':>12}{'rho':>8}")
    for v in sorted(videos, key=lambda v: runs[v]["sc"].std() / runs[v]["sc"].mean()):
        print(f"   {v:16}{runs[v]['sc'].std() / runs[v]['sc'].mean():>12.3f}{runs[v]['rho']:>8.3f}")
    print(f"   {SUBJECT} has the {'lowest' if subject_rank == 0 else f'#{subject_rank + 1} lowest'} score spread, "
          f"but Spearman(CV, rho) over the eight = {corr:+.3f}")
    supported = corr > 0.6
    print("   ->", "SUPPORTED" if supported else
          "REFUTED: a flat EVCA score does not track low rho across videos.\n"
          "      color-run has nearly as little score spread and rho +0.90. This is\n"
          "      the second dead explanation, alongside the flat-marginal one.")
    return supported


def test_edge_columns(runs) -> bool:
    """True when the subject's frame-edge columns are the standout anomaly."""
    print("\n3. Does the disagreement sit at the frame edges?")
    print(f"   'edge excess' = mean marginal-bit rank minus mean EVCA rank over the")
    print(f"   outer {EDGE_COLS} columns on each side. Positive means the encoder pays")
    print("   for edge content that EVCA considers cheap.")
    excess = {}
    for v, d in runs.items():
        ny, nx = d["grid"]
        rm = _rankdata(d["mb"]).reshape(ny, nx)
        rs = _rankdata(d["sc"]).reshape(ny, nx)
        cols = list(range(EDGE_COLS)) + list(range(nx - EDGE_COLS, nx))
        excess[v] = float(rm[:, cols].mean() - rs[:, cols].mean())
    order = sorted(excess, key=lambda v: -excess[v])
    for v in order:
        print(f"   {v:16}{excess[v]:>+8.1f}{'   <- subject' if v == SUBJECT else ''}")
    others = [excess[v] for v in runs if v != SUBJECT]
    standout = excess[SUBJECT] > max(others)
    print("   ->", f"SUPPORTED: {SUBJECT} is the largest edge excess "
          f"({excess[SUBJECT]:+.1f} vs a next-highest {max(others):+.1f}), consistent with\n"
          "      camera motion carrying content across the frame boundary -- expensive to\n"
          "      code for want of a temporal predictor, invisible to a spatial complexity\n"
          "      score. This is a CANDIDATE MECHANISM, not a demonstration: it is one\n"
          "      clip, and confirming it needs a clip-level motion measurement the probe\n"
          "      does not currently store."
          if standout else
          f"NOT the standout: {SUBJECT} sits at {excess[SUBJECT]:+.1f}, below "
          f"{max(others):+.1f} elsewhere.\n"
          "      Refuted as stated. Recorded so it is not rediscovered: on the RIGHT\n"
          "      edge alone the subject's excess is far larger than anyone else's, which\n"
          "      is what suggested the hypothesis in the first place. That one-sided cut\n"
          "      was chosen AFTER seeing the numbers, and drift-turn -- a camera-motion\n"
          "      clip with rho +0.68 -- ties the subject on the symmetric definition, so\n"
          "      the one-sided version is not evidence. Reviving it needs a motion\n"
          "      measurement decided on in advance, not a better choice of columns.")
    return standout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    runs = load(Path(args.results_dir))
    print(f"Open question 0c -- {SUBJECT} vs the other {len(RUNS) - 1} F1 probe videos")
    print("No new runs: every number below comes from the eight existing probe results.")

    real_skill = test_rho_vs_zero(runs)
    degenerate = test_predictor_degeneracy(runs)
    edges = test_edge_columns(runs)

    print("\n--- where 0c stands ---")
    if not real_skill:
        print("* EVCA SC on drift-straight is statistically indistinguishable from chance.")
    if not degenerate:
        print("* Predictor degeneracy is REFUTED, as flat-marginals already was.")
    if edges:
        print("* Frame-edge content is the one measured asymmetry, and it points at")
        print("  camera motion. UNSETTLED until a motion measurement confirms it.")
    print("* The cost model may NOT be described as uniformly adequate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
