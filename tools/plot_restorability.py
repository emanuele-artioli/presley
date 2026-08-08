#!/usr/bin/env python3
"""Post-restoration damage is predictable, and predictable the wrong way round.

`fig:restorability`. This is the article's central diagnosis and it has never
had a picture. The claim: a block's spatial complexity predicts how badly it
comes back after restoration, at within-run Spearman +0.506, same sign on
120 of 120 runs -- and the selection score *already* ranks by complexity,
because complex blocks cost the most bits. So the objective preferentially
degrades the blocks that survive restoration worst.

Drawing that correctly is not a scatter of everything. The claim is a
WITHIN-RUN rank correlation; a pooled scatter of 120 runs would show the
between-run relationship, which is a different and much weaker statement, and
would invite exactly the reading the M1 analysis had to control for. So:

  left   within-run standardized complexity against within-run standardized
         damage, which is the quantity the correlation is computed on;
  right  the distribution of the 120 per-run correlations, which is what makes
         "same sign on 120/120" visible rather than asserted.

Usage:
    python tools/plot_restorability.py --data-root .
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import figkit  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import analyze_corrected_objective as screen  # noqa: E402

SC = 0          # column of sc_mean in the feature matrix
MAX_POINTS = 6000   # scatter density cap; sampled with a fixed seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--figures-dir", default=None)
    args = ap.parse_args()

    screen.DATA_ROOT = pathlib.Path(args.data_root).resolve()
    runs, _ = screen.collect(str(screen.DATA_ROOT / "results" / "block_damage_s1b.npz"))
    if not runs:
        print("no usable runs")
        return 1

    zx, zy, rhos = [], [], []
    for r in runs:
        sc, dmg = r["X"][:, SC], r["damage"]
        if sc.std() == 0 or dmg.std() == 0:
            continue
        # Standardize WITHIN the run: that is the unit the claim is made on,
        # and it removes the per-run offsets a pooled scatter would show
        # instead of the effect.
        zx.append((sc - sc.mean()) / sc.std())
        zy.append((dmg - dmg.mean()) / dmg.std())
        rhos.append(screen.spearman(sc, dmg))

    zx = np.concatenate(zx)
    zy = np.concatenate(zy)
    rhos = np.array(rhos)

    rng = np.random.default_rng(20260808)
    if len(zx) > MAX_POINTS:
        pick = rng.choice(len(zx), MAX_POINTS, replace=False)
        sx, sy = zx[pick], zy[pick]
    else:
        sx, sy = zx, zy

    figkit.style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figkit.FULL_WIDTH, 2.4),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    ax.scatter(sx, sy, s=2, alpha=0.10, color=figkit.COLORS["accent"],
               edgecolors="none", rasterized=True)
    slope, intercept = np.polyfit(zx, zy, 1)
    xs = np.linspace(zx.min(), zx.max(), 50)
    ax.plot(xs, slope * xs + intercept, color=figkit.COLORS["loss"], linewidth=1.4)
    ax.set_xlabel("spatial complexity ($z$)")
    ax.set_ylabel("damage after restoration ($z$)")
    ax.set_title("More complex $\\rightarrow$ more damage")
    ax.set_xlim(-3, 4)
    ax.set_ylim(-3, 4)

    bx.hist(rhos, bins=24, color=figkit.COLORS["accent"], edgecolor="white",
            linewidth=0.4)
    bx.axvline(0, color="black", linewidth=0.8)
    bx.axvline(float(np.median(rhos)), color=figkit.COLORS["loss"],
               linewidth=1.2, linestyle="--")
    bx.set_xlabel("within-run Spearman $\\rho$ (one per run)")
    bx.set_ylabel("runs")
    bx.set_title(f"{int((rhos > 0).sum())}/{len(rhos)} runs positive, "
                 f"median {np.median(rhos):+.3f}")

    fig.tight_layout()

    sentence = (
        "Two panels. Left: a scatter of per-superblock spatial complexity against "
        "post-restoration damage, both standardized within each run, with a fitted "
        "line sloping upward -- more complex blocks come back worse. Right: a "
        f"histogram of the per-run rank correlations, all {int((rhos > 0).sum())} of "
        f"{len(rhos)} of them positive, with a median of {np.median(rhos):+.3f}. "
        "The selection score already ranks blocks by complexity because complex "
        "blocks cost the most bits, so the objective preferentially degrades the "
        "blocks that survive restoration worst.")

    data = {
        "claim": "post-restoration damage is predictable at transmit time, "
                 "and correlates POSITIVELY with the complexity the selection "
                 "score already ranks by",
        "unit": "the run; statistic is a within-run Spearman rho",
        "n_runs": int(len(rhos)),
        "runs_positive": int((rhos > 0).sum()),
        "rho_median": round(float(np.median(rhos)), 4),
        "rho_min": round(float(rhos.min()), 4),
        "rho_max": round(float(rhos.max()), 4),
        "restorers": list(screen.RESTORERS),
        "feature": "sc_mean (EVCA spatial complexity, 64x64 superblocks)",
        "control": "restricted to superblocks that were actually degraded; over "
                   "all superblocks the correlation is inflated to +0.684 because "
                   "complexity predicts whether a block was selected at all",
    }

    paths = figkit.emit("restorability", fig, sentence, data,
                        figures_dir=args.figures_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
