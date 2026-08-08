#!/usr/bin/env python3
"""Fit the transmit-time damage model and ship it as config/damage_predictor.json.

One coefficient set per held-out video, so a run of clip X is always scored by
a model that never saw clip X. `presley.damagemodel.load` refuses any other
pairing -- see its docstring for why a silent fallback would be undetectable.

The target is the WITHIN-RUN damage rank, not raw delta-PSNR: damage scale
varies by run (Real-ESRGAN disperses 4.9 dB between its 10th and 90th
percentile, ProPainter 8.4 dB), and the model is only ever used to order
blocks. Fitting the raw magnitude would let the runs with the widest spread
dominate the fit without making the ordering any better.

Reports held-out skill per video so the model's quality is visible at the point
it is built, rather than being taken on trust by whatever consumes it.

Usage:
    python tools/fit_damage_predictor.py --data-root . -o config/damage_predictor.json
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from presley.damagemodel import FEATURES, DamagePredictor, save  # noqa: E402

# Same collection path as the screen, so the model is fit on exactly the data
# the go/no-go decision was made on.
import analyze_corrected_objective as screen  # noqa: E402

RIDGE = 1.0


def fit_ridge_beta(X: np.ndarray, y: np.ndarray, penalty: float = RIDGE) -> np.ndarray:
    """Ridge on standardized features. Returns beta with the intercept last."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    reg = penalty * np.eye(Xs.shape[1])
    reg[-1, -1] = 0.0            # never penalize the intercept
    return np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ y)


def rank_target(damage: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(damage)) / max(1, len(damage) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT),
                    help="tree holding results/ and cache/ (the main checkout)")
    ap.add_argument("--npz")
    ap.add_argument("-o", "--out", default=str(REPO_ROOT / "config" / "damage_predictor.json"))
    args = ap.parse_args()

    screen.DATA_ROOT = pathlib.Path(args.data_root).resolve()
    npz = args.npz or str(screen.DATA_ROOT / "results" / "block_damage_s1b.npz")

    runs, skipped = screen.collect(npz)
    if skipped:
        print("runs skipped (reported, not silently dropped):")
        for k, c in skipped.most_common():
            print(f"   {c:4d}  {k}")
    videos = sorted({r["video"] for r in runs})
    print(f"\nruns: {len(runs)}   videos: {len(videos)}")
    if len(videos) < 3:
        print("too few videos to hold one out.")
        return 1

    models, skills = [], []
    print(f"\n{'held-out video':18}{'train runs':>12}{'held-out skill':>16}")
    for held_out in videos:
        train = [r for r in runs if r["video"] != held_out]
        test = [r for r in runs if r["video"] == held_out]
        X = np.vstack([r["X"] for r in train])
        y = np.concatenate([rank_target(r["damage"]) for r in train])
        beta = fit_ridge_beta(X, y)
        model = DamagePredictor(beta=beta, held_out_video=held_out, n_train_runs=len(train))

        per_run = [screen.spearman(model.predict(r["X"]), r["damage"]) for r in test]
        med = float(np.median(per_run))
        skills.append(med)
        models.append(model)
        print(f"{held_out:18}{len(train):>12}{med:>+16.3f}")

    print(f"\n{'MEDIAN':18}{'':>12}{float(np.median(skills)):>+16.3f}")

    # Direction of the fit, averaged over folds -- the sign is the article's
    # point (complexity predicts damage POSITIVELY, which is why the current
    # objective is pointed the wrong way), so it is reported, not buried.
    mean_beta = np.mean([m.beta for m in models], axis=0)
    print("\nmean standardized coefficient by feature:")
    for name, b in zip(FEATURES, mean_beta[:-1]):
        print(f"   {name:12}{b:>+9.4f}")

    save(models, args.out)
    print(f"\nwrote {args.out} ({len(models)} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
