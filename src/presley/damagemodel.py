"""Predict post-restoration damage at transmit time, and rank blocks by it.

The article's selection objective ranks blocks by a removability score R that
estimates only how many BITS a block costs. What selection should maximize is a
ratio -- bits freed per unit of damage that survives restoration -- and the
denominator was never modelled. M1 established that the denominator *is*
predictable before transmission (spatial complexity, within-run Spearman
+0.506, same sign on 120/120 runs) and, worse, predictable with the sign that
makes the current objective actively wrong: the blocks the score most wants to
degrade are the ones that come back worst.

This module is the corrected denominator. It is deliberately a linear model on
the five features declared in `docs/PREREG_M1_RESTORABILITY.md` -- adding a
sixth after seeing the results would be candidate shopping, and the whole point
is that the correction follows the pre-registered diagnosis rather than a
search.

Two design decisions that are not obvious and are load-bearing:

**Held out by video, never by run.** Several runs share a clip (bear has 33),
so holding out by run leaks content between train and test and inflates the
skill. The shipped model file carries one coefficient set per held-out video,
and `load` refuses to hand back a model that saw the video being run.

**Standardized per run at prediction time, not with the training statistics.**
The model is fit on 64x64 superblock features (that is the grid the damage was
mined on) but applied on the 8/16 block grid selection actually runs on, where
the same feature has a different scale. Because the output is only ever used as
a *ranking*, z-scoring against the run's own feature statistics is what makes
the two grids comparable. Measured, not assumed: held-out skill is +0.459 with
per-run standardization against +0.400 with the training statistics, over the
same 120 runs.

The residual limitation, which the ranking cannot repair: per-run standardizing
matches the first two moments of the feature distribution across grids, not its
shape. So the model transfers as a monotone ranker, and a claim about predicted
damage *magnitudes* at the block grid is not supported by this fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# The five pre-registered features, in the order the coefficient vector expects.
# Do NOT extend: the Holm family for the M1 analysis is sized for exactly this
# list, losers included, and a sixth feature added now would be chosen with
# knowledge of the answer.
FEATURES = ("sc_mean", "tc_mean", "sc_var", "tc_var", "frame_edge")

# Keeps the ratio away from a division through zero. A ridge prediction of a
# rank target can land slightly negative, and the ratio is only meaningful on a
# strictly positive denominator.
_DAMAGE_FLOOR = 0.05


@dataclass(frozen=True)
class DamagePredictor:
    """Coefficients on standardized features, plus the intercept."""

    beta: np.ndarray          # len(FEATURES) + 1, intercept last
    held_out_video: str       # the clip excluded from this fit
    n_train_runs: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted damage rank for each row of `features`.

        `features` is (n, len(FEATURES)), standardized here against its own
        column statistics -- see the module docstring for why that is the
        prediction-time contract rather than an approximation of one.
        """
        if features.shape[1] != len(FEATURES):
            raise ValueError(
                f"expected {len(FEATURES)} features {FEATURES}, got {features.shape[1]}")
        mu = features.mean(axis=0)
        sd = features.std(axis=0)
        sd[sd == 0] = 1.0
        standardized = np.column_stack([(features - mu) / sd, np.ones(len(features))])
        return standardized @ self.beta

    def corrected_scores(self, removability: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Bits freed per unit of predicted damage: the corrected objective.

        `removability` is the existing score R (the bits proxy, already
        carrying the background priority and the [0,1] normalization).
        Returns a score with the same orientation -- higher means a better
        block to degrade -- so it can be handed to `select_removal_mask_global`
        in place of R, leaving the clustering blur and the hard foreground
        exclusion exactly as they are.
        """
        predicted = self.predict(features)
        positive = predicted - predicted.min() + _DAMAGE_FLOOR
        return removability / positive


def block_features(spatial: np.ndarray, temporal: np.ndarray, frame_index: int) -> np.ndarray:
    """The five features for one frame, on the native block grid.

    `spatial` / `temporal` are the EVCA cubes, (frames, rows, cols). Returned
    as (rows*cols, 5) in FEATURES order, so a caller reshapes the prediction
    back to the block grid itself.

    The variance features are across frames per block position, matching how
    they were computed when the model was fit -- a per-frame variance would be
    a different quantity with the same name.
    """
    rows, cols = spatial.shape[1], spatial.shape[2]
    sc = spatial[frame_index].reshape(-1)
    tc = temporal[frame_index].reshape(-1)
    sc_var = spatial.var(axis=0).reshape(-1)
    tc_var = temporal.var(axis=0).reshape(-1)

    edge = np.zeros((rows, cols), dtype=float)
    edge[0, :] = edge[-1, :] = 1.0
    edge[:, 0] = edge[:, -1] = 1.0

    return np.column_stack([sc, tc, sc_var, tc_var, edge.reshape(-1)])


def save(models: List[DamagePredictor], path: str) -> None:
    payload = {
        "features": list(FEATURES),
        "models": [
            {"held_out_video": m.held_out_video,
             "n_train_runs": m.n_train_runs,
             "beta": [float(x) for x in m.beta]}
            for m in models
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def load(path: str, exclude_video: str) -> DamagePredictor:
    """The model fitted WITHOUT `exclude_video`.

    Raises rather than falling back when no such model exists. A silent
    fallback to a model that saw this clip would leak content into the arm
    under test and produce a flattering result that nothing downstream could
    detect.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    stored = tuple(payload.get("features", ()))
    if stored != FEATURES:
        raise ValueError(
            f"model file was fit on features {stored}, but this code expects {FEATURES}")

    by_video: Dict[str, dict] = {m["held_out_video"]: m for m in payload["models"]}
    entry: Optional[dict] = by_video.get(exclude_video)
    if entry is None:
        raise KeyError(
            f"no damage model that excludes {exclude_video!r}. Refusing to use one "
            f"that trained on it: that leaks the clip under test into its own "
            f"prediction. Available: {sorted(by_video)}")
    return DamagePredictor(beta=np.asarray(entry["beta"], dtype=float),
                           held_out_video=entry["held_out_video"],
                           n_train_runs=int(entry["n_train_runs"]))
