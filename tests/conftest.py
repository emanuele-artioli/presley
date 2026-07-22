"""Shared fixtures.

Everything here is synthetic and CPU-only: the fast tier must run without the
dataset, without `results/`, and without a GPU, so that it is usable on CI and
cheap enough to run on every change.
"""

import numpy as np
import pytest

REPO_ROOT_MARKER = "experiments.yaml"


@pytest.fixture
def rng():
    """Seeded generator — a metric test that fails only sometimes is useless."""
    return np.random.default_rng(20260722)


@pytest.fixture
def frame(rng):
    """A small BGR frame, the shape every metric helper expects."""
    return rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)


@pytest.fixture
def half_mask():
    """Boolean mask covering the left half of a 32x48 frame."""
    mask = np.zeros((32, 48), dtype=bool)
    mask[:, :24] = True
    return mask
