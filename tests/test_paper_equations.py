"""The manuscript's method section, re-implemented from the printed equations.

These tests exist because the paper and the code had silently diverged. As of
2026-08-06 `sections/presley.tex` said the importance score *inverts the sign*
of background blocks (the code multiplies them by 10, and the paragraph under
the equation said so, contradicting the equation above it); paired spatial and
temporal complexity at the same frame index (the code reads temporal from the
next frame); smoothed complexity before the region step (the code smooths
after it); and gave a graded per-block downsampling factor as the degradation
(every reported experiment runs the binary case at a fixed 2x). The restoration
algorithm's loop, as printed, did not execute at all for a binary strength map.

None of that is catchable by a compiler or a reviewer without the source, so it
is caught here instead: each test writes the paper's equation out independently
and asserts the code agrees. A future change to either side fails the test, and
the failure message names the equation to go and fix.

Deliberately NOT tested: anything requiring EVCA, UFO, a dataset or a GPU. The
arithmetic is pure and the fast tier must stay pure with it.
"""

import numpy as np
import pytest

from presley.degradation import select_removal_mask_global
from presley.preprocessing import BACKGROUND_PRIORITY, combine_removability


@pytest.fixture
def tensors(rng):
    """(spatial, temporal, background) on a small block grid over 5 frames."""
    spatial = rng.random((5, 4, 6)).astype(np.float64)
    temporal = rng.random((5, 4, 6)).astype(np.float64)
    background = rng.random((5, 4, 6)) > 0.5
    return spatial, temporal, background


def paper_score(spatial, temporal, background, alpha, beta, gamma):
    """Equations 1-4 of the method section, written out literally.

    Deliberately a plain loop rather than the vectorized form the code uses:
    if both sides shared an implementation the test would only prove the code
    equals itself.
    """
    n_frames, n_rows, n_cols = spatial.shape
    c = np.zeros_like(spatial)
    for n in range(n_frames):
        for i in range(n_rows):
            for j in range(n_cols):
                if n < n_frames - 1:  # Eq. 1
                    c[n, i, j] = alpha * spatial[n, i, j] + (1 - alpha) * temporal[n + 1, i, j]
                else:                 # last frame: no successor, no temporal term
                    c[n, i, j] = spatial[n, i, j]
                if background[n, i, j]:  # Eq. 2, background priority
                    c[n, i, j] *= gamma

    smoothed = np.zeros_like(c)     # Eq. 3, non-recursive, frame 0 unsmoothed
    smoothed[0] = c[0]
    for n in range(1, n_frames):
        smoothed[n] = beta * c[n] + (1 - beta) * c[n - 1]

    lo, hi = smoothed.min(), smoothed.max()   # Eq. 4
    return (smoothed - lo) / (hi - lo)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("beta", [0.25, 0.5, 0.99])
def test_score_matches_the_printed_equations(tensors, alpha, beta):
    spatial, temporal, background = tensors
    expected = paper_score(spatial, temporal, background, alpha, beta, BACKGROUND_PRIORITY)
    actual = combine_removability(spatial, temporal, background, alpha, beta)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_background_priority_is_the_factor_the_paper_prints():
    assert BACKGROUND_PRIORITY == 10.0, "Eq. priority prints gamma = 10"


def test_background_priority_is_a_scaling_not_a_sign_inversion(rng):
    """The specific error the manuscript carried: sign inversion, not scaling.

    A sign inversion puts the two populations on opposite sides of zero and
    makes the ordering total; scaling leaves them interleaved. The paper's own
    selection story depends on which it is, because a hard exclusion is only
    needed under the second.
    """
    spatial = np.ones((2, 3, 3))
    temporal = np.ones((2, 3, 3))
    background = np.zeros((2, 3, 3), dtype=bool)
    background[:, 0, :] = True

    scores = combine_removability(spatial, temporal, background, alpha=1.0, beta=1.0)
    assert (scores >= 0).all(), "scaling keeps every score non-negative"
    assert scores[0, 0, 0] > scores[0, 1, 0], "background must outrank foreground"


def test_a_complex_foreground_block_can_outrank_a_simple_background_one():
    """Why Algorithm `alg:select` needs a hard exclusion at all.

    Reproduces the bmx-trees failure in miniature: with a 10x priority, a
    foreground block more than 10x as complex as a background block still wins,
    so a purely score-driven top-k degrades foreground.
    """
    spatial = np.zeros((2, 1, 2))
    spatial[:, 0, 0] = 20.0   # foreground, very complex
    spatial[:, 0, 1] = 1.0    # background, simple
    temporal = np.zeros_like(spatial)
    background = np.array([[[False, True]], [[False, True]]])

    scores = combine_removability(spatial, temporal, background, alpha=1.0, beta=1.0)
    assert scores[0, 0, 0] > scores[0, 0, 1], (
        "20 > 10x1, so the soft priority is outvoted -- this is the case the "
        "hard foreground exclusion exists for")


def test_temporal_term_is_read_from_the_next_frame():
    """Eq. 1 pairs S[n] with T[n+1]; a same-index pairing is a different score."""
    spatial = np.zeros((3, 1, 1))
    temporal = np.array([[[0.0]], [[1.0]], [[0.0]]])   # a spike at frame 1
    background = np.zeros((3, 1, 1), dtype=bool)

    scores = combine_removability(spatial, temporal, background, alpha=0.0, beta=1.0)
    assert scores[0, 0, 0] == pytest.approx(1.0), "frame 0 must see temporal[1]"
    assert scores[1, 0, 0] == pytest.approx(0.0), "frame 1 must see temporal[2]"


def test_last_frame_falls_back_to_spatial_only():
    spatial = np.array([[[0.0]], [[1.0]]])
    temporal = np.zeros((2, 1, 1))
    background = np.zeros((2, 1, 1), dtype=bool)
    scores = combine_removability(spatial, temporal, background, alpha=0.0, beta=1.0)
    # alpha=0 would zero the last frame if it used the temporal term.
    assert scores[1, 0, 0] > scores[0, 0, 0]


def test_smoothing_is_not_recursive():
    """Eq. 3 mixes frame n with frame n-1's UNSMOOTHED score.

    Under a recursive formulation a spike at frame 0 would still be visible at
    frame 2; under this one it is gone.
    """
    spatial = np.zeros((3, 1, 1))
    spatial[0, 0, 0] = 1.0
    temporal = np.zeros((3, 1, 1))
    background = np.zeros((3, 1, 1), dtype=bool)

    scores = combine_removability(spatial, temporal, background, alpha=1.0, beta=0.5)
    assert scores[2, 0, 0] == pytest.approx(0.0), "frame 0's spike must not reach frame 2"


def test_beta_one_disables_smoothing(tensors):
    spatial, temporal, background = tensors
    unsmoothed = combine_removability(spatial, temporal, background, 0.5, 1.0)
    expected = paper_score(spatial, temporal, background, 0.5, 1.0, BACKGROUND_PRIORITY)
    np.testing.assert_allclose(unsmoothed, expected, rtol=0, atol=1e-12)


def test_scores_are_normalized_to_the_unit_interval(tensors):
    spatial, temporal, background = tensors
    scores = combine_removability(spatial, temporal, background, 0.5, 0.5)
    assert scores.min() == pytest.approx(0.0)
    assert scores.max() == pytest.approx(1.0)


# --- Algorithm `alg:select`: budgeted global top-k with hard exclusion --------

def test_selection_budget_matches_the_printed_formula(rng):
    """k = floor(a*J)*I -- the same total a per-row budget would have spent."""
    scores = rng.random((45, 80)).astype(np.float32)
    for amount in (0.1, 0.25, 0.5):
        mask = select_removal_mask_global(scores, amount, cluster_blocks=False)
        assert mask.sum() == int(amount * 80) * 45


def test_selection_never_touches_an_excluded_block(rng):
    """The hard exclusion is absolute, whatever the score says."""
    scores = rng.random((20, 20)).astype(np.float32)
    exclude = np.zeros((20, 20), dtype=bool)
    exclude[:10] = True          # top half is foreground
    scores[:10] = 1.0            # ...and is also the highest-scoring half

    mask = select_removal_mask_global(scores, 0.5, cluster_blocks=True, exclude=exclude)
    assert mask[:10].sum() == 0, "an excluded block must never be selected"


def test_exclusion_survives_the_clustering_blur():
    """Order matters: the blur runs first, the exclusion second.

    A hot foreground block smears score onto its neighbours through the
    clustering blur. Applying the exclusion afterwards is what stops that block
    itself from re-entering the ranking at the top.
    """
    scores = np.zeros((4, 4), dtype=np.float32)
    scores[0, 0] = 100.0          # a hot foreground block
    exclude = np.zeros((4, 4), dtype=bool)
    exclude[0, 0] = True

    mask = select_removal_mask_global(scores, 0.5, cluster_blocks=True, exclude=exclude)
    assert mask[0, 0] == 0
    assert mask[0, 1] == 1 and mask[1, 0] == 1, (
        "the blur is still applied -- neighbours inherit the hot block's score")


def test_budget_is_capped_by_the_eligible_block_count():
    """It cannot overflow into excluded blocks to make up the shortfall."""
    scores = np.ones((4, 4), dtype=np.float32)
    exclude = np.ones((4, 4), dtype=bool)
    exclude[3, 3] = False         # exactly one eligible block

    mask = select_removal_mask_global(scores, 0.75, cluster_blocks=False, exclude=exclude)
    assert mask.sum() == 1 and mask[3, 3] == 1


def test_amount_below_one_is_a_fraction_and_at_or_above_one_is_a_count():
    """A dual meaning worth pinning, because the two differ by 4x here.

    Every reported experiment passes a fraction (0.25), which is what the
    paper's Algorithm documents. The absolute-count branch exists for callers
    that want an exact per-row budget, and reading `a = 1` as "all blocks"
    would be wrong by the width of the frame.
    """
    scores = np.ones((10, 8), dtype=np.float32)
    assert select_removal_mask_global(scores, 0.5, cluster_blocks=False).sum() == 4 * 10
    assert select_removal_mask_global(scores, 1.0, cluster_blocks=False).sum() == 1 * 10
