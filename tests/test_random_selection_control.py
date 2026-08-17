"""The placement control (plan F1) must isolate the ranking and nothing else.

These check the properties the experiment's validity rests on, not that the code
runs. If any of them fails, the F1 contrast answers a different question than the
one it is pre-registered to answer.
"""
import numpy as np
import pytest

from presley.degradation import random_score_map, select_removal_mask_global


def test_random_map_is_reproducible_for_a_seed_and_frame():
    a = random_score_map((10, 16), seed=7, frame_index=3)
    b = random_score_map((10, 16), seed=7, frame_index=3)
    assert np.array_equal(a, b), "a seeded arm must reproduce exactly or it is not citable"


def test_successive_frames_are_independent():
    a = random_score_map((20, 30), seed=1, frame_index=0)
    b = random_score_map((20, 30), seed=1, frame_index=1)
    # Independent draws must not correlate; a shared stream would make the
    # control accidentally temporally smooth, which the incumbent's beta does.
    r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    assert abs(r) < 0.15, f"consecutive random frames correlate at {r:.3f}"


def test_different_seeds_give_different_placements():
    a = random_score_map((12, 20), seed=1, frame_index=0)
    b = random_score_map((12, 20), seed=2, frame_index=0)
    assert not np.array_equal(a, b)


def test_map_is_in_unit_range_like_the_real_score():
    m = random_score_map((8, 8), seed=0, frame_index=0)
    assert m.min() >= 0.0 and m.max() < 1.0, "downstream level maps assume [0,1]"


def test_random_arm_spends_the_same_budget_as_the_score_arm():
    """The whole point: same count, same strength, different placement."""
    rng = np.random.default_rng(0)
    real = rng.random((9, 16)).astype(np.float32)
    rand = random_score_map((9, 16), seed=5, frame_index=0)
    a = select_removal_mask_global(real, 0.25, cluster_blocks=True)
    b = select_removal_mask_global(rand, 0.25, cluster_blocks=True)
    assert a.sum() == b.sum(), "arms must degrade the same number of blocks"


def test_exclusion_still_binds_on_the_random_arm():
    """Otherwise the random arm would degrade foreground and lose for that reason."""
    rand = random_score_map((9, 16), seed=3, frame_index=0)
    excl = np.zeros((9, 16), dtype=bool)
    excl[:4, :] = True
    sel = select_removal_mask_global(rand, 0.25, cluster_blocks=True, exclude=excl)
    assert sel[excl].sum() == 0, "hard exclusion must apply to the control arm too"


def test_random_selection_is_clustered_not_confetti():
    """The validity condition from the pre-registration.

    The blur is applied by `select_removal_mask_global`, so the random arm gets
    contiguous patches like the incumbent. Without it the arm would lose on
    fragmentation rather than on placement. Compare against an unblurred
    selection of the same budget: blurring must reduce the number of isolated
    singletons.
    """
    rand = random_score_map((18, 32), seed=11, frame_index=0)
    blurred = select_removal_mask_global(rand, 0.25, cluster_blocks=True)
    confetti = select_removal_mask_global(rand, 0.25, cluster_blocks=False)

    def singletons(mask):
        m = mask > 0
        pad = np.pad(m, 1)
        neigh = (pad[:-2, 1:-1].astype(int) + pad[2:, 1:-1] +
                 pad[1:-1, :-2] + pad[1:-1, 2:])
        return int(((m) & (neigh == 0)).sum())

    assert singletons(blurred) < singletons(confetti), (
        "clustering must reduce isolated blocks, or the control confounds "
        "placement with fragmentation"
    )
