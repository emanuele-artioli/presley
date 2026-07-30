"""S1: filter_frame_downsample gains a graded multi-level map.

Every strength map in project history has been binary 0/1 because this
function only ever emitted round(score) at a single fixed scale, even though
`_adaptive_block_pyramid_upscale` (restoration.py) has always been able to
read an arbitrary per-block exponent -- the 2^k pyramid it implements has
never once run. The tests here pin: levels=1 (default) stays byte-identical
to the pre-S1 behavior, so no existing experiment hash or transmitted_size
moves; levels>1 quantizes the continuous score into a real ladder; and the
per-block downscale factor actually used is 2**level, matching the exponent
convention the pyramid restorer expects.
"""
import cv2
import numpy as np

from presley.degradation import filter_frame_downsample


def test_levels_1_is_byte_identical_to_pre_s1_behavior():
    """The default must not move a single existing pixel or map value."""
    rng = np.random.default_rng(0)
    block_size = 8
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    scores = np.array([[0.8, 0.2], [0.6, 0.0]], dtype=np.float32)

    new_image, new_map = filter_frame_downsample(image, scores, block_size)

    # Hand-computed pre-S1 reference: round(scores) thresholded, scale=0.5.
    expected_map = np.array([[1, 0], [1, 0]], dtype=np.int32)
    np.testing.assert_array_equal(new_map, expected_map)

    blocks = image.reshape(2, block_size, 2, block_size, 3).transpose(0, 2, 1, 3, 4)
    ref = image.copy().reshape(2, block_size, 2, block_size, 3).transpose(0, 2, 1, 3, 4)
    for by, bx in [(0, 0), (1, 0)]:
        small = cv2.resize(blocks[by, bx], (4, 4), interpolation=cv2.INTER_AREA)
        ref[by, bx] = cv2.resize(small, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
    expected_image = ref.transpose(0, 2, 1, 3, 4).reshape(16, 16, 3)

    np.testing.assert_array_equal(new_image, expected_image)


def test_levels_greater_than_1_quantizes_score_into_a_real_ladder():
    """Four blocks spanning the score range must land on four distinct levels
    at levels=3, not collapse back to a threshold."""
    block_size = 8
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    # round(score * 3): 0.05->0, 0.4->1, 0.7->2, 1.0->3
    scores = np.array([[0.05, 0.4], [0.7, 1.0]], dtype=np.float32)

    _, level_map = filter_frame_downsample(image, scores, block_size, levels=3)

    assert level_map.tolist() == [[0, 1], [2, 3]]


def test_level_sets_the_downscale_factor_as_2_to_the_level_not_scale():
    """`scale` is ignored once levels>1 -- the factor is 2**level, the exact
    exponent convention _adaptive_block_pyramid_upscale expects."""
    block_size = 8
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    scores = np.array([[1.0]], dtype=np.float32)  # -> level 2 at levels=2

    new_image, level_map = filter_frame_downsample(
        image, scores, block_size, levels=2, scale=0.9  # deliberately wrong scale
    )
    assert level_map[0, 0] == 2

    small = cv2.resize(image, (2, 2), interpolation=cv2.INTER_AREA)  # 8 // 2**2 == 2
    expected = cv2.resize(small, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(new_image, expected)


def test_sel_floor_still_guarantees_at_least_level_1_when_graded():
    """A selected block must never come back at level 0 (selected but not
    actually degraded), same contract as the binary case, generalized."""
    block_size = 8
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    scores = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)  # would floor to level 0
    sel = np.array([[True, False], [False, False]])

    _, level_map = filter_frame_downsample(image, scores, block_size, levels=3, sel=sel)

    assert level_map[0, 0] >= 1
    assert level_map[0, 1] == 0


def test_levels_clip_to_the_requested_ceiling():
    """A score of 1.0 at levels=4 must land at exactly 4, not overshoot."""
    block_size = 8
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    scores = np.array([[1.0]], dtype=np.float32)

    _, level_map = filter_frame_downsample(image, scores, block_size, levels=4)

    assert level_map[0, 0] == 4


# --- S1b: uniform_level probe mode -----------------------------------------
# The damage-aware ceiling test needs damage_b(k) -- per-block damage at a
# FIXED level. A graded run cannot supply it (each block appears at exactly one
# score-determined level), so probe runs force one level across the whole
# footprint. The footprint must stay the levels>1 footprint, because the
# levels=1 footprint is a strict subset and mixing them would confound
# "which blocks got which level" with "how much area was degraded at all".


def test_uniform_level_flattens_the_ladder_but_keeps_the_footprint():
    block_size = 8
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    scores = np.array([[0.05, 0.4], [0.7, 1.0]], dtype=np.float32)

    _, graded = filter_frame_downsample(image, scores, block_size, levels=3)
    _, uniform = filter_frame_downsample(image, scores, block_size, levels=3, uniform_level=2)

    assert graded.tolist() == [[0, 1], [2, 3]]
    assert uniform.tolist() == [[0, 2], [2, 2]]
    # identical footprint, only the assignment differs -- the whole point
    np.testing.assert_array_equal(graded > 0, uniform > 0)


def test_uniform_level_beats_the_sel_floor_to_stay_actually_uniform():
    """With a budgeted `sel`, the floor promotes selected-but-level-0 blocks to
    level 1. Flattening must happen after that floor, or the probe silently
    carries a minority of level-1 blocks -- less damaged by construction, so
    they inflate the within-level spread the S1b gate turns on. Measured at
    1.4% of all blocks on motorbike before this was fixed."""
    block_size = 8
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    # 0.0 would floor to level 1 under sel; 0.7 grades to level 2 at levels=3
    scores = np.array([[0.0, 0.7], [0.0, 0.0]], dtype=np.float32)
    sel = np.array([[True, True], [False, False]])

    _, level_map = filter_frame_downsample(
        image, scores, block_size, levels=3, sel=sel, uniform_level=3
    )

    assert set(np.unique(level_map).tolist()) == {0, 3}, level_map.tolist()
    assert level_map[0, 0] == 3  # the floored block, not left behind at 1
    assert level_map[1, 0] == 0  # unselected stays out of the footprint


def test_uniform_level_0_is_the_graded_path_unchanged():
    """Default-off must be byte-identical, so no existing hash moves."""
    block_size = 8
    rng = np.random.default_rng(6)
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    scores = np.array([[0.1, 0.5], [0.8, 1.0]], dtype=np.float32)

    a_img, a_map = filter_frame_downsample(image, scores, block_size, levels=3)
    b_img, b_map = filter_frame_downsample(image, scores, block_size, levels=3, uniform_level=0)

    np.testing.assert_array_equal(a_img, b_img)
    np.testing.assert_array_equal(a_map, b_map)


def test_uniform_level_rejects_the_binary_branch_rather_than_being_ignored():
    """levels<=1 scales by `scale`, not 2**level, so a uniform level there
    would be silently dropped -- the exact silent-no-op class of bug that has
    already cost this project a wasted probe batch."""
    block_size = 8
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    scores = np.array([[1.0]], dtype=np.float32)

    try:
        filter_frame_downsample(image, scores, block_size, levels=1, uniform_level=2)
    except ValueError as exc:
        assert "levels > 1" in str(exc)
    else:
        raise AssertionError("uniform_level with levels=1 must raise, not no-op")


def test_uniform_level_above_the_ceiling_is_rejected():
    block_size = 8
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    scores = np.array([[1.0]], dtype=np.float32)

    try:
        filter_frame_downsample(image, scores, block_size, levels=3, uniform_level=4)
    except ValueError as exc:
        assert "exceeds levels" in str(exc)
    else:
        raise AssertionError("uniform_level > levels must raise")


# --- S1b Arm B: caller-supplied level map -------------------------------------
#
# Arm B's entire validity rests on one property: it applies Arm A's level
# HISTOGRAM to a different set of blocks. If `sel` or the score quantizer were
# still allowed to touch the map, the two arms would differ in how much area
# they degrade as well as in which blocks get which level, and the comparison
# would confound assignment with area -- the exact confound S1b was designed to
# remove. These tests pin that the supplied map wins outright, and that every
# way of accidentally not using it raises instead of silently degrading to Arm A.


def test_level_map_is_used_verbatim_and_bypasses_sel():
    """The supplied assignment wins even when `sel` says degrade nothing.

    A run whose oracle map was silently intersected with `sel` would produce an
    Arm-B hash with an Arm-A-or-worse footprint and look like a valid result.
    """
    block_size = 8
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    scores = rng.random((2, 2)).astype(np.float32)
    level_map = np.array([[0, 3], [1, 2]], dtype=np.int32)

    _, out = filter_frame_downsample(
        image, scores, block_size, levels=3,
        sel=np.zeros((2, 2), dtype=bool), level_map=level_map)

    np.testing.assert_array_equal(out, level_map)


def test_level_map_preserves_the_histogram_it_was_given():
    """Arm B holds the per-level block count fixed; that is the whole design."""
    block_size = 8
    rng = np.random.default_rng(12)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    scores = rng.random((4, 4)).astype(np.float32)
    level_map = np.array([[0, 0, 1, 1], [1, 2, 2, 2],
                          [3, 0, 0, 1], [2, 3, 1, 0]], dtype=np.int32)

    _, out = filter_frame_downsample(
        image, scores, block_size, levels=3, level_map=level_map)

    np.testing.assert_array_equal(np.bincount(out.ravel(), minlength=4),
                                  np.bincount(level_map.ravel(), minlength=4))


def test_level_map_default_none_leaves_the_score_path_byte_identical():
    """Adding the parameter must not move any existing experiment hash."""
    block_size = 8
    rng = np.random.default_rng(13)
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    scores = rng.random((2, 2)).astype(np.float32)

    img_a, map_a = filter_frame_downsample(image, scores, block_size, levels=3)
    img_b, map_b = filter_frame_downsample(image, scores, block_size, levels=3,
                                           level_map=None)

    np.testing.assert_array_equal(img_a, img_b)
    np.testing.assert_array_equal(map_a, map_b)


def test_level_map_with_uniform_level_is_rejected():
    """Both set the whole assignment; letting one win silently mislabels the arm."""
    block_size = 8
    rng = np.random.default_rng(14)
    image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    scores = np.array([[1.0]], dtype=np.float32)

    try:
        filter_frame_downsample(image, scores, block_size, levels=3,
                                uniform_level=2,
                                level_map=np.array([[3]], dtype=np.int32))
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("level_map with uniform_level must raise")


def test_level_map_with_levels_1_is_rejected():
    """levels<=1 scales by `scale`, not 2**level -- a graded map would flatten."""
    block_size = 8
    rng = np.random.default_rng(15)
    image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    scores = np.array([[1.0]], dtype=np.float32)

    try:
        filter_frame_downsample(image, scores, block_size, levels=1,
                                level_map=np.array([[1]], dtype=np.int32))
    except ValueError as exc:
        assert "levels > 1" in str(exc)
    else:
        raise AssertionError("level_map with levels=1 must raise")


def test_level_map_above_the_ceiling_is_rejected():
    block_size = 8
    rng = np.random.default_rng(16)
    image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    scores = np.array([[1.0]], dtype=np.float32)

    try:
        filter_frame_downsample(image, scores, block_size, levels=3,
                                level_map=np.array([[9]], dtype=np.int32))
    except ValueError as exc:
        assert "above levels" in str(exc)
    else:
        raise AssertionError("a level above the ceiling must raise")


def test_level_map_shape_mismatch_is_rejected():
    """A map built against a different resolution must not broadcast silently."""
    block_size = 8
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    scores = rng.random((2, 2)).astype(np.float32)

    try:
        filter_frame_downsample(image, scores, block_size, levels=3,
                                level_map=np.zeros((4, 4), dtype=np.int32))
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("a mis-shaped level_map must raise")
