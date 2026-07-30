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
