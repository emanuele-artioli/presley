"""O2 re-test: transform-domain AC truncation as a first-class operator.

F5 ran AC truncation as a throwaway script at one fixed strength (keep=2),
pre-restoration, on one video. These tests cover the pipeline operator that
replaces it: that `keep` is a real strength knob, that the codec-aligned grid is
enforced rather than silently mis-applied, that the selection/budget contract
matches the other operators, and that the emitted map stays binary (which is
what lets NAFNet consume it).
"""
import numpy as np
import pytest

from presley.degradation import _ac_truncate_patch, filter_frame_ac_truncate


def _image(h=32, w=48, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_keep_equal_to_dct_size_is_a_no_op():
    """Keeping every coefficient must not touch a pixel -- the identity end of
    the strength ladder, and the guard that the DCT round-trip itself is sane."""
    patch = _image(8, 8)
    np.testing.assert_array_equal(_ac_truncate_patch(patch, keep=8), patch)


def test_keep_one_flattens_each_dct_subblock():
    """keep=1 retains DC only, so every 8x8 sub-block becomes flat."""
    patch = _image(16, 16, seed=3)
    out = _ac_truncate_patch(patch, keep=1)
    for y in (0, 8):
        for x in (0, 8):
            sub = out[y:y + 8, x:x + 8].astype(np.float32)
            # A DC-only sub-block is constant per channel; allow 1 LSB for the
            # YCrCb<->BGR round-trip's rounding.
            assert sub.reshape(-1, 3).ptp(axis=0).max() <= 1


def test_lower_keep_is_monotonically_stronger():
    """`keep` is the strength knob: smaller keep must degrade more, otherwise a
    strength sweep over it measures nothing."""
    patch = _image(16, 16, seed=7)
    errs = []
    for keep in (6, 4, 2, 1):
        out = _ac_truncate_patch(patch, keep)
        errs.append(float(np.mean((out.astype(np.float32) - patch.astype(np.float32)) ** 2)))
    assert errs == sorted(errs), f"error must grow as keep shrinks, got {errs}"


def test_only_selected_blocks_are_touched():
    image = _image(32, 32, seed=11)
    scores = np.ones((2, 2), dtype=np.float32)
    sel = np.array([[True, False], [False, False]])
    out, smap = filter_frame_ac_truncate(image, scores, 16, keep=1, sel=sel)
    np.testing.assert_array_equal(smap, np.array([[1, 0], [0, 0]]))
    # Untouched blocks are bit-exact.
    np.testing.assert_array_equal(out[:16, 16:], image[:16, 16:])
    np.testing.assert_array_equal(out[16:, :], image[16:, :])
    assert not np.array_equal(out[:16, :16], image[:16, :16])


def test_selection_floor_guarantees_a_selected_block_is_degraded():
    """Same contract as blur/downsample: a block inside the budget must never
    come back at strength 0, which would silently shrink the removal rate."""
    image = _image(32, 32, seed=13)
    scores = np.zeros((2, 2), dtype=np.float32)  # round(score) == 0 everywhere
    sel = np.array([[True, True], [False, False]])
    _, smap = filter_frame_ac_truncate(image, scores, 16, keep=2, sel=sel)
    np.testing.assert_array_equal(smap, np.array([[1, 1], [0, 0]]))


def test_map_is_binary_even_for_high_scores():
    """NAFNet reads the map as 'was this block degraded'; a non-binary map would
    be silently misinterpreted as extra restoration rounds."""
    image = _image(32, 32, seed=17)
    scores = np.full((2, 2), 3.4, dtype=np.float32)
    _, smap = filter_frame_ac_truncate(image, scores, 16, keep=2)
    assert set(np.unique(smap)).issubset({0, 1})


def test_non_multiple_block_size_is_rejected_loudly():
    """block_size 12 cannot be tiled by 8x8 transforms; a silent partial
    application would make 'transform-aligned' a false claim."""
    image = _image(24, 24)
    scores = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="multiple of dct_size"):
        filter_frame_ac_truncate(image, scores, 12)


def test_keep_below_one_is_rejected():
    image = _image(32, 32)
    scores = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="keep must be"):
        filter_frame_ac_truncate(image, scores, 16, keep=0)


def test_output_shape_survives_a_non_divisible_frame():
    """640x360 with block_size 16 divides, but 360 does not divide by 32 -- the
    padding path has to give the original geometry back."""
    image = _image(40, 48, seed=19)
    scores = np.ones((3, 3), dtype=np.float32)
    out, smap = filter_frame_ac_truncate(image, scores, 16, keep=2)
    assert out.shape == image.shape
    assert smap.shape == (40 // 16 + 1, 48 // 16)


# Deliberately NOT tested: cv2.dct/idct numerical accuracy (third-party), the
# choice of dct_size=8 for AV1 (a modelling decision, not a code contract), and
# NAFNet's own restoration quality (that is what the experiment measures).
