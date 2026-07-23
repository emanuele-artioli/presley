"""Region-restricted metrics.

The paper's whole argument is the foreground/background split, so a mask that
is applied wrongly does not crash — it quietly produces a number that reads as
a result. These tests pin the cases where that would happen.
"""

import numpy as np
import pytest

from presley.evaluation.masked import (
    _fg_tight_bbox,
    _fg_union_bbox,
    _masked_mse,
    _masked_psnr,
    _masked_ssim,
)


def test_identical_frames_saturate_psnr(frame):
    assert _masked_psnr(frame, frame) == 100.0


def test_identical_frames_have_zero_mse(frame):
    assert _masked_mse(frame, frame) == 0.0


def test_mse_matches_a_hand_computed_difference():
    ref = np.zeros((4, 4, 3), dtype=np.uint8)
    dec = np.full((4, 4, 3), 10, dtype=np.uint8)
    assert _masked_mse(ref, dec) == pytest.approx(100.0)


def test_psnr_matches_the_definition():
    ref = np.zeros((4, 4, 3), dtype=np.uint8)
    dec = np.full((4, 4, 3), 10, dtype=np.uint8)
    expected = 20 * np.log10(255.0 / 10.0)
    assert _masked_psnr(ref, dec) == pytest.approx(expected, rel=1e-6)


def test_mask_restricts_the_metric_to_its_region(frame, half_mask):
    """Corrupting only the unmasked half must leave the masked score untouched."""
    dec = frame.copy()
    dec[:, 24:] = 0  # damage the right half only; the mask covers the left

    assert _masked_mse(frame, dec, half_mask) == 0.0
    assert _masked_mse(frame, dec) > 0.0


def test_an_all_false_mask_falls_back_to_the_whole_frame(frame):
    """np.any(mask) is false, so the helpers score the full frame.

    Worth pinning because it is the surprising branch: an empty foreground
    mask yields a *whole-frame* number that is easy to mistake for an FG one.
    """
    dec = frame.copy()
    dec[:, :] = 0
    empty = np.zeros(frame.shape[:2], dtype=bool)

    assert _masked_mse(frame, dec, empty) == pytest.approx(_masked_mse(frame, dec))


def test_metrics_return_zero_for_a_missing_frame(frame):
    assert _masked_psnr(None, frame) == 0.0
    assert _masked_mse(frame, None) == 0.0
    assert _masked_ssim(None, None) == 0.0


def test_ssim_of_a_frame_with_itself_is_one(frame):
    assert _masked_ssim(frame, frame) == pytest.approx(1.0, abs=1e-6)


def test_ssim_with_an_empty_mask_is_one(frame):
    """No region to compare means "no difference found", not an exception."""
    empty = np.zeros(frame.shape[:2], dtype=bool)
    assert _masked_ssim(frame, frame, empty) == 1.0


def test_ssim_survives_a_mask_smaller_than_the_window(frame):
    """A 2px mask crop is narrower than SSIM's 7px window."""
    tiny = np.zeros(frame.shape[:2], dtype=bool)
    tiny[0:2, 0:2] = True
    assert _masked_ssim(frame, frame, tiny) == pytest.approx(1.0)


# --- bounding boxes ------------------------------------------------------------
# Both helpers return (y1, y2, x1, x2) — row bounds first. The order is easy to
# transpose at a call site, which on a non-square frame silently crops the wrong
# region instead of raising, so it is pinned here.


def test_tight_bbox_wraps_the_mask_with_padding():
    mask = np.zeros((32, 48), dtype=bool)
    mask[10:20, 12:24] = True

    assert _fg_tight_bbox(mask, w=48, h=32, pad=2) == (8, 22, 10, 26)


def test_tight_bbox_is_clamped_to_the_frame():
    """Padding at the border must not produce negative or out-of-range coords."""
    mask = np.zeros((32, 48), dtype=bool)
    mask[0:3, 0:3] = True

    y1, y2, x1, x2 = _fg_tight_bbox(mask, w=48, h=32, pad=8)

    assert 0 <= y1 < y2 <= 32
    assert 0 <= x1 < x2 <= 48


def test_bboxes_return_none_for_an_empty_mask():
    """Callers branch on None; returning a degenerate box would crop nothing."""
    empty = np.zeros((32, 48), dtype=bool)

    assert _fg_tight_bbox(empty, w=48, h=32) is None
    assert _fg_union_bbox([empty, empty], w=48, h=32) is None


def test_union_bbox_covers_every_frames_mask():
    """The union must contain each per-frame box — that is the point of it."""
    a = np.zeros((32, 48), dtype=bool)
    a[4:8, 4:8] = True
    b = np.zeros((32, 48), dtype=bool)
    b[20:24, 30:36] = True

    y1, y2, x1, x2 = _fg_union_bbox([a, b], w=48, h=32, pad=0)

    assert y1 <= 4 and x1 <= 4
    assert y2 >= 24 and x2 >= 36


def test_union_bbox_dimensions_are_even():
    """The crop is re-encoded as yuv420, which cannot take odd dimensions."""
    mask = np.zeros((32, 48), dtype=bool)
    mask[5:12, 7:18] = True  # deliberately odd offsets and extents

    y1, y2, x1, x2 = _fg_union_bbox([mask], w=48, h=32, pad=1)

    assert (y2 - y1) % 2 == 0
    assert (x2 - x1) % 2 == 0
