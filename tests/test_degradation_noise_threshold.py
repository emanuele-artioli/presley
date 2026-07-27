"""Noise selection must match blur/downsample's round(score)>0 threshold."""
import numpy as np

from presley.degradation import (
    filter_frame_gaussian,
    filter_frame_noise,
)


def test_noise_default_selection_matches_blur_not_score_times_variance():
    """Regression for the coverage bug in NOISE_MODE_DECISION_REPORT.md.

    With noise_variance=50, the old code selected when round(score*50)>0
    (score >= 0.01, ~95% of blocks). Blur selects round(score)>0 (score >= 0.5).
    Mid-range scores must not be noised under the default sel=None path.
    """
    rng = np.random.default_rng(0)
    block_size = 8
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    # One block at 0.2 (old noise path would select; blur would not),
    # one at 0.8 (both should select).
    scores = np.array([[0.2, 0.8], [0.0, 0.0]], dtype=np.float32)

    _, blur_map = filter_frame_gaussian(image, scores, block_size, kernel_size=3)
    _, noise_map = filter_frame_noise(image, scores, block_size, noise_variance=50.0)

    assert blur_map[0, 0] == 0
    assert blur_map[0, 1] > 0
    assert noise_map[0, 0] == 0, "score 0.2 must not be selected under round(score)>0"
    assert noise_map[0, 1] > 0, "score 0.8 must still be noised"
    # Adaptive strength: selected block carries score * variance, not a binary 1.
    assert noise_map[0, 1] == np.float32(0.8 * 50.0)


def test_noise_explicit_sel_still_overrides_threshold():
    """An explicit budget mask must select a low-score block that the default would skip."""
    rng = np.random.default_rng(1)
    block_size = 8
    image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    scores = np.array([[0.2, 0.0], [0.0, 0.0]], dtype=np.float32)
    sel = np.array([[True, False], [False, False]])

    _, noise_map = filter_frame_noise(
        image, scores, block_size, noise_variance=50.0, sel=sel
    )
    assert noise_map[0, 0] > 0
    assert noise_map[0, 1] == 0
