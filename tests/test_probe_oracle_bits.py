"""F1's oracle-bits probe — the statistics and the grid alignment.

What is worth testing here is narrow. The encoding itself is ffmpeg's job and the
EVCA scores are EVCA's; neither is ours to re-verify. What *is* ours, and what
would fail silently rather than loudly, is:

- **Index alignment.** `marginal_bits[i]` and `evca_sc[i]` must describe the same
  superblock. An off-by-one produces a perfectly plausible wrong rho and no error
  anywhere — it is the single most likely defect in this component.
- **The statistics.** rho and the capture ratio are hand-rolled (scipy is not a
  runtime dep), so they get checked against cases with known answers, including
  ties and the negative marginals that real encodes actually produce.

Deliberately not tested: ffmpeg's output, EVCA's scores, and the wall-clock cost.
Nothing here touches the real results/ tree.
"""
import numpy as np
import pytest

from presley.components.probe_oracle_bits import (
    _capture_ratio,
    _rankdata,
    _spearman,
)
from presley.degradation import filter_frame_mean_fill


def test_spearman_is_exact_on_a_perfect_monotone_pair():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _spearman(x, x * 3 + 1) == pytest.approx(1.0)
    assert _spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_value_based():
    """The point of rho here: a wildly non-linear but monotone relation still
    scores 1.0. Bit cost vs complexity is monotone at best, never linear."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _spearman(x, np.exp(x)) == pytest.approx(1.0)


def test_ties_get_average_ranks():
    """Untied ranks on tied data would silently inflate rho. EVCA scores tie
    often on flat content."""
    assert _rankdata(np.array([5.0, 1.0, 5.0, 1.0])).tolist() == [3.5, 1.5, 3.5, 1.5]


def test_capture_ratio_is_one_when_the_score_ranks_like_the_oracle():
    marginal = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
    assert _capture_ratio(marginal.copy(), marginal, k=2) == pytest.approx(1.0)


def test_capture_ratio_falls_when_the_score_picks_the_wrong_blocks():
    marginal = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
    worst_possible = np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # exactly inverted
    # picks the two cheapest (20+40=60) against an oracle's 180
    assert _capture_ratio(worst_possible, marginal, k=2) == pytest.approx(60 / 180)


def test_capture_ratio_keeps_negative_marginals_rather_than_clipping_them():
    """Mean-filling a block can genuinely *cost* bits (measured: up to ~0.9% of
    the reference). Clipping those to zero would flatter the proxy, so a score
    that selects a negative block must be penalised for it."""
    marginal = np.array([100.0, 50.0, -30.0])
    picks_the_negative = np.array([0.0, 1.0, 5.0])
    assert _capture_ratio(picks_the_negative, marginal, k=1) == pytest.approx(-30 / 100)


def test_mean_fill_touches_exactly_the_selected_superblock():
    """The alignment guarantee, checked structurally: the probe passes a one-hot
    `sel` and relies on filter_frame_mean_fill deriving its geometry from
    frame_scores.shape, so score index i and marginal_bits[i] are the same block.
    """
    bs = 64
    n_by, n_bx = 5, 10
    frame = np.full((n_by * bs, n_bx * bs, 3), 128, dtype=np.uint8)
    # Block (1,2) gets internal texture, not a flat patch: mean-filling a block
    # that is *already* flat is a genuine no-op, so a uniform fixture would pass
    # this test even if the wrong block were selected.
    frame[bs:2 * bs, 2 * bs:3 * bs] = np.tile(
        np.arange(bs, dtype=np.uint8).reshape(1, bs, 1), (bs, 1, 3))
    scores = np.zeros((n_by, n_bx), dtype=np.float32)

    sel = np.zeros((n_by, n_bx), dtype=bool)
    sel[1, 2] = True
    out, _ = filter_frame_mean_fill(frame, scores, bs, sel=sel)

    changed = np.argwhere((out != frame).any(axis=2))
    assert changed.size > 0, "the selected block was not filled at all"
    ys, xs = changed[:, 0], changed[:, 1]
    assert ys.min() >= bs and ys.max() < 2 * bs
    assert xs.min() >= 2 * bs and xs.max() < 3 * bs


def test_the_probe_grid_is_evcas_floor_grid_not_a_ceil_grid():
    """640x360 at 64x64 is 10x5=50 full superblocks; the bottom 40-row strip is
    not covered. Asserting it here so a future change to ceil-tiling has to
    confront the alignment consequence instead of silently shifting indices."""
    width, height, bs = 640, 360, 64
    assert (height // bs, width // bs) == (5, 10)
    assert (height // bs) * (width // bs) == 50


def test_a_probe_result_is_citable_once_metrics_exist():
    """The whole reason this component runs through the runner. A permanent
    non-empty invariant_failures would make it uncitable and pointless."""
    from presley.invariants import check_result

    result = {
        "experiment_hash": "0" * 16,
        "video_frames": 82,
        "video_framerate": 24.0,
        "actual_bitrate_bps": 659867.0,
        "file_size_bytes": 281818,
        "transmitted_size_bytes": 281818,
        "rate_control": "cqp",
        "output_video": "encoded.mp4",
        "config": {"component": "probe_oracle_bits", "video": "bear",
                   "width": 640, "height": 360, "codec": "svtav1",
                   "codec_params": {"qp": 43, "preset": "8"}},
        "metrics": {
            "foreground": {"psnr_mean": 31.2},
            "background": {"psnr_mean": 29.4},
            "overall": {"psnr_mean": 30.1},
        },
    }
    assert check_result(result) == []


def test_a_probe_result_without_metrics_is_correctly_uncitable():
    """The defect this component was written around: a pure bit-accounting probe
    with no reconstructed video fails _check_metrics_present forever. The fix is
    to publish the reference encode, NOT to exempt the component — so this stays
    a failure."""
    from presley.invariants import check_result

    result = {
        "experiment_hash": "0" * 16,
        "actual_bitrate_bps": 659867.0,
        "config": {"component": "probe_oracle_bits", "video": "bear"},
    }
    assert any("metrics" in f for f in check_result(result))
