"""Tests for tools/analyze_throughput.py.

The load-bearing claim of Wave 2C is negative: that a pooled median over a
shared, contended box can invert the ordering between two resolutions, and that
the tool refuses to publish such a median. These tests pin the refusal, since a
silent failure of it would put an impossible number back in the paper.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))

import analyze_throughput as A  # noqa: E402


# --------------------------------------------------------------------------
# Robust statistics
# --------------------------------------------------------------------------

def test_iqr_of_single_point_is_zero():
    assert A.iqr([5.0]) == 0.0


def test_iqr_of_empty_is_zero():
    assert A.iqr([]) == 0.0


def test_iqr_matches_linear_interpolation():
    # quartiles of 1..5 by the (n-1) position rule are 2 and 4
    assert A.iqr([1, 2, 3, 4, 5]) == pytest.approx(2.0)


def test_iqr_ignores_a_single_outlier_more_than_range_does():
    tight = [10, 10, 10, 10, 10, 10, 10, 1000]
    assert A.iqr(tight) < (max(tight) - min(tight))


# --------------------------------------------------------------------------
# Batch clustering
# --------------------------------------------------------------------------

def test_cluster_batches_splits_on_a_large_gap():
    t = [0, 60, 120, 10 * 3600, 10 * 3600 + 60]
    assert A.cluster_batches(t, gap_s=3600) == [0, 0, 0, 1, 1]


def test_cluster_batches_keeps_one_batch_when_evenly_spaced():
    assert A.cluster_batches([0, 100, 200, 300], gap_s=3600) == [0, 0, 0, 0]


def test_cluster_batches_preserves_caller_order():
    # unsorted input; ids must line up with the positions given, not the sort
    ids = A.cluster_batches([10 * 3600, 0, 60], gap_s=3600)
    assert ids == [1, 0, 0]


def test_cluster_batches_of_empty_is_empty():
    assert A.cluster_batches([]) == []


def test_cluster_batches_of_one_is_batch_zero():
    assert A.cluster_batches([12345.0]) == [0]


# --------------------------------------------------------------------------
# Disjointness -- the mixture detector
# --------------------------------------------------------------------------

def test_ranges_disjoint_true_when_separated():
    assert A.ranges_disjoint([(0.7, 1.2), (23.8, 37.4)]) is True


def test_ranges_disjoint_false_when_overlapping():
    assert A.ranges_disjoint([(16.6, 45.4), (23.3, 36.0)]) is False


def test_ranges_disjoint_false_when_merely_touching():
    # sharing an endpoint means a value exists in both -- not a mixture
    assert A.ranges_disjoint([(1.0, 2.0), (2.0, 3.0)]) is False


def test_ranges_disjoint_true_for_a_single_range():
    assert A.ranges_disjoint([(1.0, 2.0)]) is True


# --------------------------------------------------------------------------
# The cell verdict
# --------------------------------------------------------------------------

def _cell(batches, **kw):
    c = A.EncodeCell(codec="svtav1", width=1280, height=720, preset="8", **kw)
    c.batches = [A.BatchRow(batch=i, n=n, med=med, lo=lo, hi=hi,
                            spread=spread, qps=[43], when=float(i))
                 for i, (n, med, lo, hi, spread) in enumerate(batches)]
    return c


def test_disjoint_batches_are_not_measurable():
    # the real svtav1 720p cell: 0.7-1.2 fps and 23.8-37.4 fps pooled
    cell = _cell([(8, 1.0, 0.68, 1.24, 0.2), (8, 34.5, 23.78, 37.37, 0.2)])
    verdict, reason = A.judge_cell(cell)
    assert verdict == "not measurable"
    assert "non-overlapping" in reason


def test_overlapping_batches_are_measurable():
    # the real svtav1 1080p cell: two batches that share values
    cell = _cell([(8, 31.4, 16.56, 45.40, 0.3), (8, 28.8, 23.30, 36.02, 0.2)])
    assert A.judge_cell(cell)[0] == "ok"


def test_too_few_runs_is_not_measurable():
    cell = _cell([(2, 39.6, 38.5, 40.7, 0.05)])
    verdict, reason = A.judge_cell(cell)
    assert verdict == "not measurable"
    assert "n=2" in reason


def test_wide_within_batch_spread_is_flagged():
    cell = _cell([(14, 12.6, 1.18, 38.10, 1.71)])
    verdict, reason = A.judge_cell(cell)
    assert verdict == "wide"
    assert "contention" in reason


def test_a_clean_single_batch_passes():
    cell = _cell([(8, 34.5, 23.78, 37.37, 0.2)])
    assert A.judge_cell(cell) == ("ok", "")


# --------------------------------------------------------------------------
# Timing arithmetic
# --------------------------------------------------------------------------

def _timing(**kw):
    base = dict(hash="h", component="baselines", video="bear", codec="svtav1",
                width=640, height=360, preset="8", qp=43, frames=82,
                encode_s=2.0, restore_s=None, transport=None, fill=None)
    base.update(kw)
    return A.Timing(**base)


def test_encode_fps_is_frames_over_seconds():
    assert _timing().encode_fps == pytest.approx(41.0)


def test_zero_encode_time_yields_no_fps_rather_than_dividing():
    assert _timing(encode_s=0.0).encode_fps is None


def test_missing_frame_count_yields_no_fps():
    assert _timing(frames=None).encode_fps is None


def test_restore_fps_is_none_for_a_baseline_with_no_restoration():
    assert _timing().restore_fps is None


def test_wall_fps_sums_encode_and_restore():
    t = _timing(encode_s=2.0, restore_s=18.0)
    assert t.wall_fps == pytest.approx(82 / 20.0)


def test_wall_fps_is_none_when_nothing_was_timed():
    assert _timing(encode_s=None, restore_s=None).wall_fps is None


# --------------------------------------------------------------------------
# Cell construction end to end, on synthetic runs
# --------------------------------------------------------------------------

def test_build_encode_cells_separates_batches_and_refuses_the_pooled_median():
    slow = [_timing(encode_s=100.0, mtime=0.0, qp=34) for _ in range(4)]
    fast = [_timing(encode_s=2.5, mtime=20 * 24 * 3600.0, qp=58) for _ in range(4)]
    cells = A.build_encode_cells(slow + fast)
    assert len(cells) == 1
    assert len(cells[0].batches) == 2
    assert cells[0].verdict == "not measurable"


def test_build_encode_cells_ignores_non_baseline_components():
    assert A.build_encode_cells([_timing(component="presley_ai")]) == []
