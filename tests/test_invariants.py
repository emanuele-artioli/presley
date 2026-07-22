"""Unit tests for the invariant checks themselves.

These run in the fast tier on synthetic results. The `-m invariants` tier in
tests/invariants/ applies the same checks to the real results/ tree.
"""

import pytest

from presley.invariants import check_goal1, check_result


def good_result(**overrides):
    """A result that satisfies every invariant, for tests to perturb."""
    base = {
        "experiment_hash": "aaaa",
        "actual_bitrate_bps": 1_000_000.0,
        "transmitted_size_bytes": 125_000,  # exactly 1 Mbit over 1 s
        "video_frames": 30,
        "video_framerate": 30.0,
        "rate_control": "cqp",
        "config": {"component": "baselines", "video": "tennis"},
        "metrics": {
            "foreground": {"psnr_mean": 35.0},
            "background": {"psnr_mean": 30.0},
            "overall": {"psnr_mean": 32.0},
        },
    }
    base.update(overrides)
    return base


def test_a_well_formed_result_has_no_failures():
    assert check_result(good_result()) == []


def test_missing_metrics_are_reported():
    assert check_result(good_result(metrics={}))


def test_a_null_psnr_is_a_failure():
    """A null metric is a failed measurement wearing the shape of a result."""
    result = good_result()
    result["metrics"]["background"]["psnr_mean"] = None

    failures = check_result(result)
    assert any("psnr_mean" in f for f in failures)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5.0, 0.0])
def test_implausible_psnr_values_are_rejected(bad):
    result = good_result()
    result["metrics"]["foreground"]["psnr_mean"] = bad
    assert check_result(result)


def test_bitrate_disagreeing_with_transmitted_bytes_is_a_failure():
    assert check_result(good_result(transmitted_size_bytes=1000))


def test_a_missing_bitrate_is_a_failure():
    assert check_result(good_result(actual_bitrate_bps=None))


# --- the fixed-QP mandate ------------------------------------------------------


@pytest.mark.parametrize("mode", ["vbr_1pass", "vbr_2pass"])
def test_a_vbr_degradation_run_is_flagged(mode):
    """The rule this module exists for.

    Under a bitrate target the encoder spends it regardless of source
    complexity, so degradation cannot free bits — the run measures the encoder,
    not the method, and its numbers invert Goal 1.
    """
    result = good_result(
        rate_control=mode,
        config={"component": "elvis", "video": "tennis", "degradation": "freeze"},
    )
    failures = check_result(result)
    assert any("fixed-QP" in f for f in failures)


@pytest.mark.parametrize("mode", ["cqp", "crf"])
def test_a_fixed_quality_degradation_run_passes(mode):
    result = good_result(
        rate_control=mode,
        config={"component": "presley_ai", "video": "tennis", "degradation": "blackout"},
    )
    assert check_result(result) == []


def test_vbr_is_fine_for_a_non_degrading_component():
    """Baselines legitimately run under VBR; the mandate is about degradation."""
    result = good_result(rate_control="vbr_2pass", config={"component": "baselines"})
    assert check_result(result) == []


def test_missing_rate_control_on_a_degradation_run_is_a_failure():
    """Unverifiable is not the same as compliant."""
    result = good_result(config={"component": "elvis", "degradation": "freeze"})
    del result["rate_control"]
    assert check_result(result)


# --- restoration ---------------------------------------------------------------


def restored(bg_lpips, transmitted_lpips, **config):
    result = good_result(config={"component": "presley_ai", "restorer": "propainter", **config})
    result["metrics"]["background"]["lpips_mean"] = bg_lpips
    result["metrics"]["transmitted"] = {"background": {"lpips_mean": transmitted_lpips}}
    return result


def test_a_perceptually_worse_background_is_flagged():
    failures = check_result(restored(0.40, 0.20))
    assert any("perceptually worse" in f for f in failures)


def test_a_perceptually_better_background_passes():
    assert check_result(restored(0.20, 0.40)) == []


def test_restoration_is_never_judged_on_psnr():
    """A generative restorer legitimately lowers BG-PSNR while looking better.

    Flat or frozen fill is mathematically closer to the original than invented
    texture, so a PSNR formulation of this check rewards a model for declining
    to hallucinate. Against the real results tree it flagged 39 healthy runs.
    """
    result = good_result(config={"component": "presley_ai", "restorer": "propainter"})
    result["metrics"]["background"]["psnr_mean"] = 25.0
    result["metrics"]["transmitted"] = {"background": {"psnr_mean": 31.0}}

    assert check_result(result) == []


def test_a_sub_jnd_lpips_regression_is_tolerated():
    assert check_result(restored(0.212, 0.200)) == []


def test_the_check_is_skipped_when_perceptual_data_is_missing():
    """Unevaluable is not failing — and must not silently become a PSNR check."""
    result = good_result(config={"component": "presley_ai", "restorer": "propainter"})
    result["metrics"]["transmitted"] = {"background": {"psnr_mean": 99.0}}
    assert check_result(result) == []


def test_restoration_check_is_skipped_without_a_restorer():
    result = good_result(config={"component": "baselines"})
    result["metrics"]["background"]["lpips_mean"] = 0.9
    result["metrics"]["transmitted"] = {"background": {"lpips_mean": 0.1}}
    assert check_result(result) == []


# --- Goal 1 --------------------------------------------------------------------


def test_goal1_flags_a_method_that_costs_more_at_equal_quality():
    baseline = good_result(actual_bitrate_bps=1_000_000.0)
    candidate = good_result(actual_bitrate_bps=1_200_000.0)
    candidate["experiment_hash"] = "bbbb"

    failures = check_goal1(baseline, candidate)
    assert any("more bits" in f for f in failures)


def test_goal1_passes_when_the_method_is_cheaper():
    baseline = good_result(actual_bitrate_bps=1_000_000.0)
    candidate = good_result(actual_bitrate_bps=800_000.0)

    assert check_goal1(baseline, candidate) == []


def test_goal1_is_not_evaluated_across_a_real_quality_difference():
    """Comparing bitrates at different quality is not a like-for-like claim."""
    baseline = good_result()
    candidate = good_result(actual_bitrate_bps=1_500_000.0)
    candidate["metrics"]["foreground"]["psnr_mean"] = 45.0  # far outside JND

    assert check_goal1(baseline, candidate) == []
