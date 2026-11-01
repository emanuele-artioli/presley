"""Reporting margins must not silently become claims about human perception."""
import pytest

from presley.compare import (
    JND, REPORTING_MARGINS, GroupComparison, _to_jsonable,
    format_groups, format_pairwise, same_quality,
)
from presley.suite import PairedDelta, SuiteComparison, assess_metric, format_suite


def test_pair_reports_a_margin_without_claiming_perceptual_equality():
    def result(lpips):
        return {"metrics": {"foreground": {"lpips_mean": lpips}}}

    comparison = same_quality(result(0.20), result(0.24))
    # Historical codes remain readable by existing analyses.
    assert comparison.verdict == "indistinguishable"
    assert JND is REPORTING_MARGINS
    assert comparison.metrics[2].delta == pytest.approx(0.04)
    report = format_pairwise(comparison)
    assert "verdict: within_reporting_margins" in report
    assert "not calibrated perceptual thresholds" in report
    assert "verdict: indistinguishable" not in report
    exported = _to_jsonable(comparison)
    assert exported["margin_kind"] == "uncalibrated_reporting_margin"
    assert exported["legacy_verdict"] == "indistinguishable"
    assert exported["reporting_verdict"] == "within_reporting_margins"


@pytest.mark.parametrize("delta,legacy,interpretation", [
    (-0.02, "sub_jnd_significant", "significant_metric_effect_below_reporting_margin"),
    (-0.10, "perceptual_win", "significant_metric_improvement_above_reporting_margin"),
    (+0.10, "perceptual_loss", "significant_metric_degradation_above_reporting_margin"),
])
def test_suite_preserves_evidence_and_uses_metric_wording(delta, legacy, interpretation):
    pairs = [PairedDelta((f"v{i}",), f"a{i}", f"b{i}", 0.5, 0.5 + delta,
                         delta, abs(delta) < JND["lpips"][0]) for i in range(6)]
    result = assess_metric(pairs, "lpips", "background", is_primary=True)
    assert result.verdict == legacy
    assert result.sign_p == pytest.approx(0.03125)
    assert result.mean_delta == pytest.approx(delta)
    assert result.reporting_verdict == interpretation
    assert "imperceptible" not in result.wording
    suite = SuiteComparison(arm_a="a", arm_b="b", region="background",
                            pair_by=["video"], family_size=1)
    suite.metrics.append(result)
    report = format_suite(suite)
    assert f"VERDICT: {interpretation}" in report
    assert f"VERDICT: {legacy}" not in report
    assert "not calibrated perceptual thresholds" in report
    exported = _to_jsonable(suite)["metrics"][0]
    assert exported["legacy_verdict"] == legacy
    assert exported["reporting_verdict"] == interpretation
    assert exported["sign_p"] == pytest.approx(0.03125)


def test_group_bitrate_choice_is_explicitly_conditioned_on_reporting_margins():
    group = GroupComparison(("bear",), ["video"], ["a", "b"], "a", [],
                            True, "b", -10.0)
    report = format_groups([group])
    assert "within_reporting_margins: True" in report
    assert "quality_indistinguishable:" not in report
    assert "bitrate winner: b (-10.0% vs baseline)" in report
    exported = _to_jsonable(group)
    assert exported["quality_indistinguishable"] is True  # compatibility
    assert exported["within_reporting_margins"] is True
    assert exported["margin_kind"] == "uncalibrated_reporting_margin"
