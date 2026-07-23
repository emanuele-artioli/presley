"""The JND gate.

`presley-compare` decides whether a quality difference is real, so it stands
between every measured delta and every sentence in the paper. Two failure modes
matter more than the rest: calling an imperceptible delta a result, and reading
a union-bbox key as if it were a foreground metric. Both produce a plausible
number rather than an error, so both are pinned here.
"""

import pytest

from presley.compare import (
    BANNED_FG_KEYS,
    JND,
    REGION_METRIC_KEYS,
    _metric_value,
    bitrate_bps,
    compare_metric,
    group_experiments,
    pick_baseline,
    same_quality,
)


def result(hash_id="aaaa", *, foreground=None, background=None, overall=None, **top):
    """A result.json-shaped dict with only the fields the comparison reads."""
    metrics = {}
    if foreground is not None:
        metrics["foreground"] = foreground
    if background is not None:
        metrics["background"] = background
    if overall is not None:
        metrics["overall"] = overall
    return {"experiment_hash": hash_id, "metrics": metrics, **top}


# --- foreground citability -----------------------------------------------------


def test_foreground_never_exposes_a_banned_key():
    """The FG row must not offer vmaf/fvmd/dists_mean/fid at all.

    These are union-bbox artifacts — up to 100% of the frame on some videos —
    so a "foreground" number built on them is a whole-frame number wearing the
    wrong label.
    """
    assert not set(REGION_METRIC_KEYS["foreground"].values()) & BANNED_FG_KEYS


def test_reading_a_banned_key_for_foreground_raises(monkeypatch):
    """Even if the key table is hand-edited, the guard must still fire."""
    monkeypatch.setitem(REGION_METRIC_KEYS["foreground"], "vmaf", "vmaf_fg_bbox")

    with pytest.raises(ValueError, match="banned union-bbox key"):
        _metric_value(result(foreground={"vmaf_fg_bbox": 90.0}), "foreground", "vmaf")


def test_foreground_dists_reads_the_true_masked_key():
    key, value = _metric_value(
        result(foreground={"dists_fg": 0.1, "dists_mean": 0.9}), "foreground", "dists"
    )
    assert key == "dists_fg"
    assert value == 0.1


def test_a_missing_metric_is_unavailable_rather_than_an_error():
    key, value = _metric_value(result(foreground={}), "foreground", "lpips")
    assert key == "lpips_mean"
    assert value is None


# --- the JND verdict -----------------------------------------------------------


def test_a_sub_jnd_delta_is_not_a_difference():
    """The rule the whole tool exists for: small deltas are not trends."""
    lpips_jnd = JND["lpips"][0]
    a = result("a", foreground={"lpips_mean": 0.30, "psnr_mean": 35.0})
    b = result("b", foreground={"lpips_mean": 0.30 + lpips_jnd / 2, "psnr_mean": 35.1})

    assert same_quality(a, b).verdict == "indistinguishable"


def test_a_supra_jnd_delta_is_a_difference():
    lpips_jnd = JND["lpips"][0]
    a = result("a", foreground={"lpips_mean": 0.30, "psnr_mean": 35.0})
    b = result("b", foreground={"lpips_mean": 0.30 + lpips_jnd * 3, "psnr_mean": 35.0})

    outcome = same_quality(a, b)
    assert outcome.verdict == "distinguishable"
    assert "lpips_mean" in outcome.binding_metric


def test_psnr_alone_cannot_claim_perceptual_parity():
    """A fast_only result has PSNR and nothing else.

    Reporting that as plain "indistinguishable" is how mean_fill — the highest
    PSNR and the perceptually worst output — would get written up as a win.
    """
    a = result("a", foreground={"psnr_mean": 35.0})
    b = result("b", foreground={"psnr_mean": 35.2})

    outcome = same_quality(a, b)
    assert outcome.verdict == "indistinguishable_psnr_only"
    assert not outcome.perceptual_backing
    assert outcome.warnings


def test_no_gating_metric_means_no_verdict():
    a = result("a", foreground={})
    b = result("b", foreground={})
    assert same_quality(a, b).verdict == "insufficient_data"


def test_fvmd_never_gates_a_verdict():
    """FVMD has no established JND, so it may inform but must not decide."""
    a = result("a", foreground={"psnr_mean": 35.0, "fvmd": 10.0})
    b = result("b", foreground={"psnr_mean": 35.1, "fvmd": 5000.0})

    outcome = same_quality(a, b)
    assert outcome.verdict != "distinguishable"
    fvmd = next(m for m in outcome.metrics if m.metric == "fvmd")
    assert fvmd.available and not fvmd.gates_verdict


def test_the_worst_metric_is_the_binding_one():
    """When several metrics fail, the report must name the largest violation."""
    a = result("a", foreground={"psnr_mean": 35.0, "lpips_mean": 0.30})
    b = result(
        "b",
        foreground={
            "psnr_mean": 35.0 + JND["psnr"][0] * 2,
            "lpips_mean": 0.30 + JND["lpips"][0] * 10,
        },
    )
    assert "lpips_mean" in same_quality(a, b).binding_metric


def test_lower_is_better_metrics_are_declared_as_such():
    """Direction drives how a delta is worded; a flip would invert every claim."""
    assert JND["lpips"][1] is True
    assert JND["dists"][1] is True
    assert JND["psnr"][1] is False
    assert JND["ssim"][1] is False


def test_comparison_delta_is_b_minus_a():
    a = result("a", foreground={"psnr_mean": 30.0})
    b = result("b", foreground={"psnr_mean": 32.0})
    assert compare_metric(a, b, "foreground", "psnr").delta == pytest.approx(2.0)


# --- bitrate -------------------------------------------------------------------


def test_bitrate_reads_the_actual_field():
    value, warnings = bitrate_bps(result(actual_bitrate_bps=1234.0))
    assert value == 1234.0
    assert not warnings


def test_bitrate_flags_an_accounting_mismatch():
    """actual_bitrate_bps must agree with the transmitted bytes it came from.

    A silent disagreement means the bitrate axis of a rate-distortion claim is
    measuring something other than what was sent.
    """
    _, warnings = bitrate_bps(
        result(
            actual_bitrate_bps=1_000_000.0,
            transmitted_size_bytes=1000,  # 8 kbit over 1 s — nowhere near 1 Mbps
            video_frames=30,
            video_framerate=30.0,
        )
    )
    assert warnings


def test_consistent_accounting_produces_no_warning():
    # 30 frames at 30 fps is 1 s; 125000 bytes is exactly 1 Mbit.
    _, warnings = bitrate_bps(
        result(
            actual_bitrate_bps=1_000_000.0,
            transmitted_size_bytes=125_000,
            video_frames=30,
            video_framerate=30.0,
        )
    )
    assert not warnings


def test_percent_delta_is_relative_to_a():
    a = result("a", foreground={"psnr_mean": 35.0}, actual_bitrate_bps=1000.0)
    b = result("b", foreground={"psnr_mean": 35.0}, actual_bitrate_bps=900.0)
    assert same_quality(a, b).bitrate_delta_pct == pytest.approx(-10.0)


# --- grouping ------------------------------------------------------------------


def test_grouping_splits_on_every_requested_key():
    entries = [
        {"config": {"component": "baselines", "video": "tennis"}},
        {"config": {"component": "elvis", "video": "tennis"}},
        {"config": {"component": "baselines", "video": "bear"}},
    ]
    groups = group_experiments(entries, ["component", "video"])
    assert len(groups) == 3


def test_grouping_reads_dotted_config_keys():
    """Matched-QP sweeps group on codec_params.qp, which is nested."""
    entries = [
        {"config": {"component": "baselines", "codec_params": {"qp": 30}}},
        {"config": {"component": "baselines", "codec_params": {"qp": 35}}},
    ]
    assert len(group_experiments(entries, ["codec_params.qp"])) == 2


def test_baseline_pick_finds_the_named_component():
    group = [
        {"config": {"component": "elvis"}},
        {"config": {"component": "baselines"}},
    ]
    assert pick_baseline(group, "baselines")["config"]["component"] == "baselines"


def test_baseline_pick_returns_none_when_absent():
    """No baseline in the group means no comparison target, not a wrong one."""
    group = [{"config": {"component": "elvis"}}]
    assert pick_baseline(group, "baselines") is None
