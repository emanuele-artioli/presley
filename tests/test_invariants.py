"""Unit tests for the invariant checks themselves.

These run in the fast tier on synthetic results. The `-m invariants` tier in
tests/invariants/ applies the same checks to the real results/ tree.
"""

import json
import multiprocessing
import os

import pytest

from presley.invariants import backfill, check_goal1, check_result


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


def test_sticky_pre_metrics_failure_clears_once_metrics_exist():
    """Runner writes invariant_failures before evaluate_all fills metrics.

    A result that was checked too early must become citable after metrics land
    (the evaluation path re-runs check_result; this asserts the contract).
    """
    premature = good_result()
    del premature["metrics"]
    premature["invariant_failures"] = check_result(premature)
    assert premature["invariant_failures"], "expected a metrics-missing failure"
    assert any("metrics" in f for f in premature["invariant_failures"])

    healed = dict(premature)
    healed["metrics"] = good_result()["metrics"]
    healed["invariant_failures"] = check_result(healed)
    assert healed["invariant_failures"] == []


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


# --- saturation ----------------------------------------------------------------


def saturated(output_frac, transmitted_frac, **config):
    result = good_result(config={"component": "presley_ai", "restorer": "nafnet", **config})
    result["metrics"]["saturation"] = {
        "output_clipped_frac": output_frac,
        "transmitted_clipped_frac": transmitted_frac,
        "pixels": 30 * 640 * 360 * 3,
    }
    return result


def test_the_nafnet_divergence_is_flagged():
    """The exact run the check exists for: `6ccb1ab63e10b7d6`, bike-packing k=7.

    8.66% of output pixels at 0/255 against 0.14% in its transmitted input, and
    6.79 dB of BG-PSNR destroyed, with `invariant_failures` empty at the time.
    """
    failures = check_result(saturated(0.0866, 0.0014))
    assert any("clipped" in f for f in failures)


def test_the_mildest_damaged_run_is_still_flagged():
    """k=31, `0e16465e6d71cc84`: 1.12% vs 0.14%, and still costs 0.86 dB.

    The threshold has to sit below this one — a divergence that only shows up
    on 2 frames of 69 is exactly the case a human reading a table would miss.
    """
    assert check_result(saturated(0.0112, 0.0014))


def test_content_that_was_already_saturated_passes():
    """Dark or blown-out source content saturates on both sides and is not a defect.

    Judged on the excess, so a 30%-black clip is not penalised for being black —
    only for the restorer adding clipping the input did not have.
    """
    assert check_result(saturated(0.302, 0.300)) == []


def test_the_healthy_control_passes():
    """`bear` at the same k=7 NAFNet setting: 0.09% out, 0.04% in, +0.13 dB."""
    assert check_result(saturated(0.0009, 0.0004)) == []


def test_a_run_evaluated_before_the_check_existed_is_left_alone():
    """The decision was to apply this to new runs only.

    Old results carry no `metrics.saturation`, so they keep the citability
    status they were assessed under rather than silently going uncitable.
    """
    result = good_result(config={"component": "presley_ai", "restorer": "nafnet"})
    assert check_result(result) == []


def test_saturation_without_a_transmitted_side_is_not_judged():
    """No input to compare against means no way to tell divergence from content."""
    result = good_result(config={"component": "presley_ai", "restorer": "nafnet"})
    result["metrics"]["saturation"] = {"output_clipped_frac": 0.5, "pixels": 100}
    assert check_result(result) == []


def test_saturation_check_is_skipped_without_a_restorer():
    """A blackout degradation writes whole black blocks — 0/255 by design."""
    result = saturated(0.40, 0.001, component="elvis")
    result["config"].pop("restorer")
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


def _backfill_worker(results_dir, ready, go, error_queue):
    """One concurrent backfill sweep, started only once its sibling is ready."""
    try:
        ready.set()
        go.wait(timeout=30)
        for _ in range(5):  # several sweeps, to widen the interleaving window
            backfill(results_dir, force=True)
    except BaseException as exc:  # pragma: no cover - reported via the queue
        error_queue.put(f"{type(exc).__name__}: {exc}")


def test_two_concurrent_backfills_over_one_tree_both_succeed(tmp_path):
    """Two runners share one results/ tree; neither may die on the other's temp file.

    Observed 2026-07-30: `backfill` wrote a fixed `<path>.tmp`, so the second
    process's `os.replace` hit FileNotFoundError after the first had already
    consumed it — a completed experiment wave exiting non-zero, and a plausible
    lost write. Built against tmp_path: the real results/ tree is gitignored and
    is never a test target.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hashes = [f"{i:016x}" for i in range(60)]
    for hash_id in hashes:
        run_dir = results_dir / hash_id
        run_dir.mkdir()
        (run_dir / "result.json").write_text(
            json.dumps(good_result(experiment_hash=hash_id))
        )

    ctx = multiprocessing.get_context("fork")
    errors = ctx.Queue()
    go = ctx.Event()
    ready = [ctx.Event(), ctx.Event()]
    procs = [
        ctx.Process(
            target=_backfill_worker, args=(str(results_dir), r, go, errors)
        )
        for r in ready
    ]
    for proc in procs:
        proc.start()
    for r in ready:
        assert r.wait(timeout=30), "a backfill worker never started"
    go.set()
    for proc in procs:
        proc.join(timeout=120)

    reported = []
    while not errors.empty():
        reported.append(errors.get())
    assert reported == [], f"concurrent backfill failed: {reported}"
    assert [p.exitcode for p in procs] == [0, 0]

    # Every result survived intact, and no temp file was orphaned in the tree.
    for hash_id in hashes:
        run_dir = results_dir / hash_id
        payload = json.loads((run_dir / "result.json").read_text())
        assert payload["experiment_hash"] == hash_id
        assert payload["invariant_failures"] == []
        assert [f for f in os.listdir(run_dir) if f != "result.json"] == []
