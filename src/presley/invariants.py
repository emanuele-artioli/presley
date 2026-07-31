"""Machine-checkable versions of the methodology rules in CLAUDE.md.

The rules that matter most here are the ones a wrong result does not announce:
a VBR degradation run produces a perfectly well-formed `result.json` whose
numbers mean the opposite of what the paper would claim from them, and a
restorer that makes the background worse still writes plausible PSNR. Prose in
CLAUDE.md cannot stop either from being cited months later, so the checks live
in code and their verdict is written into the result itself.

`run_single_experiment` calls `check_result` and stores the outcome under
`invariant_failures`. **A result with a non-empty `invariant_failures` is not
citable** — re-check it before it reaches a report or the paper. The
`-m invariants` pytest tier re-runs these over `results/` so an existing
directory cannot quietly drift out of compliance either.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Components that degrade the source before encoding. For these, Goal 1 only
# means anything under fixed QP/CRF: under VBR the encoder spends its bitrate
# target regardless of source complexity, so degradation cannot free bits — it
# only makes the content harder to code at that target, and the holes steal bits
# from the foreground, inverting the result. 25/25 matched VBR pairs encoded to
# MORE bits than the pristine baseline, with zero counterexamples.
DEGRADING_COMPONENTS = {"elvis", "presley_ai"}

# Rate-control modes with no bitrate target, i.e. the ones where a bitrate
# saving is attributable to the method rather than to the encoder's budget.
FIXED_QUALITY_MODES = {"cqp", "crf"}

REGIONS = ("foreground", "background", "overall")

# How much more of the output a restorer may clip to 0/255 than its own input
# did, before the run stops being evidence about the method. See
# `_check_output_not_saturated` for where 0.5 pp comes from.
SATURATION_EXCESS_LIMIT = 0.005


def _is_bad_number(value: Any) -> bool:
    """True when the value is absent or not a real measurement.

    `bool` counts as bad deliberately: it passes `isinstance(x, int)`, so a
    result carrying `psnr_mean: true` would otherwise sail through as 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    return math.isnan(value) or math.isinf(value)


def check_result(result: Dict[str, Any]) -> List[str]:
    """Everything checkable from one result alone. Returns failure descriptions."""
    failures: List[str] = []
    config = result.get("config") or {}
    component = config.get("component")

    failures += _check_metrics_present(result)
    failures += _check_bitrate_accounting(result)
    failures += _check_fixed_qp_mandate(result, component, config)
    failures += _check_restoration_did_not_hurt(result, config)
    failures += _check_output_not_saturated(result, config)
    return failures


def _check_metrics_present(result: Dict[str, Any]) -> List[str]:
    metrics = result.get("metrics")
    if not metrics:
        return ["metrics block is missing or empty"]

    failures = []
    for region in REGIONS:
        block = metrics.get(region)
        if not block:
            failures.append(f"metrics.{region} is missing")
            continue
        psnr = block.get("psnr_mean")
        if _is_bad_number(psnr):
            failures.append(f"metrics.{region}.psnr_mean is not a finite number ({psnr!r})")
        elif psnr <= 0:
            failures.append(f"metrics.{region}.psnr_mean is {psnr}, which is not a real measurement")
    return failures


def _check_bitrate_accounting(result: Dict[str, Any]) -> List[str]:
    """The bitrate axis must describe what was actually transmitted.

    `file_size_bytes` is the lossless restored output for elvis/presley_ai — a
    decode-side artifact tens of MB large — so every component computes
    `actual_bitrate_bps` from transmitted bytes instead. This confirms the two
    still agree; if they do not, the rate axis of any RD claim is measuring
    something other than the payload.
    """
    actual = result.get("actual_bitrate_bps")
    if actual is None:
        return ["actual_bitrate_bps is missing; bitrate claims cannot be made from this run"]
    if _is_bad_number(actual) or actual <= 0:
        return [f"actual_bitrate_bps is not a positive number ({actual!r})"]

    transmitted = result.get("transmitted_size_bytes")
    frames = result.get("video_frames")
    fps = result.get("video_framerate")
    if not (transmitted and frames and fps):
        return []

    duration = frames / fps
    if not duration:
        return []
    expected = (transmitted * 8) / duration
    if expected and abs(actual - expected) / expected > 0.01:
        return [
            f"actual_bitrate_bps ({actual:.0f}) disagrees with the bitrate implied by "
            f"transmitted_size_bytes ({expected:.0f}) by more than 1%"
        ]
    return []


def _check_fixed_qp_mandate(
    result: Dict[str, Any], component: Optional[str], config: Dict[str, Any]
) -> List[str]:
    """Degradation experiments must be fixed-QP/CRF. See DEGRADING_COMPONENTS."""
    if component not in DEGRADING_COMPONENTS:
        return []
    if not config.get("degradation"):
        return []

    mode = result.get("rate_control")
    if mode is None:
        return ["rate_control is missing, so the fixed-QP mandate cannot be verified"]
    if mode not in FIXED_QUALITY_MODES:
        return [
            f"degradation experiment ran under rate_control={mode!r}; under a bitrate "
            f"target degradation cannot free bits, so this run is not evidence about "
            f"the method (fixed-QP/CRF only)"
        ]
    return []


def _check_restoration_did_not_hurt(
    result: Dict[str, Any], config: Dict[str, Any]
) -> List[str]:
    """Restoration must not leave the background perceptually worse than it found it.

    Judged on LPIPS, never PSNR. A generative restorer that hallucinates plausible
    detail routinely scores *lower* BG-PSNR than the degraded input it was given —
    a frozen or flat-filled block is mathematically closer to the original than
    invented texture is — so a PSNR-based version of this check fires on precisely
    the runs where the model did its job. (Checked against the real results tree:
    a PSNR formulation flagged 39 legitimate generative runs.)

    When LPIPS is missing on either side the check stays silent rather than
    falling back to PSNR: unevaluable is not the same as failing, and a
    PSNR-shaped verdict here would be worse than none.
    """
    if not config.get("restorer"):
        return []

    metrics = result.get("metrics") or {}
    restored = (metrics.get("background") or {}).get("lpips_mean")
    degraded = ((metrics.get("transmitted") or {}).get("background") or {}).get("lpips_mean")
    if _is_bad_number(restored) or _is_bad_number(degraded):
        return []

    # LPIPS is lower-is-better. The margin is one JND (see compare.JND), so this
    # only fires on a perceptible regression, not on measurement noise.
    if restored > degraded + 0.05:
        return [
            f"restoration left the background perceptually worse than the degraded "
            f"input it received (BG LPIPS {restored:.4f} vs transmitted {degraded:.4f}; "
            f"lower is better)"
        ]
    return []


def _check_output_not_saturated(
    result: Dict[str, Any], config: Dict[str, Any]
) -> List[str]:
    """A restorer must not clip a meaningful share of pixels it was handed clean.

    A model that diverges numerically writes no NaN and no missing field: the
    garbage lands at the 8-bit limits and the run reads as well-formed with a
    merely disappointing score. NAFNet did exactly this on bike-packing —
    4.31%/4.35% of pixels at 0/255 against 0.03%/0.11% in its own transmitted
    input, 6.79 dB of BG-PSNR destroyed, `invariant_failures` empty
    (research-log/bugs.md, top entry).

    Judged as an *excess* over the transmitted input rather than an absolute
    level, because legitimately dark or blown-out content saturates on both
    sides and is not a defect. The threshold is 0.5 percentage points: the three
    damaged bike-packing runs sit at 8.7 / 6.5 / 1.1 pp of excess (the mildest
    still costing 0.86 dB) while the healthy control is at 0.05 pp, so the gap
    between "diverged" and "fine" is two orders of magnitude wide and the exact
    cut is not load-bearing.

    **Applies to new runs only, by construction.** `metrics.saturation` is
    written by `evaluation.run`, so results evaluated before that existed carry
    no saturation block and this check stays silent on them — they keep the
    citability status they were assessed under. That is a deliberate decision,
    not an oversight: re-judging already-cited runs is a separate call. Re-run
    an old experiment's evaluation if you want it covered.
    """
    if not config.get("restorer"):
        return []

    saturation = result.get("metrics", {}).get("saturation")
    if not saturation:
        return []

    output = saturation.get("output_clipped_frac")
    transmitted = saturation.get("transmitted_clipped_frac")
    if _is_bad_number(output) or _is_bad_number(transmitted):
        return []

    excess = output - transmitted
    if excess > SATURATION_EXCESS_LIMIT:
        return [
            f"restoration clipped {output:.2%} of output pixels to 0/255 against "
            f"{transmitted:.2%} in the input it received (+{excess:.2%}, limit "
            f"{SATURATION_EXCESS_LIMIT:.2%}); this is the signature of a numerical "
            f"divergence, not of a restorer that merely underperformed"
        ]
    return []


def backfill(results_dir: str = "results", force: bool = False) -> Dict[str, List[str]]:
    """Write an `invariant_failures` verdict into every result.json in place.

    Runs written before a check existed carry no verdict, which is the one state
    the citation rule cannot act on — `results-report` and `update-paper` refuse
    results whose verdict is non-empty, so a *missing* verdict reads as "fine".
    This is a metadata-only pass: no re-encoding, no re-evaluation, and it is
    re-entrant, and safe to run concurrently with another backfill over the
    same tree.

    Returns the offending hashes and their failures.
    """
    import os

    from presley import db as _db

    # Still driven by the on-disk directory listing rather than by the DB: this
    # sweep's whole job is to reach runs the bookkeeping has not caught up with,
    # including ones written before the DB existed. Reads and writes go through
    # the DB helpers, so the verdict lands in both stores and the unique-temp-name
    # write (the race in research-log/bugs.md) is handled once inside write_mirror.
    offenders: Dict[str, List[str]] = {}
    for entry in sorted(os.listdir(results_dir)):
        if entry.startswith("_") or entry.startswith("."):
            continue
        if not os.path.isfile(os.path.join(results_dir, entry, "result.json")):
            continue
        try:
            result = _db.load_run(results_dir, entry)
        except (ValueError, OSError):
            continue
        if result is None:
            continue
        if "invariant_failures" in result and not force:
            if result["invariant_failures"]:
                offenders[entry] = result["invariant_failures"]
            continue
        failures = check_result(result)
        result["invariant_failures"] = failures
        _db.save_run(results_dir, entry, result)
        if failures:
            offenders[entry] = failures
    return offenders


def main() -> None:
    """`python -m presley.invariants [results_dir] [--force]`"""
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    results_dir = args[0] if args else "results"
    offenders = backfill(results_dir, force="--force" in sys.argv[1:])
    if not offenders:
        print(f"{results_dir}: every result satisfies its invariants")
        return
    print(f"{len(offenders)} result(s) are NOT citable:")
    for hash_id, failures in offenders.items():
        print(f"  {hash_id}")
        for failure in failures:
            print(f"    - {failure}")


def check_goal1(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> List[str]:
    """Goal 1: at matched quality the method must cost fewer bits, not more.

    Only meaningful for two runs of the same video at the same rate-control
    setting, and only when both are fixed-quality — caller's responsibility.
    """
    from presley.compare import same_quality

    outcome = same_quality(baseline, candidate, region="foreground")
    if outcome.verdict == "distinguishable":
        return []  # a real quality difference; the bitrate comparison is not like-for-like
    if outcome.bitrate_delta_pct is None:
        return ["bitrate missing on one side; Goal 1 cannot be evaluated"]
    if outcome.bitrate_delta_pct > 0:
        return [
            f"at indistinguishable foreground quality the method cost "
            f"{outcome.bitrate_delta_pct:+.1f}% more bits than the baseline"
        ]
    return []


if __name__ == "__main__":
    main()
