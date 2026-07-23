"""Apply the invariant checks to the real results/ tree.

Excluded from the default run (marker `invariants`) because it needs the
results directory, which is gitignored and only exists on the host that
produced it:

    pytest -m invariants

This is the sweep that catches a directory written before a check existed, or
one whose numbers were later backfilled into an inconsistent state. It reports
every offending hash rather than stopping at the first, because the useful
output is the list to re-check.
"""

import json
from pathlib import Path

import pytest

from presley.invariants import check_result

RESULTS = Path(__file__).resolve().parents[2] / "results"

pytestmark = pytest.mark.invariants


def load_all():
    if not RESULTS.is_dir():
        pytest.skip("results/ not present on this host")
    loaded = []
    for path in sorted(RESULTS.glob("*/result.json")):
        try:
            loaded.append((path.parent.name, json.loads(path.read_text())))
        except (json.JSONDecodeError, OSError):
            continue  # a corrupt result is re-run, not asserted on
    if not loaded:
        pytest.skip("results/ has no readable result.json files")
    return loaded


def test_every_result_carries_a_verdict():
    """A missing verdict is the one state the citation rule cannot act on.

    `results-report` and `update-paper` refuse a result whose
    `invariant_failures` is non-empty — so a result with *no* verdict reads as
    fine. Runs predating a check must be backfilled:

        python -m presley.invariants results/
    """
    missing = [h for h, r in load_all() if "invariant_failures" not in r]
    assert not missing, (
        f"{len(missing)} result(s) have no invariant verdict and would be treated "
        f"as citable by default; run `python -m presley.invariants results/`. "
        f"First few: {missing[:10]}"
    )


def test_stored_verdicts_match_the_current_checks():
    """A stale verdict is worse than none: it asserts a result is clean.

    Divergence means the checks were tightened after the run, so the recorded
    verdict no longer reflects what we now know about that result.
    """
    stale = []
    for h, result in load_all():
        stored = result.get("invariant_failures")
        if stored is None:
            continue  # covered by the test above
        if sorted(stored) != sorted(check_result(result)):
            stale.append(h)
    assert not stale, (
        f"{len(stale)} result(s) carry a verdict that disagrees with the current "
        f"checks; re-run `python -m presley.invariants results/ --force`. "
        f"First few: {stale[:10]}"
    )


def test_flagged_results_are_reported():
    """Not an assertion about the tree — a standing report of what is uncitable.

    Violations are a normal research outcome; the failsafe is that they stay
    visible and recorded, not that they never happen. This prints the current
    list so it cannot quietly grow.
    """
    offenders = {h: f for h, r in load_all() if (f := check_result(r))}
    if offenders:
        print(f"\n{len(offenders)} result(s) are NOT citable:")
        for h, failures in offenders.items():
            print(f"  {h}: {'; '.join(failures)}")
