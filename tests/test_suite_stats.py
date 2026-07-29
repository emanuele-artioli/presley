"""The suite significance layer.

This module's whole reason to exist is to let a sub-JND effect be reported at
all, so the tests that matter are the ones pinning the guardrails that stop it
becoming a laundering route for imperceptible deltas:

  - the p-value floor at small n (the finding that a 5/5 suite is underpowered,
    not significant -- if this regresses the project starts citing p=0.031
    one-tailed values again),
  - a sub-JND significant effect never getting a `perceptual_win` verdict,
  - `same_quality`'s existing single-pair verdicts staying bit-identical,
  - poisoned runs (`invariant_failures`) never entering a suite.

Deliberately not tested: the bootstrap's coverage properties (that is a
property of the percentile bootstrap, not of this code) and the exact
enumeration counts of Wilcoxon beyond the small hand-checkable cases.
"""
import math

import pytest

from presley.compare import JND, same_quality
from presley.suite import (
    ALPHA,
    PairedDelta,
    _n_needed,
    assess_metric,
    bootstrap_ci,
    cohens_dz,
    collect_pairs,
    equivalence_tost,
    holm_adjust,
    min_attainable_sign_p,
    sign_test_p,
    wilcoxon_signed_rank_p,
)


def pairs_from(deltas, metric="lpips"):
    jnd = JND[metric][0]
    return [
        PairedDelta((f"v{i}",), f"a{i}", f"b{i}", 0.5, 0.5 + d, d, abs(d) < jnd)
        for i, d in enumerate(deltas)
    ]


# --- the p-value floor: the finding that motivated the whole module ------------


def test_five_out_of_five_is_underpowered_not_significant():
    """F3's motivating case. 5/5 same-direction LPIPS deltas look decisive and
    are not: the exact two-tailed sign test bottoms out at 0.0625 > 0.05, so no
    n=5 suite can ever be significant. The often-quoted 0.031 is one-tailed,
    which is indefensible when the direction was read off the data first."""
    f3_lpips = [-0.0150, -0.0172, -0.0173, -0.0270, -0.0305]
    result = assess_metric(pairs_from(f3_lpips), "lpips", "foreground", is_primary=True)

    assert result.n == 5
    assert result.direction == "better"
    assert result.consistent is True
    assert result.sign_p == pytest.approx(0.0625)
    assert result.min_attainable_p == pytest.approx(0.0625)
    assert result.underpowered is True
    assert result.significant is False
    assert result.verdict == "underpowered"
    # The remedy has to travel with the verdict, or "underpowered" reads as "no".
    assert "n>=6" in " ".join(result.warnings)


def test_six_pairs_is_the_first_size_that_can_clear_alpha():
    assert min_attainable_sign_p(5) == pytest.approx(0.0625)
    assert min_attainable_sign_p(6) == pytest.approx(0.03125)
    assert _n_needed(ALPHA, family_size=1) == 6


def test_correction_family_raises_the_n_required():
    """Shopping for a winner costs pairs, not just p. With the 6 restorer
    candidates of docs/EXPERIMENTS_QUEUED.md, 6 pairs no longer suffice."""
    assert _n_needed(ALPHA, family_size=6) > 6
    six_pairs = assess_metric(pairs_from([-0.01] * 6), "lpips", "foreground",
                              family_size=6, is_primary=True)
    assert six_pairs.verdict == "underpowered"
    uncorrected = assess_metric(pairs_from([-0.01] * 6), "lpips", "foreground",
                                family_size=1, is_primary=True)
    assert uncorrected.significant is True


# --- the guardrail: significance never launders a sub-JND effect --------------


def test_significant_sub_jnd_effect_is_labelled_not_promoted():
    """Seven unanimous pairs at 1/5 of the LPIPS JND: real, reproducible, and
    still invisible. It must earn a label, never a win."""
    result = assess_metric(pairs_from([-0.01] * 7), "lpips", "foreground", is_primary=True)
    assert result.significant is True
    assert result.clears_jnd is False
    assert result.verdict == "sub_jnd_significant"
    assert "never as a perceptual win" in result.wording


def test_perceptual_win_needs_magnitude_on_a_majority_of_pairs():
    """One big video must not carry a suite of imperceptible ones over the JND
    line, even though it moves the mean past it."""
    one_giant = [-0.60, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001]
    assert abs(sum(one_giant) / len(one_giant)) > JND["lpips"][0]
    result = assess_metric(pairs_from(one_giant), "lpips", "foreground", is_primary=True)
    assert result.clears_jnd is False
    assert result.verdict == "sub_jnd_significant"

    genuine = assess_metric(pairs_from([-0.09] * 7), "lpips", "foreground", is_primary=True)
    assert genuine.clears_jnd is True
    assert genuine.verdict == "perceptual_win"


def test_mixed_direction_never_produces_a_claim():
    result = assess_metric(pairs_from([-0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02]),
                           "lpips", "foreground", is_primary=True)
    assert result.verdict == "no_consistent_direction"
    assert result.significant is False


def test_large_effect_size_on_an_invisible_delta_is_still_not_a_win():
    """dz is unbounded by perceptibility -- BSRGAN's real dz=+6.55 sits on a
    0.36xJND delta. Pinning this stops dz being quoted as if it were a verdict."""
    result = assess_metric(pairs_from([-0.0100, -0.0101, -0.0102, -0.0103,
                                       -0.0104, -0.0105, -0.0106]),
                           "lpips", "foreground", is_primary=True)
    assert result.effect_size_dz is not None and abs(result.effect_size_dz) > 5
    assert result.verdict == "sub_jnd_significant"


def test_corroborating_metric_winning_alone_is_flagged():
    """Hard rule 3: FG claims come from lpips/dists_fg. A PSNR-only 'win' has
    to arrive carrying its own caveat."""
    result = assess_metric(pairs_from([1.0] * 7, metric="psnr"), "psnr", "foreground",
                           is_primary=False)
    assert result.verdict == "perceptual_win"
    assert any("not the primary metric" in w for w in result.warnings)


# --- exact tests --------------------------------------------------------------


def test_sign_test_matches_hand_computed_values():
    assert sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert sign_test_p(6, 0) == pytest.approx(2 / 64)
    assert sign_test_p(2, 0) == pytest.approx(0.5)
    assert sign_test_p(3, 3) == pytest.approx(1.0)
    assert sign_test_p(0, 0) == 1.0


def test_wilcoxon_floor_matches_the_sign_test_floor_at_n5():
    """Both bottom out at 0.0625, so 'use Wilcoxon instead' is not an escape
    from the n=5 problem -- a tempting and wrong fix."""
    assert wilcoxon_signed_rank_p([-1, -2, -3, -4, -5]) == pytest.approx(0.0625)
    assert wilcoxon_signed_rank_p([-1, -2, -3, -4, -5, -6]) == pytest.approx(0.03125)


def test_wilcoxon_handles_ties_and_zeros():
    assert wilcoxon_signed_rank_p([0, 0, 0]) == 1.0
    # Tied magnitudes share a midrank; symmetric input stays non-significant.
    assert wilcoxon_signed_rank_p([-2, 2, -2, 2]) == pytest.approx(1.0)


def test_bootstrap_ci_is_seeded_and_reproducible():
    deltas = [-0.01, -0.02, -0.015, -0.03, -0.025]
    assert bootstrap_ci(deltas) == bootstrap_ci(deltas)
    lo, hi = bootstrap_ci(deltas)
    assert lo < sum(deltas) / len(deltas) < hi


def test_cohens_dz_is_none_without_variance():
    assert cohens_dz([0.1, 0.1, 0.1]) is None
    assert cohens_dz([0.1]) is None


def test_holm_is_monotone_and_never_exceeds_one():
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.9])
    assert all(0 <= p <= 1 for p in adjusted)
    assert adjusted[0] <= adjusted[2] <= adjusted[1]  # by ascending raw p


# --- equivalence (the mirror claim) -------------------------------------------


def test_tiny_spread_supports_the_robustness_claim():
    """tab:ablation's alpha/beta grid: 0.0-0.05 dB against a 0.5 dB JND. This is
    the one audited claim the layer says is currently UNDER-stated."""
    result = equivalence_tost([0.0, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.02, 0.01], 0.5)
    assert result.equivalent is True
    assert result.verdict == "equivalent"


def test_a_wide_spread_at_small_n_is_not_declared_equivalent():
    """The failure mode equivalence exists to catch: a small point estimate
    whose CI still runs past the JND. 'Not shown different' is not 'shown
    equivalent', and only this test separates them."""
    result = equivalence_tost([0.05, -0.9, 0.8, -0.7, 0.6], 0.5)
    assert result.equivalent is False
    assert result.verdict == "not_shown_equivalent"


def test_equivalence_at_n2_is_flagged_as_not_credible():
    """Every landed restorer twin is n=2 and every one of them 'passes' TOST.
    Without this flag the audit would have read those as demonstrated ties."""
    result = equivalence_tost([-0.019, -0.010], 0.05)
    assert result.equivalent is True
    assert any("below the credible minimum" in w for w in result.warnings)


def test_one_pair_over_the_bound_blocks_equivalence():
    """The claim wording is per-pair ('all deltas are sub-JND'), so a tight mean
    around zero must not certify a suite containing a supra-JND pair."""
    result = equivalence_tost([0.6, -0.6, 0.01, -0.01, 0.0], 0.5)
    assert result.max_abs_delta == pytest.approx(0.6)
    assert result.equivalent is False


# --- citability and pairing hygiene -------------------------------------------


def run(hash_id, video, arm, lpips, *, invariant_failures=None):
    result = {
        "experiment_hash": hash_id,
        "config": {"video": video, "restorer": arm},
        "metrics": {"foreground": {"lpips_mean": lpips}},
    }
    if invariant_failures is not None:
        result["invariant_failures"] = invariant_failures
    return result


def test_a_run_with_invariant_failures_never_enters_a_suite():
    results = [
        run("aaa", "bear", "x", 0.20, invariant_failures=[]),
        run("bbb", "bear", "y", 0.18, invariant_failures=["goal1_fg_regression"]),
        run("ccc", "camel", "x", 0.30, invariant_failures=[]),
        run("ddd", "camel", "y", 0.28, invariant_failures=[]),
    ]
    pairs, skipped = collect_pairs(results, ["video"], "restorer", "x", "y",
                                   "foreground", "lpips")
    assert [p.pair_id for p in pairs] == [("camel",)]
    assert any("bbb" in s and "invariant_failures" in s for s in skipped)


def test_duplicate_arm_is_reported_rather_than_silently_picked():
    """Two runs on the same arm for one video means the pairing is ambiguous;
    quietly keeping the first would make the delta depend on scan order."""
    results = [
        run("aaa", "bear", "x", 0.20, invariant_failures=[]),
        run("aab", "bear", "x", 0.25, invariant_failures=[]),
        run("bbb", "bear", "y", 0.18, invariant_failures=[]),
    ]
    _, skipped = collect_pairs(results, ["video"], "restorer", "x", "y",
                               "foreground", "lpips")
    assert any("AMBIGUOUS PAIRING" in s and "aab" in s for s in skipped)


def test_unchecked_run_is_kept_but_warned_about():
    """Absent `invariant_failures` means never checked, which is not the same as
    failed -- dropping it would silently shrink n, the thing this module exists
    to keep visible."""
    results = [run("aaa", "bear", "x", 0.20), run("bbb", "bear", "y", 0.18)]
    pairs, skipped = collect_pairs(results, ["video"], "restorer", "x", "y",
                                   "foreground", "lpips")
    assert len(pairs) == 1
    assert any("never checked" in s for s in skipped)


def test_foreground_pairing_still_refuses_banned_keys():
    """The suite layer reads through compare._metric_value, so hard rule 3's
    union-bbox ban applies to it unchanged."""
    from presley.compare import REGION_METRIC_KEYS, _metric_value
    original = REGION_METRIC_KEYS["foreground"]["dists"]
    REGION_METRIC_KEYS["foreground"]["dists"] = "dists_mean"
    try:
        with pytest.raises(ValueError):
            _metric_value({"metrics": {"foreground": {"dists_mean": 0.1}}},
                          "foreground", "dists")
    finally:
        REGION_METRIC_KEYS["foreground"]["dists"] = original


# --- the existing single-comparison behaviour is untouched --------------------


@pytest.mark.parametrize("lpips_b,expected", [
    (0.20, "indistinguishable"),      # delta 0, within JND
    (0.24, "indistinguishable"),      # delta 0.04, within the 0.05 JND
    (0.28, "distinguishable"),        # delta 0.08, outside
])
def test_single_pair_verdicts_are_unchanged_by_the_suite_layer(lpips_b, expected):
    """The suite layer must be purely additive: one data point still behaves
    exactly as it did before, so no landed verdict silently flips."""
    a = {"experiment_hash": "a", "metrics": {"foreground": {"lpips_mean": 0.20}}}
    b = {"experiment_hash": "b", "metrics": {"foreground": {"lpips_mean": lpips_b}}}
    assert same_quality(a, b, region="foreground").verdict == expected


def test_two_pairs_cannot_produce_any_verdict_beyond_underpowered():
    """Every landed restorer twin comparison is n=2 (bear, camel). The floor
    there is 0.5, so none of them can be upgraded by this layer -- pinned so a
    future session cannot quietly make n=2 claimable."""
    assert min_attainable_sign_p(2) == pytest.approx(0.5)
    result = assess_metric(pairs_from([-0.001, -0.002]), "lpips", "foreground",
                           is_primary=True)
    assert result.verdict == "underpowered"
