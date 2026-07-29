"""Suite-level significance layer, sitting *on top of* the JND gate in compare.py.

Why this exists: JND is the right rule for one comparison and the wrong rule for
a suite. Observed motivating case (docs/WAVE1_FALSIFIERS.md, F3): SVT-AV1
`tune=0` (VQ) vs the PSNR default, rate-matched, moved LPIPS in the same
direction on 5/5 videos (-0.0150 .. -0.0305) and DISTS on 5/5. Every delta is
far inside the 0.05 LPIPS JND, so hard rule 2 discards all of it -- yet a
consistent direction across independent videos is itself evidence, and the
framework had no way to say "small, but real and reproducible".

What this module does NOT do: it never overrides the JND verdict, and it never
converts a sub-JND effect into a perceptual claim. `compare.same_quality` is
untouched, so every existing single-pair verdict is bit-identical to before.
The strongest thing a sub-JND effect can earn here is the label
`sub_jnd_significant`, whose mandated wording (`WORDING`) is explicitly not a
perceptual win.

Three things the design has to get right, in order of how badly they bite:

1. **Small-N sign/rank tests have a p-value floor.** With n paired videos the
   smallest two-tailed p an exact sign test can return is 2/2**n. At n=5 that
   is **0.0625** -- so *no* 5/5 result can ever reach a two-tailed alpha of
   0.05, no matter how clean it looks. (The often-quoted 0.031 for 5/5 is the
   one-tailed value; one-tailed is not defensible here because the direction
   was chosen after seeing the data.) Exact Wilcoxon has the same 0.0625 floor
   at n=5. `min_attainable_p` is therefore reported on every result and an
   otherwise-consistent suite that cannot clear alpha by construction is
   verdicted `underpowered`, not `not_significant` -- the two mean very
   different things for what to do next (add a video vs drop the hypothesis).
2. **Metrics are not independent replicates.** LPIPS and DISTS agreeing 5/5 is
   close to one piece of evidence, not two: both are deep perceptual metrics on
   the same frames. So the multiple-comparisons family is over *candidates
   tried*, never over metrics, and one primary metric per goal (hard rule 3)
   carries the claim while the rest corroborate.
3. **The project explicitly shops for a winner.** The restorer matrix in
   docs/EXPERIMENTS_QUEUED.md is "try backbones until one beats Real-ESRGAN"
   -- six candidates so far. An uncorrected per-candidate alpha of 0.05 over
   six candidates is a ~26% chance of at least one spurious winner. Pass
   `family_size` (candidates tried against the same baseline for the same
   claim, *including the ones that lost and including this one*); Holm
   correction is applied and the count is reported alongside the verdict.

Everything is exact and dependency-light on purpose: binomial and Wilcoxon
null distributions are enumerated rather than approximated (n is single digits
here, where normal approximations are worst), and the bootstrap is seeded so a
number quoted in the paper reproduces.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from presley.compare import (
    JND,
    REGION_METRIC_KEYS,
    _config_value,
    _metric_value,
    scan_results,
)

# Two-tailed alpha for every suite test. Two-tailed is not negotiable: in every
# case this module is built for, the direction of the effect was read off the
# data before the test was run, so a one-tailed test would be assuming the
# answer. (This is exactly what turns F3's "5/5, p=0.031" into p=0.0625.)
ALPHA = 0.05

# Bootstrap settings. Seeded so a CI quoted in the paper is reproducible; a
# CI that moves between runs is not a number anyone can cite.
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 20260729

# Per-goal primary metric (hard rule 3: FG claims only from foreground.lpips_mean
# / dists_fg). The primary metric carries the claim; anything else in the same
# suite is corroborating and never enters the correction family.
PRIMARY_METRIC = {
    "foreground": "lpips",
    "background": "lpips",
    "overall": "lpips",
}

# The one wording each verdict is allowed to have in the paper or the log.
# `sub_jnd_significant` is the whole point of this module and the easiest thing
# to over-claim, so its sentence is fixed rather than left to a future session.
WORDING = {
    "perceptual_win": (
        "consistent across the suite, statistically significant, and above the "
        "perceptual threshold -- may be worded as an improvement"
    ),
    "sub_jnd_significant": (
        "consistent and statistically significant across the suite but BELOW the "
        "perceptual threshold -- report as a small, reproducible, imperceptible "
        "effect; never as a perceptual win, a quality improvement, or a 'better' "
        "result"
    ),
    "consistent_not_significant": (
        "direction is consistent but does not reach significance -- not evidence; "
        "report as an observation with n stated, or add pairs"
    ),
    "underpowered": (
        "direction is consistent but n is too small for ANY two-tailed test to "
        "reach alpha -- this is an unfinished experiment, not a negative result; "
        "add pairs before drawing either conclusion"
    ),
    "no_consistent_direction": (
        "direction is not consistent across the suite -- no claim"
    ),
    "insufficient_pairs": "fewer than 2 usable pairs -- fall back to the JND verdict",
}


# --------------------------------------------------------------------------
# Exact tests (enumerated, not approximated)
# --------------------------------------------------------------------------

def sign_test_p(n_positive: int, n_negative: int) -> float:
    """Exact two-tailed sign test. Zeros are dropped by the caller (standard),
    so n = n_positive + n_negative is the number of informative pairs."""
    n = n_positive + n_negative
    if n == 0:
        return 1.0
    k = min(n_positive, n_negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def min_attainable_sign_p(n: int) -> float:
    """Smallest two-tailed p an n-pair sign test can return (all pairs agreeing).

    Exposed because it is the difference between "we tested and found nothing"
    and "we could not have found anything": at n=5 this is 0.0625, above the
    0.05 alpha, so a 5/5 suite is underpowered by construction. n>=6 is the
    first size that can clear alpha (2/2**6 = 0.03125)."""
    if n <= 0:
        return 1.0
    return min(1.0, 2.0 / (2 ** n))


def wilcoxon_signed_rank_p(deltas: Sequence[float]) -> Optional[float]:
    """Exact two-tailed Wilcoxon signed-rank p by enumerating all 2**n sign
    assignments of the observed |deltas|.

    Exact rather than normal-approximated because n is single digits here,
    which is precisely where the approximation is worst. Returns None above
    n=20 (2**20 enumerations) -- larger suites than this project will ever
    have, and the caller falls back to the sign test.

    Ties in |delta| get midranks, matching the standard treatment. Zeros are
    dropped, as in the sign test.
    """
    nonzero = [d for d in deltas if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    if n > 20:
        return None

    ranks = _midranks([abs(d) for d in nonzero])
    observed = sum(r for d, r in zip(nonzero, ranks) if d > 0)
    total = sum(ranks)
    # Two-tailed: how extreme is the observed W+ relative to total/2?
    deviation = abs(observed - total / 2.0)

    count = 0
    for mask in range(2 ** n):
        w_plus = sum(ranks[i] for i in range(n) if (mask >> i) & 1)
        if abs(w_plus - total / 2.0) >= deviation - 1e-12:
            count += 1
    return min(1.0, count / (2 ** n))


def _midranks(values: Sequence[float]) -> List[float]:
    """Ranks of `values` (1-based), ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def bootstrap_ci(deltas: Sequence[float], confidence: float = 0.95,
                 resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> Tuple[Optional[float], Optional[float]]:
    """Seeded percentile bootstrap CI on the mean paired delta.

    Percentile (not BCa): with n<=10 the bias-correction term is itself
    estimated from the same handful of points and buys nothing real. The CI is
    reported as a magnitude statement, and it is NOT a substitute for the sign
    test -- at tiny n the bootstrap resamples the same few values and its
    coverage is optimistic. It is here to answer "how big", after the sign
    test has answered "is it there".
    """
    values = list(deltas)
    n = len(values)
    if n < 2:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_i = int((1.0 - confidence) / 2.0 * resamples)
    hi_i = min(resamples - 1, int((1.0 + confidence) / 2.0 * resamples))
    return means[lo_i], means[hi_i]


def cohens_dz(deltas: Sequence[float]) -> Optional[float]:
    """Paired effect size: mean(delta) / sd(delta), sample sd.

    Reported as a unitless companion to `jnd_ratio`, never on its own: a huge
    dz on a delta 20x below JND still describes something nobody can see. That
    is the whole trap this module exists to keep the project out of."""
    n = len(deltas)
    if n < 2:
        return None
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    # Not `var <= 0`: identical floats leave a ~1e-34 residue, which divides
    # into a dz of ~1e15 and reads as an overwhelming effect. Scale the
    # zero-variance test to the data, so genuinely constant deltas return None
    # instead of the largest effect size in the report.
    scale = max(abs(d) for d in deltas) or 1.0
    if math.sqrt(var) <= 1e-12 * scale:
        return None
    return mean / math.sqrt(var)


@dataclass
class EquivalenceResult:
    n: int
    bound: float                  # the JND, used as the equivalence margin
    mean_delta: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    equivalent: bool              # CI wholly inside +/- bound
    max_abs_delta: Optional[float]
    verdict: str
    note: str
    warnings: List[str] = field(default_factory=list)


# Below this many pairs the percentile bootstrap resamples so few distinct
# values that its interval is essentially the observed range: it cannot
# represent the sampling variability it is supposed to, and will happily
# certify equivalence from two points. Equivalence claims below this n are
# emitted with the finding, never as a bare verdict.
MIN_CREDIBLE_EQUIVALENCE_N = 4


def equivalence_tost(deltas: Sequence[float], bound: float,
                     confidence: float = 0.90) -> EquivalenceResult:
    """Bootstrap TOST: is the effect demonstrably SMALLER than the JND?

    This is the mirror of `assess_metric`, and the audit needed it. "All deltas
    are within JND, therefore the parameter does not matter" (the alpha/beta
    robustness claim, `CLAIM(tab:ablation)`) is an *equivalence* claim, and a
    difference test can never support one -- failing to reject the null is not
    evidence for it. Two very different situations both currently get worded as
    "within JND":

      - deltas of 0.03 dB against a 0.5 dB JND, tight CI -> genuinely
        equivalent; the robustness claim is real and is currently UNDER-stated
      - deltas of 0.4 dB against a 0.5 dB JND, CI running to 0.9 -> nothing was
        shown either way, and "no impact" is an over-statement

    Standard TOST uses a 90% CI for a 5% one-sided test on each bound; that
    convention is kept. Equivalence is declared only when the whole CI sits
    inside +/-bound, so a wide CI fails even when the point estimate is tiny --
    which is the honest outcome for a small n.
    """
    n = len(deltas)
    if n < 2:
        return EquivalenceResult(n, bound, None, None, None, False, None,
                                 "insufficient_pairs",
                                 "fewer than 2 pairs -- equivalence cannot be tested")
    mean = sum(deltas) / n
    lo, hi = bootstrap_ci(deltas, confidence=confidence)
    max_abs = max(abs(d) for d in deltas)
    # Both conditions, because the claim being supported is worded per-pair
    # ("all deltas are sub-JND"), not about the mean: a suite averaging to
    # nothing while one pair sits above the JND has not shown that.
    equivalent = lo is not None and -bound < lo and hi < bound and max_abs < bound
    if equivalent:
        verdict = "equivalent"
        note = (f"the {int(confidence*100)}% CI [{lo:+.4f}, {hi:+.4f}] lies wholly inside "
                f"+/-{bound}, so the effect is demonstrably below the perceptual "
                "threshold -- 'no perceptible impact' is supported, not merely unrefuted")
    else:
        verdict = "not_shown_equivalent"
        note = (f"the {int(confidence*100)}% CI [{lo:+.4f}, {hi:+.4f}] (max |delta| "
                f"{max_abs:.4f}) is not contained in "
                f"+/-{bound}: the data are consistent with an effect at or above the "
                "perceptual threshold. Do NOT word this as 'no impact' or "
                "'robust' -- it is 'not measured precisely enough to tell'")
    warnings: List[str] = []
    if equivalent and n < MIN_CREDIBLE_EQUIVALENCE_N:
        warnings.append(
            f"n={n} is below the credible minimum ({MIN_CREDIBLE_EQUIVALENCE_N}) for an "
            "equivalence claim: a bootstrap over this few points reproduces the observed "
            "range rather than the sampling distribution, so it will certify almost any "
            "tight pair. Treat this as 'consistent with equivalence', not as evidence of it"
        )
    return EquivalenceResult(n, bound, mean, lo, hi, equivalent, max_abs, verdict, note,
                             warnings)


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values, returned in input order.

    Holm rather than plain Bonferroni: same family-wise error control, strictly
    more power, no extra assumptions. Use this when the whole candidate family
    is tested at once (all restorers vs the same baseline). `SuiteMetricResult`
    also carries a single-result Bonferroni value for the common case where
    only one candidate is being written up but k were tried."""
    indexed = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    k = len(pvalues)
    out = [0.0] * k
    running = 0.0
    for rank, i in enumerate(indexed):
        adjusted = min(1.0, (k - rank) * pvalues[i])
        running = max(running, adjusted)  # enforce monotonicity
        out[i] = running
    return out


# --------------------------------------------------------------------------
# Pair collection
# --------------------------------------------------------------------------

@dataclass
class PairedDelta:
    pair_id: Tuple            # e.g. ("camel",) -- the value(s) of pair_by
    hash_a: str
    hash_b: str
    value_a: float
    value_b: float
    delta: float              # b - a, signed, same convention as compare.py
    within_jnd: Optional[bool]


@dataclass
class SuiteMetricResult:
    metric: str
    region: str
    key: Optional[str]
    is_primary: bool
    pairs: List[PairedDelta] = field(default_factory=list)
    n: int = 0
    n_positive: int = 0
    n_negative: int = 0
    n_zero: int = 0
    direction: Optional[str] = None       # "better" / "worse" / None (b vs a)
    consistent: bool = False
    mean_delta: Optional[float] = None
    median_delta: Optional[float] = None
    sign_p: Optional[float] = None
    wilcoxon_p: Optional[float] = None
    min_attainable_p: Optional[float] = None
    underpowered: bool = False
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    ci_excludes_zero: Optional[bool] = None
    effect_size_dz: Optional[float] = None
    jnd: Optional[float] = None
    jnd_ratio: Optional[float] = None     # |mean_delta| / jnd
    pairs_clearing_jnd: int = 0
    clears_jnd: bool = False
    family_size: int = 1
    sign_p_corrected: Optional[float] = None
    significant: bool = False
    verdict: str = "insufficient_pairs"
    wording: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class SuiteComparison:
    arm_a: str
    arm_b: str
    region: str
    pair_by: List[str]
    family_size: int
    metrics: List[SuiteMetricResult] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def primary(self) -> Optional[SuiteMetricResult]:
        for m in self.metrics:
            if m.is_primary:
                return m
        return None


def _citable(result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """A run with a non-empty `invariant_failures` is never citable (hard rule).
    Enforced here rather than left to the reader, because a suite silently
    including one poisoned pair is exactly how a bad number reaches the paper.

    A run with the key *absent* has never been checked; that is a different
    problem from a run that failed, so it is warned about but not dropped --
    dropping it would silently shrink n, which this module is built to make
    visible rather than hide."""
    failures = result.get("invariant_failures")
    if failures:
        return False, f"invariant_failures: {failures}"
    if "invariant_failures" not in result:
        return True, "no invariant_failures field (never checked by presley-invariants)"
    return True, None


def collect_pairs(results: List[Dict[str, Any]], pair_by: List[str],
                  arm_key: str, arm_a: Any, arm_b: Any, region: str,
                  metric: str) -> Tuple[List[PairedDelta], List[str]]:
    """Build one paired delta per `pair_by` value (typically the video).

    Pairing is what makes the sign test legitimate: each pair is the same
    content under two arms, so the between-video variance -- which dwarfs every
    effect this project measures -- cancels. An unpaired comparison across
    videos would be dominated by content difficulty and is not supported here.
    """
    jnd_value = JND[metric][0] if metric in JND else None
    skipped: List[str] = []
    duplicates: List[str] = []
    buckets: Dict[Tuple, Dict[str, Dict[str, Any]]] = {}

    for result in results:
        config = result.get("config", {})
        arm = _config_value(config, arm_key)
        if arm == arm_a:
            side = "a"
        elif arm == arm_b:
            side = "b"
        else:
            continue
        ok, note = _citable(result)
        hash_id = result.get("experiment_hash", "?")
        if not ok:
            skipped.append(f"{hash_id}: {note}")
            continue
        if note:
            skipped.append(f"{hash_id}: WARNING {note} (kept)")
        key = tuple(_config_value(config, k) for k in pair_by)
        if any(v is None for v in key):
            skipped.append(f"{hash_id}: missing pair key(s) {pair_by}")
            continue
        bucket = buckets.setdefault(key, {})
        if side in bucket:
            duplicates.append(
                f"{hash_id} vs {bucket[side].get('experiment_hash', '?')} "
                f"on arm {side!r} for pair {key}"
            )
            continue
        bucket[side] = result

    if duplicates:
        # Loud but bounded. This means pair_by does not identify a run, so the
        # kept member of each collision is scan-order-dependent -- the reader
        # has to fix the query, and does not need all 27 collisions to know it.
        shown = "; ".join(duplicates[:3])
        more = f" (+{len(duplicates) - 3} more)" if len(duplicates) > 3 else ""
        skipped.append(
            f"AMBIGUOUS PAIRING: {len(duplicates)} run(s) collided on an arm that "
            f"--pair-by does not distinguish, and were dropped: {shown}{more}. "
            "Add the config keys that differ (degradation, codec, width/height) to "
            "--pair-by; until then the surviving pairs are scan-order-dependent."
        )

    pairs: List[PairedDelta] = []
    unpaired = 0
    for key in sorted(buckets, key=lambda k: tuple(str(v) for v in k)):
        bucket = buckets[key]
        if "a" not in bucket or "b" not in bucket:
            unpaired += 1
            continue
        _, value_a = _metric_value(bucket["a"], region, metric)
        _, value_b = _metric_value(bucket["b"], region, metric)
        if value_a is None or value_b is None:
            skipped.append(f"pair {key}: {region}.{metric} unavailable on one arm")
            continue
        delta = value_b - value_a
        pairs.append(PairedDelta(
            pair_id=key,
            hash_a=bucket["a"].get("experiment_hash", "?"),
            hash_b=bucket["b"].get("experiment_hash", "?"),
            value_a=value_a, value_b=value_b, delta=delta,
            within_jnd=(abs(delta) < jnd_value) if jnd_value else None,
        ))
    if unpaired:
        # One line, not one per config: with a wide pair_by most configs have
        # only the baseline arm, and 25 lines of that buries the real skips
        # (invariant failures, missing metrics) that a reader must not miss.
        skipped.append(
            f"{unpaired} config(s) had only one arm and were dropped (unpaired runs "
            "carry no paired delta); widen or narrow --pair-by if that count surprises you"
        )
    return pairs, skipped


# --------------------------------------------------------------------------
# The combined decision rule
# --------------------------------------------------------------------------

def assess_metric(pairs: List[PairedDelta], metric: str, region: str,
                  family_size: int = 1, alpha: float = ALPHA,
                  is_primary: bool = False) -> SuiteMetricResult:
    """JND magnitude AND suite significance, combined into one verdict.

    The rule, in the order it is applied:

      n < 2                              -> insufficient_pairs (JND governs)
      direction not unanimous            -> no_consistent_direction
      min_attainable_p > alpha           -> underpowered
      corrected p > alpha                -> consistent_not_significant
      corrected p <= alpha, sub-JND      -> sub_jnd_significant
      corrected p <= alpha, clears JND   -> perceptual_win

    "Clears JND" deliberately requires BOTH |mean delta| >= JND and a majority
    of individual pairs clearing it, so one large video cannot carry a suite of
    imperceptible ones over the line.

    Direction is required to be unanimous rather than merely majority: with the
    n<=10 suites this project runs, a 4/6 split has a two-tailed sign p of
    0.69 and would fail the test anyway, so unanimity costs nothing and makes
    the reported `direction` field honest.
    """
    jnd_value = JND[metric][0] if metric in JND else None
    key = REGION_METRIC_KEYS.get(region, {}).get(metric)
    lower_is_better = JND[metric][1] if metric in JND else None
    out = SuiteMetricResult(metric=metric, region=region, key=key,
                            is_primary=is_primary, pairs=pairs,
                            jnd=jnd_value, family_size=max(1, family_size))
    out.n = len(pairs)
    deltas = [p.delta for p in pairs]
    out.n_positive = sum(1 for d in deltas if d > 0)
    out.n_negative = sum(1 for d in deltas if d < 0)
    out.n_zero = sum(1 for d in deltas if d == 0)

    if out.n < 2:
        out.verdict = "insufficient_pairs"
        out.wording = WORDING[out.verdict]
        return out

    out.mean_delta = sum(deltas) / out.n
    out.median_delta = _median(deltas)
    out.effect_size_dz = cohens_dz(deltas)
    out.ci_low, out.ci_high = bootstrap_ci(deltas)
    if out.ci_low is not None and out.ci_high is not None:
        out.ci_excludes_zero = (out.ci_low > 0) or (out.ci_high < 0)

    informative = out.n_positive + out.n_negative
    out.sign_p = sign_test_p(out.n_positive, out.n_negative)
    out.wilcoxon_p = wilcoxon_signed_rank_p(deltas)
    out.min_attainable_p = min_attainable_sign_p(informative)
    out.sign_p_corrected = min(1.0, out.sign_p * out.family_size)

    if jnd_value:
        out.jnd_ratio = abs(out.mean_delta) / jnd_value
        out.pairs_clearing_jnd = sum(1 for p in pairs if p.within_jnd is False)
        out.clears_jnd = (
            abs(out.mean_delta) >= jnd_value
            and out.pairs_clearing_jnd * 2 >= out.n
        )

    if out.n_zero:
        out.warnings.append(
            f"{out.n_zero} pair(s) had delta exactly 0 and were dropped from the "
            "sign/rank tests; an exact 0 usually means the two arms produced "
            "identical pixels, which is a wiring question before it is a statistics one"
        )

    unanimous = informative > 0 and (out.n_positive == 0 or out.n_negative == 0)
    out.consistent = unanimous
    if unanimous and lower_is_better is not None:
        improved = (out.n_negative == informative) if lower_is_better else (out.n_positive == informative)
        out.direction = "better" if improved else "worse"

    # min_attainable_p is compared against the CORRECTED bar: with k candidates
    # tried, the n needed to clear alpha grows too, and hiding that would let a
    # suite look merely "not significant" when it was never able to be.
    corrected_floor = min(1.0, out.min_attainable_p * out.family_size)
    out.underpowered = corrected_floor > alpha
    out.significant = out.sign_p_corrected is not None and out.sign_p_corrected <= alpha

    if not unanimous:
        out.verdict = "no_consistent_direction"
    elif out.underpowered:
        out.verdict = "underpowered"
        out.warnings.append(
            f"n={informative} informative pair(s), family_size={out.family_size}: the "
            f"smallest attainable two-tailed p is {corrected_floor:.4f} > alpha={alpha}. "
            f"No result at this n can be significant. Need n>="
            f"{_n_needed(alpha, out.family_size)} pairs."
        )
    elif not out.significant:
        out.verdict = "consistent_not_significant"
    elif out.clears_jnd:
        out.verdict = "perceptual_win"
    else:
        out.verdict = "sub_jnd_significant"

    if out.verdict == "sub_jnd_significant" and out.ci_excludes_zero is False:
        out.warnings.append(
            "sign test is significant but the bootstrap CI on the mean delta "
            "includes 0 -- the direction is consistent, the magnitude is not "
            "pinned down; state the CI, do not state the mean alone"
        )
    if out.verdict == "perceptual_win" and not is_primary:
        out.warnings.append(
            f"{metric} is not the primary metric for region={region} "
            f"({PRIMARY_METRIC.get(region)}); corroborating metrics do not carry claims "
            "(hard rule 3)"
        )
    out.wording = WORDING[out.verdict]
    return out


def _n_needed(alpha: float = ALPHA, family_size: int = 1) -> int:
    """Smallest number of unanimous pairs whose corrected two-tailed sign-test p
    can reach alpha. Reported so an underpowered suite comes with its remedy."""
    n = 1
    while n <= 64:
        if min(1.0, min_attainable_sign_p(n) * max(1, family_size)) <= alpha:
            return n
        n += 1
    return n


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def compare_suite(results_dir: str = "results", pair_by: Optional[List[str]] = None,
                  arm_key: str = "config.restorer", arm_a: Any = None, arm_b: Any = None,
                  region: str = "foreground", metrics: Optional[List[str]] = None,
                  family_size: int = 1, alpha: float = ALPHA,
                  results: Optional[List[Dict[str, Any]]] = None) -> SuiteComparison:
    """Scan results/, pair arm_a against arm_b by `pair_by`, assess each metric."""
    if not pair_by:
        raise ValueError("pair_by must be a non-empty list of dotted config keys (e.g. ['video'])")
    if arm_a is None or arm_b is None:
        raise ValueError("both arm_a and arm_b are required")
    if results is None:
        results = scan_results(results_dir)

    available = [m for m in REGION_METRIC_KEYS.get(region, {}) if m in JND]
    if metrics is None:
        metrics = [m for m in available if JND[m][2]]  # gating metrics only
    primary = PRIMARY_METRIC.get(region)

    out = SuiteComparison(arm_a=str(arm_a), arm_b=str(arm_b), region=region,
                          pair_by=pair_by, family_size=max(1, family_size))
    seen_skips = set()
    for metric in metrics:
        pairs, skipped = collect_pairs(results, pair_by, arm_key, arm_a, arm_b, region, metric)
        for s in skipped:
            if s not in seen_skips:
                seen_skips.add(s)
                out.skipped.append(s)
        out.metrics.append(assess_metric(
            pairs, metric, region, family_size=family_size, alpha=alpha,
            is_primary=(metric == primary),
        ))

    if out.family_size == 1:
        out.warnings.append(
            "family_size=1: no multiple-comparisons correction applied. If more than one "
            "candidate was tried against this baseline for this claim (see the restorer "
            "matrix in docs/EXPERIMENTS_QUEUED.md), pass --candidates-tried with the "
            "total INCLUDING the ones that lost."
        )
    return out


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def format_suite(suite: SuiteComparison) -> str:
    lines = [
        f"=== suite: {suite.arm_a} vs {suite.arm_b}  (region={suite.region}, "
        f"paired by {'+'.join(suite.pair_by)}, candidates tried={suite.family_size}) ===",
        "",
    ]
    for m in suite.metrics:
        star = " *primary*" if m.is_primary else ""
        lines.append(f"-- {m.metric} ({m.key}){star}")
        if m.n == 0:
            lines.append("   no usable pairs")
            lines.append("")
            continue
        for p in m.pairs:
            label = ",".join(str(v) for v in p.pair_id)
            flag = "" if p.within_jnd is None else ("  (within JND)" if p.within_jnd else "  (CLEARS JND)")
            lines.append(f"   {label:<20} {p.value_a:>9.4f} -> {p.value_b:>9.4f}   d={p.delta:>+9.4f}{flag}")
        lines.append(
            f"   n={m.n}  +{m.n_positive}/-{m.n_negative}"
            + (f"/0x{m.n_zero}" if m.n_zero else "")
            + f"  direction={m.direction or 'mixed'}"
        )
        if m.mean_delta is not None:
            ci = (f"[{m.ci_low:+.4f}, {m.ci_high:+.4f}]"
                  if m.ci_low is not None else "--")
            jnd_ratio = f"{m.jnd_ratio:.2f}xJND" if m.jnd_ratio is not None else "--"
            dz = f"{m.effect_size_dz:+.2f}" if m.effect_size_dz is not None else "--"
            lines.append(f"   mean d={m.mean_delta:+.4f} ({jnd_ratio})  95% CI {ci}  dz={dz}")
        if m.sign_p is not None:
            wil = f"{m.wilcoxon_p:.4f}" if m.wilcoxon_p is not None else "--"
            lines.append(
                f"   sign p={m.sign_p:.4f}  wilcoxon p={wil}  "
                f"corrected p={m.sign_p_corrected:.4f}  "
                f"(floor at this n: {m.min_attainable_p:.4f})"
            )
        lines.append(f"   VERDICT: {m.verdict}")
        lines.append(f"     -> {m.wording}")
        for w in m.warnings:
            lines.append(f"     warning: {w}")
        lines.append("")
    for s in suite.skipped:
        lines.append(f"skipped: {s}")
    for w in suite.warnings:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
