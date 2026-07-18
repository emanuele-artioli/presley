"""JND-based comparison of finished PRESLEY experiments (results/<hash>/result.json).

Read-only against results/. Computes no metrics itself -- it only reads what
evaluation.py already wrote, and decides whether two experiments are at
perceptually indistinguishable quality so their bitrate can be compared.

Why this exists: at fixed QP, degradation methods legitimately win on bits
while losing a fraction of a dB on FG quality. Reporting that FG loss as a
"result" without checking whether it's below the just-noticeable-difference
misrepresents the finding -- see CLAUDE.md's "never dress up imperceptible
deltas" rule. This module is the single source of truth for the JND
thresholds so that rule doesn't have to be re-derived by hand each session.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# metric -> (JND threshold, lower_is_better, gates_verdict)
# Thresholds are literature just-noticeable-differences. fid never gates: it is
# corroborating-only even in "overall" (whole-frame FID at N~60-90 vs D=2048
# needs evaluation.py's --fid-validity gate before it's citable at all), and for
# "foreground" the only available key (fid_fg_bbox) is a bbox crop, not a true
# region metric -- see REGION_METRIC_KEYS below.
JND: Dict[str, Tuple[float, bool, bool]] = {
    "psnr":  (0.5,  False, True),
    "ssim":  (0.05, False, True),
    "lpips": (0.05, True,  True),
    "dists": (0.05, True,  True),
    "vmaf":  (6.0,  False, True),
    "fid":   (1.5,  True,  False),
}
# FVMD has no established JND anywhere in the literature. Reported for context
# under every region; never gates the verdict.
FVMD_METRIC = "fvmd"

# Per-region key overrides. This is what enforces the FG-citability rules from
# .claude/skills/results-report/SKILL.md in code instead of prose: vmaf/fvmd/old
# dists_mean/fid are union-bbox artifacts (100% of frame on some videos) and
# must never be read as a "foreground" quality signal, so they're simply absent
# from the foreground row here rather than merely undocumented.
REGION_METRIC_KEYS: Dict[str, Dict[str, str]] = {
    "foreground": {
        "psnr": "psnr_mean", "ssim": "ssim_mean", "lpips": "lpips_mean",
        "dists": "dists_fg", "fid": "fid_fg_bbox",
    },
    "background": {
        "psnr": "psnr_mean", "ssim": "ssim_mean", "lpips": "lpips_mean",
        "dists": "dists_bg",
    },
    "overall": {
        "psnr": "psnr_mean", "ssim": "ssim_mean", "lpips": "lpips_mean",
        "dists": "dists_mean", "vmaf": "vmaf_mean", "fid": "fid",
    },
}
# Keys that are known union-bbox artifacts under "foreground" specifically.
# _metric_value refuses to read these for region="foreground" even if a caller
# hand-edits REGION_METRIC_KEYS, since this is the one invariant that must hold.
BANNED_FG_KEYS = {"vmaf_mean", "vmaf_neg_mean", "fvmd", "dists_mean", "fid"}


@dataclass
class MetricComparison:
    metric: str
    key: Optional[str]
    available: bool
    value_a: Optional[float]
    value_b: Optional[float]
    delta: Optional[float]           # b - a, signed
    jnd: Optional[float]
    within_jnd: Optional[bool]
    jnd_margin: Optional[float]      # |delta| / jnd; None if not available
    gates_verdict: bool


@dataclass
class ComparisonResult:
    hash_a: str
    hash_b: str
    region: str
    metrics: List[MetricComparison] = field(default_factory=list)
    verdict: str = "insufficient_data"
    binding_metric: Optional[str] = None
    perceptual_backing: bool = False
    bitrate_a_bps: Optional[float] = None
    bitrate_b_bps: Optional[float] = None
    bitrate_delta_pct: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class GroupComparison:
    group_key: Tuple
    group_by: List[str]
    members: List[str]
    baseline_hash: Optional[str]
    comparisons: List[ComparisonResult]
    quality_indistinguishable: bool
    bitrate_winner: Optional[str]
    bitrate_saving_pct: Optional[float]
    note: Optional[str] = None


def load_result(hash_or_path: str, results_dir: str = "results") -> Dict[str, Any]:
    """Load results/<hash>/result.json. Accepts a bare hash, a unique hash
    prefix, or a full path to a result.json."""
    if os.path.isfile(hash_or_path):
        path = hash_or_path
    else:
        direct = os.path.join(results_dir, hash_or_path, "result.json")
        if os.path.isfile(direct):
            path = direct
        else:
            matches = [d for d in os.listdir(results_dir) if d.startswith(hash_or_path)]
            if not matches:
                raise FileNotFoundError(f"No results/<hash>/ matching prefix {hash_or_path!r} under {results_dir}")
            if len(matches) > 1:
                raise ValueError(f"Ambiguous hash prefix {hash_or_path!r}: matches {matches}")
            path = os.path.join(results_dir, matches[0], "result.json")
    with open(path) as f:
        return json.load(f)


def bitrate_bps(result: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    """actual_bitrate_bps plus an integrity check against transmitted_size_bytes.

    Never reads file_size_bytes for bitrate: for elvis/presley_ai it's the
    lossless FFV1 *restored* output (tens of MB), a decode-side artifact
    unrelated to what was actually transmitted. actual_bitrate_bps is already
    computed from transmitted bytes (video + side-channel maps) by every
    component (see elvis.py/presley_ai.py/roi.py/baselines.py), so it's the
    uniform comparator -- this just double-checks that invariant held.
    """
    warnings: List[str] = []
    actual = result.get("actual_bitrate_bps")
    transmitted = result.get("transmitted_size_bytes")
    frames = result.get("video_frames")
    fps = result.get("video_framerate")
    if actual is not None and transmitted and frames and fps:
        duration = frames / fps
        expected = (transmitted * 8) / duration if duration else None
        if expected and actual and abs(actual - expected) / expected > 0.01:
            warnings.append(
                f"actual_bitrate_bps ({actual:.0f}) disagrees with transmitted_size_bytes-derived "
                f"bitrate ({expected:.0f}) by >1%"
            )
    return actual, warnings


def _metric_value(result: Dict[str, Any], region: str, metric: str) -> Tuple[Optional[str], Optional[float]]:
    """Look up REGION_METRIC_KEYS[region][metric] in result['metrics'][region].
    Returns (key_used, value); (None, None) if unavailable. Never raises on
    missing data -- raises only if asked to read a banned key for foreground."""
    key = REGION_METRIC_KEYS.get(region, {}).get(metric)
    if key is None:
        return None, None
    if region == "foreground" and key in BANNED_FG_KEYS:
        raise ValueError(f"{key!r} is a banned union-bbox key for region='foreground'; not a true FG metric")
    value = result.get("metrics", {}).get(region, {}).get(key)
    return key, value


def compare_metric(exp_a: Dict[str, Any], exp_b: Dict[str, Any], region: str, metric: str) -> MetricComparison:
    jnd, lower_is_better, gates = JND[metric]
    key, value_a = _metric_value(exp_a, region, metric)
    _, value_b = _metric_value(exp_b, region, metric)
    if key is None or value_a is None or value_b is None:
        return MetricComparison(metric, key, False, value_a, value_b, None, jnd, None, None, gates)
    delta = value_b - value_a
    within = abs(delta) < jnd
    margin = abs(delta) / jnd if jnd else None
    return MetricComparison(metric, key, True, value_a, value_b, delta, jnd, within, margin, gates)


def _fvmd_comparison(exp_a: Dict[str, Any], exp_b: Dict[str, Any], region: str) -> MetricComparison:
    """FVMD has no established JND; always non-gating, reported for context only."""
    key = FVMD_METRIC  # same key name in every region block
    value_a = exp_a.get("metrics", {}).get(region, {}).get(key)
    value_b = exp_b.get("metrics", {}).get(region, {}).get(key)
    if value_a is None or value_b is None:
        return MetricComparison("fvmd", key, False, value_a, value_b, None, None, None, None, False)
    delta = value_b - value_a
    return MetricComparison("fvmd", key, True, value_a, value_b, delta, None, None, None, False)


def same_quality(exp_a: Dict[str, Any], exp_b: Dict[str, Any], region: str = "foreground") -> ComparisonResult:
    """Pairwise JND comparison of two loaded result.json dicts for one region.

    verdict:
      indistinguishable            - every available gating metric within JND,
                                      and at least one perceptual metric
                                      (lpips/dists/ssim) backed the call.
      indistinguishable_psnr_only  - within JND but PSNR was the only gating
                                      metric available (e.g. a fast_only entry).
                                      Not a license to claim perceptual parity.
      distinguishable               - >=1 gating metric outside its JND.
      insufficient_data            - no gating metric available at all.
    """
    hash_a = exp_a.get("experiment_hash", "?")
    hash_b = exp_b.get("experiment_hash", "?")
    metrics: List[MetricComparison] = []
    for metric in JND:
        metrics.append(compare_metric(exp_a, exp_b, region, metric))
    metrics.append(_fvmd_comparison(exp_a, exp_b, region))

    gating = [m for m in metrics if m.gates_verdict and m.available]
    result = ComparisonResult(hash_a=hash_a, hash_b=hash_b, region=region, metrics=metrics)

    if not gating:
        result.verdict = "insufficient_data"
    else:
        outside = [m for m in gating if not m.within_jnd]
        if outside:
            worst = max(outside, key=lambda m: m.jnd_margin or 0)
            result.verdict = "distinguishable"
            result.binding_metric = f"{region}.{worst.key} (Δ={worst.delta:+.4f}, {worst.jnd_margin:.2f}×JND)"
        else:
            perceptual_backing = any(m.metric in ("lpips", "dists", "ssim") for m in gating)
            result.perceptual_backing = perceptual_backing
            if perceptual_backing:
                result.verdict = "indistinguishable"
            else:
                result.verdict = "indistinguishable_psnr_only"
                result.warnings.append(
                    "Only PSNR was available to gate this verdict -- PSNR alone is not evidence of "
                    "perceptual equivalence (see CLAUDE.md's mean_fill finding: highest PSNR can be the "
                    "perceptually worst result). Run full evaluation (not --fast-metrics) before trusting this."
                )

    ba, warn_a = bitrate_bps(exp_a)
    bb, warn_b = bitrate_bps(exp_b)
    result.bitrate_a_bps = ba
    result.bitrate_b_bps = bb
    if ba and bb:
        result.bitrate_delta_pct = 100.0 * (bb - ba) / ba
    result.warnings.extend(warn_a)
    result.warnings.extend(warn_b)
    return result


def same_quality_by_hash(hash_a: str, hash_b: str, results_dir: str = "results",
                          region: str = "foreground") -> ComparisonResult:
    exp_a = load_result(hash_a, results_dir)
    exp_b = load_result(hash_b, results_dir)
    return same_quality(exp_a, exp_b, region)


# --------------------------------------------------------------------------
# Group scan
# --------------------------------------------------------------------------

def scan_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    """Load every results/<hash>/result.json with a metrics block. Isolates
    bad/unreadable entries (warns, skips) rather than crashing the whole scan,
    matching evaluate_all's per-experiment isolation."""
    out = []
    if not os.path.isdir(results_dir):
        return out
    for entry in sorted(os.listdir(results_dir)):
        if entry.startswith("_"):
            continue
        path = os.path.join(results_dir, entry, "result.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"presley-compare: skipping {path}: {e}")
            continue
        if "metrics" not in data:
            continue
        out.append(data)
    return out


def _config_value(config: Dict[str, Any], dotted_key: str) -> Any:
    """Dotted-path lookup, e.g. 'codec_params.qp' -> config['codec_params']['qp'].
    Returns None (not KeyError) on any missing segment."""
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def group_experiments(results: List[Dict[str, Any]], group_by: List[str]) -> Dict[Tuple, List[Dict[str, Any]]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for result in results:
        config = result.get("config", {})
        key = tuple(_config_value(config, k) for k in group_by)
        if any(v is None for v in key):
            continue  # this experiment doesn't have all the grouping keys; not comparable
        groups.setdefault(key, []).append(result)
    return groups


def pick_baseline(group: List[Dict[str, Any]], baseline_component: Optional[str]) -> Optional[Dict[str, Any]]:
    if baseline_component is None:
        return None
    for result in group:
        if result.get("config", {}).get("component") == baseline_component:
            return result
    return None


def compare_groups(results_dir: str = "results", group_by: Optional[List[str]] = None,
                    region: str = "foreground",
                    baseline_component: Optional[str] = None) -> List[GroupComparison]:
    """Scan + group + compare. With baseline_component set, compares baseline
    vs every other group member (star topology -- matches the driving use
    case of "does the bridge tie the baseline at matched QP, and who's
    cheaper"). Without it, all-pairs within each group.

    bitrate_winner is only ever populated among members whose quality vs the
    baseline (or, in all-pairs mode, ALL group members) is indistinguishable
    -- a group that fails the quality gate never gets a silent bitrate
    comparison across a real quality gap.
    """
    if not group_by:
        raise ValueError("group_by must be a non-empty list of dotted config keys")
    results = scan_results(results_dir)
    groups = group_experiments(results, group_by)

    out: List[GroupComparison] = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        member_hashes = [m.get("experiment_hash", "?") for m in members]

        if baseline_component is not None:
            baseline = pick_baseline(members, baseline_component)
            if baseline is None:
                out.append(GroupComparison(
                    group_key=key, group_by=group_by, members=member_hashes,
                    baseline_hash=None, comparisons=[], quality_indistinguishable=False,
                    bitrate_winner=None, bitrate_saving_pct=None,
                    note=f"no member with component={baseline_component!r} in this group",
                ))
                continue
            others = [m for m in members if m is not baseline]
            comparisons = [same_quality(baseline, m, region=region) for m in others]
            baseline_hash = baseline.get("experiment_hash", "?")
        else:
            comparisons = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    comparisons.append(same_quality(members[i], members[j], region=region))
            baseline_hash = None

        ok_verdicts = {"indistinguishable", "indistinguishable_psnr_only"}
        quality_ok = all(c.verdict in ok_verdicts for c in comparisons) and len(comparisons) > 0

        bitrate_winner = None
        bitrate_saving_pct = None
        note = None
        if quality_ok:
            candidates = [baseline] + others if baseline_component is not None else members
            with_bitrate = [(m, bitrate_bps(m)[0]) for m in candidates]
            with_bitrate = [(m, b) for m, b in with_bitrate if b]
            if with_bitrate:
                winner, winner_bps = min(with_bitrate, key=lambda pair: pair[1])
                bitrate_winner = winner.get("experiment_hash", "?")
                if baseline_component is not None and winner is not baseline:
                    base_bps = bitrate_bps(baseline)[0]
                    if base_bps:
                        bitrate_saving_pct = 100.0 * (winner_bps - base_bps) / base_bps
        else:
            distinguishable = [c for c in comparisons if c.verdict == "distinguishable"]
            if distinguishable:

                def _worst_margin(c: ComparisonResult) -> float:
                    outside = [m for m in c.metrics if m.gates_verdict and m.available and not m.within_jnd]
                    return max((m.jnd_margin or 0) for m in outside) if outside else 0

                worst = max(distinguishable, key=_worst_margin)
                note = f"N/A — quality differs (binding: {worst.binding_metric})"
            else:
                note = "N/A — insufficient data to establish a quality verdict"

        out.append(GroupComparison(
            group_key=key, group_by=group_by, members=member_hashes,
            baseline_hash=baseline_hash, comparisons=comparisons,
            quality_indistinguishable=quality_ok, bitrate_winner=bitrate_winner,
            bitrate_saving_pct=bitrate_saving_pct, note=note,
        ))
    return out


# --------------------------------------------------------------------------
# Formatting / CLI
# --------------------------------------------------------------------------

def format_pairwise(result: ComparisonResult) -> str:
    lines = [f"=== {result.hash_a} vs {result.hash_b}  (region={result.region}) ==="]
    lines.append(f"{'metric':<8} {'key':<16} {'A':>10} {'B':>10} {'delta':>10} {'JND':>6} {'within':>7} {'margin':>7}")
    for m in result.metrics:
        if not m.available:
            lines.append(f"{m.metric:<8} {str(m.key):<16} {'--':>10} {'--':>10} {'--':>10} {'--':>6} {'--':>7} {'--':>7}")
            continue
        jnd_str = f"{m.jnd:.3f}" if m.jnd is not None else "--"
        within_str = "" if m.within_jnd is None else ("yes" if m.within_jnd else "NO")
        margin_str = f"{m.jnd_margin:.2f}x" if m.jnd_margin is not None else "--"
        gate_mark = "" if m.gates_verdict else " (context only)"
        lines.append(
            f"{m.metric:<8} {m.key:<16} {m.value_a:>10.4f} {m.value_b:>10.4f} "
            f"{m.delta:>+10.4f} {jnd_str:>6} {within_str:>7} {margin_str:>7}{gate_mark}"
        )
    lines.append("")
    if result.bitrate_a_bps and result.bitrate_b_bps:
        lines.append(
            f"bitrate: A={result.bitrate_a_bps:.0f} bps  B={result.bitrate_b_bps:.0f} bps  "
            f"delta={result.bitrate_delta_pct:+.1f}%"
        )
    lines.append(f"verdict: {result.verdict}" + (f"  (binding: {result.binding_metric})" if result.binding_metric else ""))
    for w in result.warnings:
        lines.append(f"warning: {w}")
    return "\n".join(lines)


def format_groups(groups: List[GroupComparison]) -> str:
    lines = []
    for g in groups:
        header = ", ".join(f"{k}={v}" for k, v in zip(g.group_by, g.group_key))
        lines.append(f"=== {header} ===")
        lines.append(f"  members: {', '.join(g.members)}")
        if g.baseline_hash:
            lines.append(f"  baseline: {g.baseline_hash}")
        lines.append(f"  quality_indistinguishable: {g.quality_indistinguishable}")
        if g.quality_indistinguishable:
            saving = f" ({g.bitrate_saving_pct:+.1f}% vs baseline)" if g.bitrate_saving_pct is not None else ""
            lines.append(f"  bitrate winner: {g.bitrate_winner}{saving}")
        else:
            lines.append(f"  bitrate winner: {g.note}")
        lines.append("")
    return "\n".join(lines)


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=str, default="results", nargs="?")
    parser.add_argument("--region", choices=["foreground", "background", "overall"], default="foreground")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text tables")

    pairwise = parser.add_argument_group("pairwise mode")
    pairwise.add_argument("--hash-a", type=str, default=None)
    pairwise.add_argument("--hash-b", type=str, default=None)

    group = parser.add_argument_group("group-scan mode")
    group.add_argument("--group-by", type=str, default=None,
                        help="Comma-separated dotted config keys, e.g. video,width,height,codec_params.qp")
    group.add_argument("--baseline-component", type=str, default=None,
                        help="config.component value to compare every other group member against (star topology). "
                             "Without this, all-pairs comparison within each group.")

    args = parser.parse_args()

    pairwise_mode = args.hash_a is not None or args.hash_b is not None
    group_mode = args.group_by is not None
    if pairwise_mode and group_mode:
        parser.error("--hash-a/--hash-b and --group-by are mutually exclusive")
    if not pairwise_mode and not group_mode:
        parser.error("specify either --hash-a/--hash-b or --group-by")
    if pairwise_mode and (args.hash_a is None or args.hash_b is None):
        parser.error("pairwise mode requires both --hash-a and --hash-b")

    if pairwise_mode:
        result = same_quality_by_hash(args.hash_a, args.hash_b, args.results_dir, region=args.region)
        print(json.dumps(_to_jsonable(result), indent=2) if args.json else format_pairwise(result))
    else:
        group_by = [k.strip() for k in args.group_by.split(",") if k.strip()]
        groups = compare_groups(args.results_dir, group_by=group_by, region=args.region,
                                 baseline_component=args.baseline_component)
        print(json.dumps(_to_jsonable(groups), indent=2) if args.json else format_groups(groups))


if __name__ == "__main__":
    main()
