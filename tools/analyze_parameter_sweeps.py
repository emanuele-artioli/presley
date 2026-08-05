"""W4b: the parameter sweeps that were already on disk and invisible to us.

Discovered while fixing the control-matching bug. The analysis keyed arms on
`(component, transport, block_size, shrink_amount)`, which does not name a
single degradation-strength parameter -- so `dancing`@QP43's three `blur` runs
at `blur_kernel` 7/15/31 were one identity, and a sweep that had already been
paid for in GPU time read as a duplicate. **No re-runs are needed. The runs were
executed correctly; the defect was in the analysis.**

This tool finds every config key that varies *inside* one (operating point, arm)
group and reports what varying it did to the three axes. That is the strongest
form the comparison can take with data already on disk: both ends of the sweep
share a video, a codec, a QP, a resolution, an arm and every other config value,
so the only thing that differs is the swept parameter.

**It is descriptive and says so.** Sweeps here cover 3-8 groups, far below hard
rule 2b's n>=8 videos, so no p-value is computed and no row is citable as a
result on its own. What it produces is a direction and a magnitude per sweep,
and a count of how consistent that direction is -- enough to say which
parameters are worth a designed experiment and which are visibly inert.

⚠ Already checked and negative before this tool existed: `inpainter_params` does
**not** explain ProPainter's 10-40x timing split. 158 of 161 ProPainter runs at
640x360 carry an empty params dict and still split 77 slow / 81 fast. W3's
timing campaign is still required; do not read a speed column here as closing it.

Hard rules inherited from `build_operating_map`: fixed-QP only, BG-LPIPS is the
quality verdict, `actual_bitrate_bps` for rate, runs with `invariant_failures`
dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_operating_map import Arm, load, score_arms  # noqa: E402

# Keys that identify *which arm this is* rather than how it was tuned. A group
# is formed within one arm, so these can never be the swept parameter, and
# listing them as sweeps would report a comparison between different arms.
ARM_IDENTITY_KEYS = frozenset({
    "component", "degradation", "removal_mode", "restorer", "inpainter",
    "video", "codec", "codec_params", "width", "height", "target_bitrate",
})
# Below this many groups a "direction" is one or two observations.
MIN_GROUPS_TO_REPORT = 2

# `None` is a legitimate parameter *value* here -- it means the key is absent
# from the config, which for `downsample_uniform_level` is the graded mode and
# for `mask_source` is the default source. A sentinel is therefore required to
# mean "no winner could be determined", because using None for both silently
# deleted every group the absent-key variant won: the graded arm wins quality
# 8 of 8 and the first version of this tool reported 1.
NO_BEST = object()


@dataclass
class SweepGroup:
    """One (operating point, arm) where exactly one config key varies."""
    op: Tuple
    label: str
    key: str
    points: List[Tuple[Any, Arm]]

    @property
    def video(self) -> str:
        return self.op[0]

    def spread(self, axis: str) -> Optional[float]:
        values = [v for v in (_axis(a, axis) for _, a in self.points) if v is not None]
        return (max(values) - min(values)) if len(values) >= 2 else None

    def best(self, axis: str) -> Any:
        """The parameter value that optimises `axis`, or NO_BEST if fewer than
        two of this group's runs carry that axis at all."""
        scored = [(v, a) for v, a in self.points if _axis(a, axis) is not None]
        if len(scored) < 2:
            return NO_BEST
        return min(scored, key=lambda va: _axis(va[1], axis))[0]


def _value_order(point: Tuple[Any, Arm]) -> Tuple[int, float, str]:
    """Numeric parameter values sort numerically, not lexically -- `blur_kernel`
    7/15/31 read as a ladder, and as 15/31/7 under a plain string sort."""
    value = point[0]
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


def _axis(arm: Arm, axis: str) -> Optional[float]:
    """Lower is better on every axis, so `min` is always the winner."""
    if axis == "quality":
        return arm.bg_lpips
    if axis == "bitrate":
        return arm.dbits_pct
    return -arm.fps if arm.fps is not None else None


def find_sweeps(by_op: Dict[Tuple, List[Arm]]) -> Dict[str, List[SweepGroup]]:
    """Config keys that vary within one (operating point, arm) group.

    A group is only a sweep of `key` when `key` is the **only** thing that
    differs. Two runs differing in both `blur_kernel` and `block_size` are not
    a blur_kernel sweep, and reporting them as one attributes a block-size
    effect to the kernel -- the same shape of error as the control-matching bug
    that revealed these sweeps in the first place.
    """
    grouped: Dict[Tuple, List[Arm]] = defaultdict(list)
    for op, arms in by_op.items():
        for arm in arms:
            grouped[(op, arm.label)].append(arm)

    sweeps: Dict[str, List[SweepGroup]] = defaultdict(list)
    for (op, label), arms in grouped.items():
        if len(arms) < 2:
            continue
        configs = [dict(a.config) for a in arms]
        keys = {k for cfg in configs for k in cfg} - ARM_IDENTITY_KEYS
        varying = {k for k in keys
                   if len({cfg.get(k) for cfg in configs}) > 1}
        for key in varying:
            # Restrict to the sub-group where everything *else* is constant.
            buckets: Dict[Tuple, List[Tuple[Any, Arm]]] = defaultdict(list)
            for cfg, arm in zip(configs, arms):
                rest = tuple(sorted((k, v) for k, v in cfg.items() if k != key))
                buckets[rest].append((cfg.get(key), arm))
            for points in buckets.values():
                distinct = {v for v, _ in points}
                if len(distinct) < 2:
                    continue
                sweeps[key].append(SweepGroup(op=op, label=label, key=key,
                                              points=sorted(points, key=_value_order)))
    return dict(sweeps)


def summarise(key: str, groups: Sequence[SweepGroup]) -> Dict[str, Any]:
    """Direction and magnitude per axis, with the group and video counts that
    say how far the number can be trusted."""
    out: Dict[str, Any] = {
        "key": key,
        "n_groups": len(groups),
        "n_videos": len({g.video for g in groups}),
        "values_seen": sorted({str(v) for g in groups for v, _ in g.points}),
        "axes": {},
    }
    for axis in ("quality", "bitrate", "speed"):
        spreads = [s for s in (g.spread(axis) for g in groups) if s is not None]
        winners: Dict[str, int] = defaultdict(int)
        for g in groups:
            best = g.best(axis)
            if best is not NO_BEST:
                winners[str(best)] += 1
        if not spreads:
            continue
        spreads.sort()
        out["axes"][axis] = {
            "n": len(spreads),
            "median_spread": spreads[len(spreads) // 2],
            "max_spread": spreads[-1],
            "best_value_counts": dict(sorted(winners.items(), key=lambda kv: -kv[1])),
            # A parameter whose best value is the same one nearly everywhere is
            # a tuning default waiting to be adopted; one whose best value moves
            # per cell is a knob, and the paper cannot quote a single setting.
            "modal_share": (max(winners.values()) / sum(winners.values()))
                           if winners else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="results/presley.db")
    ap.add_argument("--key", default=None, help="print every group for one key")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sweeps = find_sweeps(score_arms(load(args.db)))
    summaries = [summarise(k, g) for k, g in sorted(sweeps.items())
                 if len(g) >= MIN_GROUPS_TO_REPORT]
    summaries.sort(key=lambda s: -s["n_groups"])

    print("Parameter sweeps already on disk -- DESCRIPTIVE ONLY, n is far below")
    print("the n>=8 videos hard rule 2b requires for any significance claim.\n")
    print(f"{'key':30} {'groups':>6} {'videos':>7}  "
          f"{'quality spread':>14} {'bits spread pp':>15} {'best-value modal':>17}")
    print("-" * 96)
    for s in summaries:
        q = s["axes"].get("quality", {})
        b = s["axes"].get("bitrate", {})
        modal = q.get("modal_share")
        print(f"{s['key']:30} {s['n_groups']:>6} {s['n_videos']:>7}  "
              f"{q.get('median_spread', float('nan')):>14.4f} "
              f"{b.get('median_spread', float('nan')):>15.2f} "
              f"{(f'{modal:.2f}' if modal is not None else 'n/a'):>17}")

    if args.key:
        print(f"\nEvery group for {args.key}:")
        for g in sorted(sweeps.get(args.key, []), key=lambda g: str(g.op)):
            print(f"  {g.video} {g.op[1]} QP{g.op[2]} {g.op[3]}x{g.op[4]} / {g.label}")
            for value, arm in g.points:
                fps = f"{arm.fps:7.2f}" if arm.fps is not None else "    n/a"
                print(f"      {str(value):>28}  bg_lpips {arm.bg_lpips:.4f}  "
                      f"dbits {arm.dbits_pct:+7.2f}%  fps {fps}  {arm.hash}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summaries, fh, indent=1, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
