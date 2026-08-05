"""The quality / bitrate / speed frontier: what a deployer actually chooses between.

Replaces the operating-map framing, which died when three waves agreed that
content does not select the transport (`docs/WAVE1_OPERATING_MAP.md`,
`docs/WAVE2A_CONTENT_AXIS.md`). What survived is better: **the three axes a
deployer cares about genuinely conflict, and we can quantify the frontier.**

The claim this tool exists to support and to try to break:

    Degrade-and-restore has no single best configuration. Buying quality costs
    bits, buying speed costs bits, and we can say how much.

Three axes per arm, all measured *within* an operating point:

  * **quality**  — restored BG-LPIPS, absolute. Lower is better. Never the
    restoration *gain*: blackout posts the corpus's largest gain and its worst
    absolute result, because it starts from the worst place.
  * **bitrate**  — Δbits vs the pristine baseline at the same fixed QP.
  * **speed**    — restoration fps.

⚠ **The methodological point this tool is built around: never pool across
cells without matching.** Each arm's naive median is taken over whatever
operating points it happened to be run at — 147 arms for
`downsample+realesrgan` against 8 for `blackout+e2fgvi`, at different QPs on
different videos. That is the same coverage confound that produced a withdrawn
motion correlation in Wave 2A and a physically impossible throughput ordering in
Wave 2C. `matched_summary` restricts every comparison to the cells where **all**
compared arms are present; `naive_summary` is computed too, purely so the gap
between them is visible rather than assumed small.

Hard rules enforced: fixed-QP only; `actual_bitrate_bps` for rate; BG-LPIPS is
the quality verdict and BG-PSNR never is; FG verdicts only from true masked
metrics; runs with `invariant_failures` are dropped.
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
from build_operating_map import (  # noqa: E402  (sibling script, not a package)
    BG_LPIPS_JND, FG_PSNR_JND, Arm, attach_residuals, eligible, load, score_arms,
)

# Two arms are "the same bitrate" when within this many percentage points.
# Not a perceptual threshold -- it is the smallest bitrate difference this
# corpus resolves reliably (compare.py tolerates 1% on its own integrity check).
BITS_TIE_PP = 1.0


# --------------------------------------------------------------------------
# Pareto
# --------------------------------------------------------------------------

def dominates(a: Arm, b: Arm) -> bool:
    """True when `a` is at least as good as `b` on all three axes and strictly
    better on one. Arms without timing are compared on two axes only -- absent
    data is not evidence of being dominated, which is the error that would
    quietly delete the cheap classical fills from the frontier."""
    axes_a = [(-a.bg_lpips), (-a.dbits_pct)]
    axes_b = [(-b.bg_lpips), (-b.dbits_pct)]
    if a.fps is not None and b.fps is not None:
        axes_a.append(a.fps)
        axes_b.append(b.fps)
    return all(x >= y for x, y in zip(axes_a, axes_b)) and any(
        x > y for x, y in zip(axes_a, axes_b))


def pareto_front(arms: Sequence[Arm]) -> List[Arm]:
    return [a for a in arms if not any(dominates(b, a) for b in arms if b is not a)]


# --------------------------------------------------------------------------
# The conflict measure -- the headline
# --------------------------------------------------------------------------

@dataclass
class ConflictCell:
    op: Tuple
    rate_winner: str
    rate_winner_dbits: float
    rate_winner_fps: Optional[float]
    faster_alt: Optional[str] = None
    faster_alt_dbits: Optional[float] = None
    faster_alt_fps: Optional[float] = None
    speedup: Optional[float] = None
    bits_given_up_pp: Optional[float] = None
    free: bool = False


def conflict_scan(by_op: Dict[Tuple, List[Arm]]) -> List[ConflictCell]:
    """How much does buying speed cost, for a deployer optimising bitrate?

    This exists because the first reading of the cost data claimed rate-first
    recommendations were "often replaceable for free". They are not. That
    analysis found arms that were faster at indistinguishable *quality* and
    never checked what happened to the *bitrate* -- the axis a rate-first
    deployer is optimising. Holding quality and ignoring bits is not "free"; it
    is measuring the wrong axis.

    For each cell: take the arm that saves the most bits, find every arm that is
    within a quality JND of it and meaningfully faster, and record what that
    speed costs in bitrate.
    """
    out: List[ConflictCell] = []
    for op, arms in sorted(by_op.items(), key=lambda kv: str(kv[0])):
        # Collapse duplicates first, under the bitrate objective, since that is
        # the objective this scan is about. Without it the scan reports "swaps"
        # from an arm to *itself* -- two configs of one label, e.g. blur_kernel
        # 7 vs 31 -- which is a within-arm tuning difference, not a choice a
        # deployer makes between arms. It also has to be the same rule the
        # summaries use, or the two halves of this report disagree silently.
        elig = [a for a in collapse_duplicates(eligible(arms), "bitrate").values()
                if a.fps is not None]
        if len(elig) < 2:
            continue
        rate_winner = min(elig, key=lambda a: a.dbits_pct)
        if rate_winner.fps is None:
            continue
        faster = [a for a in elig
                  if abs(a.bg_lpips - rate_winner.bg_lpips) < BG_LPIPS_JND
                  and a.fps > rate_winner.fps * 1.05]
        cell = ConflictCell(op=op, rate_winner=rate_winner.label,
                            rate_winner_dbits=rate_winner.dbits_pct,
                            rate_winner_fps=rate_winner.fps)
        if faster:
            best = max(faster, key=lambda a: a.fps)
            cell.faster_alt = best.label
            cell.faster_alt_dbits = best.dbits_pct
            cell.faster_alt_fps = best.fps
            cell.speedup = best.fps / rate_winner.fps
            cell.bits_given_up_pp = best.dbits_pct - rate_winner.dbits_pct
            # "Free" means it also holds the bitrate saving, not merely quality.
            cell.free = any(a.dbits_pct <= rate_winner.dbits_pct + BITS_TIE_PP
                            for a in faster)
        out.append(cell)
    return out


# --------------------------------------------------------------------------
# Summaries -- naive vs matched
# --------------------------------------------------------------------------

def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def collapse_duplicates(arms: Sequence[Arm], objective: str) -> Dict[str, Arm]:
    """One representative per arm label within a cell.

    **The aggregation rule, stated because leaving it implicit already changed
    published counts.** Wave 2B's sign tests read 9/9 or 8/9 depending purely on
    whether duplicates were collapsed best-of or first-of, and the rule was
    never written down. Several runs can share a (transport, fill) label inside
    one operating point while differing in a degradation parameter --
    `blur_kernel` 7/15/31, or block size.

    Rule: **best-of, under the objective being reported.** A deployer picking
    `blur+nafnet` would tune its kernel, not average over kernels, so best-of is
    the deployment-relevant summary. First-of is arbitrary; mean-of blends
    configurations a user chooses *between*. The objective is a parameter rather
    than hard-coded so that "best" never silently means "best on quality" while
    a bitrate column is being reported.
    """
    key = {"quality": lambda a: a.bg_lpips,
           "bitrate": lambda a: a.dbits_pct,
           "speed": lambda a: -(a.fps or float("-inf"))}[objective]
    best: Dict[str, Arm] = {}
    for a in arms:
        if a.label not in best or key(a) < key(best[a.label]):
            best[a.label] = a
    return best


def naive_summary(by_op: Dict[Tuple, List[Arm]], min_cells: int = 8) -> List[Dict[str, Any]]:
    """Per-arm medians pooled over every cell the arm appears in. NOT matched --
    reported only so the gap against `matched_summary` is visible."""
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"bg": [], "bits": [], "fps": [], "cells": []})
    for op, arms in by_op.items():
        for label, a in collapse_duplicates(arms, "quality").items():
            acc[label]["bg"].append(a.bg_lpips)
            acc[label]["bits"].append(a.dbits_pct)
            acc[label]["cells"].append(op)
            if a.fps is not None:
                acc[label]["fps"].append(a.fps)
    out = []
    for label, d in acc.items():
        if len(d["cells"]) < min_cells:
            continue
        out.append({"arm": label, "cells": len(d["cells"]),
                    "bg_lpips": _median(d["bg"]), "dbits_pct": _median(d["bits"]),
                    "fps": _median(d["fps"]) if d["fps"] else None})
    return sorted(out, key=lambda r: r["bg_lpips"])


def matched_summary(by_op: Dict[Tuple, List[Arm]], labels: Sequence[str],
                    objective: str = "quality") -> Optional[Dict[str, Any]]:
    """Compare `labels` only on the cells where **every one of them** appears.

    This is the whole point of the module. An unmatched comparison between an
    arm run at 147 operating points and one run at 8 is a statement about
    scheduling, not about the arms.
    """
    shared = []
    for op, arms in by_op.items():
        best = collapse_duplicates(arms, objective)
        if all(l in best for l in labels):
            shared.append((op, {l: best[l] for l in labels}))
    if not shared:
        return None
    per_arm = {}
    for label in labels:
        arms = [d[label] for _, d in shared]
        fps = [a.fps for a in arms if a.fps is not None]
        per_arm[label] = {
            "bg_lpips": _median([a.bg_lpips for a in arms]),
            "dbits_pct": _median([a.dbits_pct for a in arms]),
            "fps": _median(fps) if fps else None,
            "fps_n": len(fps),
        }
    return {"cells": len(shared), "videos": len({op[0] for op, _ in shared}),
            "arms": per_arm}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _fmt_op(op: Tuple) -> str:
    video, codec, qp, w, h = op
    return f"{video:<14} {codec:<7} QP{qp:<4} {w}x{h}"


def report(by_op: Dict[Tuple, List[Arm]], top_labels: Sequence[str],
           verbose: bool) -> Dict[str, Any]:
    print("=" * 78)
    print("CONFLICT -- what buying speed costs a deployer who is optimising bitrate")
    print("=" * 78)
    cells = conflict_scan(by_op)
    with_alt = [c for c in cells if c.faster_alt]
    free = [c for c in with_alt if c.free]
    costly = [c for c in with_alt if not c.free]
    print(f"  {len(cells)} cells have >=2 eligible arms with timing")
    print(f"  {len(with_alt)} of those offer a FASTER arm at indistinguishable quality")
    print(f"    genuinely free (also holds the bitrate saving) : {len(free)}")
    print(f"    costs bitrate                                  : {len(costly)}")
    if costly:
        give = [c.bits_given_up_pp for c in costly]
        print(f"  bitrate given up by the faster arm: median {_median(give):.1f} pp, "
              f"range {min(give):.1f}-{max(give):.1f} pp")
        print("\n  the expensive swaps, worst first:")
        print(f"    {'operating point':<38} {'rate winner':<24}{'->':^4}{'faster arm':<24}"
              f"{'speed':>7}{'bits lost':>11}")
        for c in sorted(costly, key=lambda c: -(c.bits_given_up_pp or 0))[:10]:
            print(f"    {_fmt_op(c.op):<38} {c.rate_winner:<24}{'->':^4}{c.faster_alt:<24}"
                  f"{c.speedup:>6.1f}x{c.bits_given_up_pp:>10.1f}pp")
    print("\n  Read this as the tradeoff being real: at equal quality, speed and")
    print("  bitrate trade against each other rather than coming together.\n")

    print("=" * 78)
    print("PER-ARM SUMMARY -- naive pooling vs matched pooling")
    print("=" * 78)
    naive = naive_summary(by_op)
    print("  NAIVE (each arm over its own cells -- confounded by coverage, do not cite):")
    print(f"    {'arm':<26}{'cells':>6}{'BG-LPIPS':>10}{'dbits':>9}{'fps':>8}")
    for r in naive:
        fps = f"{r['fps']:.2f}" if r["fps"] is not None else "--"
        print(f"    {r['arm']:<26}{r['cells']:>6}{r['bg_lpips']:>10.4f}"
              f"{r['dbits_pct']:>8.1f}%{fps:>8}")

    matched = matched_summary(by_op, top_labels)
    print(f"\n  MATCHED (only cells carrying all of: {', '.join(top_labels)}):")
    if matched is None:
        print("    no cell carries all of them -- the comparison does not exist")
    else:
        print(f"    {matched['cells']} cells over {matched['videos']} videos")
        print(f"    {'arm':<26}{'BG-LPIPS':>10}{'dbits':>9}{'fps':>8}")
        for label, d in matched["arms"].items():
            fps = f"{d['fps']:.2f}" if d["fps"] is not None else "--"
            print(f"    {label:<26}{d['bg_lpips']:>10.4f}{d['dbits_pct']:>8.1f}%{fps:>8}")
        print("\n    Compare against the naive rows above: any arm whose numbers move")
        print("    between the two tables was being flattered or punished by coverage.")

    print("\n" + "=" * 78)
    print("PARETO FRONTIER, per cell")
    print("=" * 78)
    on_front: Dict[str, int] = defaultdict(int)
    appears: Dict[str, int] = defaultdict(int)
    for op, arms in by_op.items():
        elig = eligible(arms)
        if len(elig) < 2:
            continue
        front = pareto_front(elig)
        for a in {a.label for a in elig}:
            appears[a] += 1
        for a in {a.label for a in front}:
            on_front[a] += 1
    print(f"  {'arm':<26}{'on frontier':>13}{'of cells':>10}{'share':>8}")
    for label in sorted(appears, key=lambda l: -(on_front[l] / appears[l])):
        if appears[label] < 4:
            continue
        print(f"  {label:<26}{on_front[label]:>13}{appears[label]:>10}"
              f"{100*on_front[label]/appears[label]:>7.0f}%")
    dominated = [l for l in appears if appears[l] >= 4 and on_front[l] == 0]
    if dominated:
        print(f"\n  never on any frontier (dominated wherever they appear): "
              f"{', '.join(sorted(dominated))}")
    print()
    return {
        "conflict": {"cells": len(cells), "with_faster_alt": len(with_alt),
                     "free": len(free), "costly": len(costly),
                     "bits_given_up_pp": [c.bits_given_up_pp for c in costly]},
        "naive": naive, "matched": matched,
        "frontier": {l: {"on_front": on_front[l], "appears": appears[l]}
                     for l in appears},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="results/presley.db")
    ap.add_argument("--arms", default="downsample+realesrgan,blackout+propainter,blur+nafnet",
                    help="comma-separated arm labels for the matched comparison")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    runs = load(args.db)
    by_op = score_arms(runs)
    attach_residuals(by_op)
    labels = [s.strip() for s in args.arms.split(",") if s.strip()]

    n_arms = sum(len(v) for v in by_op.values())
    print(f"{len(runs)} fixed-QP citable runs -> {n_arms} arms across {len(by_op)} "
          f"operating points on {len({op[0] for op in by_op})} videos\n")
    out = report(by_op, labels, args.verbose)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
