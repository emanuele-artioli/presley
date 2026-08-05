"""W4: which quality/bitrate orderings in this corpus are significance-backed.

This analysis produced `docs/W4_SIGNIFICANCE.md` on 2026-08-03 and was **not
committed** -- the numbers were in the report and the code that made them was
not, so nobody could re-run it after adding videos. That is the gap this file
closes; it reproduces the published table on the same data before being used to
extend it.

Method, and every choice in it is one the corpus has already been burned by:

  * **Baseline arm is `downsample+realesrgan`**, the quality claim's incumbent.
    Every other arm is a candidate against it.
  * **Pairing is within an operating point**, never across. Each arm's coverage
    differs by an order of magnitude, and pooling unmatched cells is what
    produced a withdrawn motion result and an impossible throughput ordering.
  * **The unit is the video** (hard rule 2b), so a video with five cells does
    not outvote one with a single cell. Per-video value = mean of its per-cell
    deltas.
  * **Both arms must be deployable in that cell** -- `build_operating_map.
    eligible`: saves bits against the pristine baseline and costs no visible
    foreground quality. Applied to the baseline arm *and* the candidate, since a
    comparison between two arms nobody would deploy is not evidence about a
    deployment choice. This is the filter the published table referred to
    obliquely as videos "blocked by an ineligible baseline".
  * **Duplicate arms inside a cell collapse best-of under the objective being
    reported**, the rule pinned in `build_tradeoff_surface.collapse_duplicates`.
    Reused rather than restated so the two tools cannot drift apart.

  * **Two-tailed exact sign test**, then **Holm within a family of
    `--family-size` candidates per objective** (default 14: every arm ever
    compared against this baseline, losers included). Untested candidates enter
    Holm at p=1.0, which is what makes the smallest p multiply by the full 14.
  * **n>=8 videos** (hard rule 2b's restorer floor) or the row is reported as
    underpowered, whatever its p.

**Order matters between eligibility and collapsing, and only one order
reproduces the published table.** Filtering for eligibility *before* collapsing
lets a cell contribute through whichever of its duplicate runs is deployable;
collapsing first can hand the cell to a run that then fails the filter, and the
cell is lost. Filter-then-collapse reproduces all eight published rows exactly
(`blur+nafnet` 11/11 p_Holm 0.0137, `blackout+propainter` 10/10 p_Holm 0.0254,
`freeze+propainter` n=6, `ac_truncate+nafnet` n=7); collapse-then-filter drops
`freeze+propainter`/quality to n=5. Do not reorder them.

Quality is BG-LPIPS, absolute and lower-better; BG-PSNR is never the verdict.
Bitrate is Δbits vs the matched pristine baseline. A positive delta in either
column means the *baseline arm* won that video.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from build_operating_map import DROPPED, Arm, eligible, load, score_arms  # noqa: E402
from build_tradeoff_surface import collapse_duplicates  # noqa: E402
from presley.suite import holm_adjust, sign_test_p  # noqa: E402

BASE_LABEL = "downsample+realesrgan"
# Hard rule 2b: a restorer comparison needs this many videos before its p means
# anything, however small that p is.
N_FLOOR = 8
OBJECTIVES = ("quality", "bitrate")


def _value(arm: Arm, objective: str) -> float:
    """Lower is better on both axes, so a positive candidate-minus-baseline
    delta always means the baseline won."""
    return arm.bg_lpips if objective == "quality" else arm.dbits_pct


def paired_deltas(by_op: Dict[Tuple, List[Arm]], candidate: str,
                  objective: str) -> Dict[str, List[float]]:
    """video -> per-cell (candidate - baseline) deltas, matched within cells."""
    per_video: Dict[str, List[float]] = defaultdict(list)
    for op, arms in by_op.items():
        best = collapse_duplicates(eligible(arms), objective)
        if BASE_LABEL not in best or candidate not in best:
            continue
        per_video[op[0]].append(
            _value(best[candidate], objective) - _value(best[BASE_LABEL], objective))
    return dict(per_video)


class Row:
    def __init__(self, candidate: str, objective: str,
                 per_video: Dict[str, List[float]]):
        self.candidate = candidate
        self.objective = objective
        self.videos = sorted(per_video)
        self.means = [sum(v) / len(v) for _, v in sorted(per_video.items())]
        self.n = len(self.means)
        # Exact zeros cannot happen with float metrics, but a tie would be
        # dropped rather than counted for either side.
        self.base_wins = sum(1 for d in self.means if d > 0)
        self.cand_wins = sum(1 for d in self.means if d < 0)
        self.p_raw = sign_test_p(self.base_wins, self.cand_wins) if self.n else 1.0
        self.p_holm: Optional[float] = None

    @property
    def verdict(self) -> str:
        if self.n < N_FLOOR:
            return f"underpowered (n<{N_FLOOR})"
        if self.p_holm is not None and self.p_holm < 0.05:
            return "SIGNIFICANT"
        return "n.s."


def holm_within_family(rows: Sequence[Row], family_size: int) -> None:
    """Holm over `family_size` candidates, padding the untested ones with p=1.0.

    Padding is the point: the family is every arm ever compared against this
    baseline, not merely the ones that survived to be tabulated. Adjusting over
    only the reported rows would silently shrink the correction each time a
    losing arm was dropped from the write-up.
    """
    pad = max(0, family_size - len(rows))
    adjusted = holm_adjust([r.p_raw for r in rows] + [1.0] * pad)
    for row, p in zip(rows, adjusted):
        row.p_holm = p


def format_drops() -> str:
    """Every run that could not be scored, by reason.

    Printed unconditionally, not on a flag. Twice now a wave of clean runs has
    landed and `n` has not moved, because a fresh run carries no region LPIPS
    until `presley-evaluate --backfill-lpips` touches it and the arm simply
    never appears. A silent exclusion is indistinguishable from the runs having
    failed, and the second occurrence is what turned this from an anecdote into
    output.
    """
    if not DROPPED:
        return "no runs dropped"
    lines = ["runs excluded before scoring, by reason:"]
    for reason, hashes in sorted(DROPPED.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {len(hashes):4d}  {reason}")
    return "\n".join(lines)


def analyse(db: str, family_size: int) -> List[Row]:
    by_op = score_arms(load(db))
    candidates = sorted({a.label for arms in by_op.values() for a in arms}
                        - {BASE_LABEL})
    rows: List[Row] = []
    for objective in OBJECTIVES:
        family = [Row(c, objective, paired_deltas(by_op, c, objective))
                  for c in candidates]
        family = [r for r in family if r.n > 0]
        holm_within_family(family, family_size)
        rows.extend(family)
    return rows


def format_rows(rows: Sequence[Row], min_n: int) -> str:
    lines = [f"baseline arm: {BASE_LABEL}    (a positive delta = baseline wins)",
             "",
             f"{'candidate':<28} {'objective':<9} {'n':>3} {'base wins':>10} "
             f"{'p raw':>8} {'p_Holm':>8}  verdict",
             "-" * 88]
    shown = [r for r in rows if r.n >= min_n]
    for r in sorted(shown, key=lambda r: (r.p_holm if r.p_holm is not None else 1.0)):
        lines.append(
            f"{r.candidate:<28} {r.objective:<9} {r.n:>3} "
            f"{r.base_wins:>4}/{r.n:<5} {r.p_raw:>8.4f} {r.p_holm:>8.4f}  {r.verdict}")
    hidden = len(rows) - len(shown)
    if hidden:
        lines.append("")
        lines.append(f"({hidden} rows with n < {min_n} not shown; -f 1 to see them)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="results/presley.db")
    ap.add_argument("--family-size", type=int, default=14,
                    help="candidates ever compared against the baseline, losers included")
    ap.add_argument("-f", "--min-n", type=int, default=5,
                    help="hide rows below this many paired videos")
    ap.add_argument("--candidate", default=None,
                    help="print the per-video detail for one arm")
    args = ap.parse_args()

    rows = analyse(args.db, args.family_size)
    print(format_rows(rows, args.min_n))
    print()
    print(format_drops())

    if args.candidate:
        print()
        for r in rows:
            if r.candidate != args.candidate:
                continue
            print(f"{r.candidate} / {r.objective}: per-video means")
            for video, mean in zip(r.videos, r.means):
                print(f"  {video:<24} {mean:+.5f}  "
                      f"{'baseline wins' if mean > 0 else 'candidate wins'}")


if __name__ == "__main__":
    main()
