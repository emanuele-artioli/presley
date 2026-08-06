"""Score methods against the three goals on three axes — the (B) deliverable.

Three sub-tables, one per goal (select / degrade / restore), sharing one cell
format. Rows are choices a deployer could actually make, plus a **null row**
(what happens if you skip the goal) and an **oracle row** where one exists.

The point of the null row is that several results here are only interpretable
against it: "the score captures 0.833 of the oracle's bits" means nothing until
you know random selection captures 0.402. The null therefore goes IN the table,
never in a footnote.

## The four marks, and why they are a type rather than a convention

    number + verdict   measured; the verdict says whether it won
    EQUIVALENT (=)     TOST-equivalent to the null within +-JND
    NOT_APPLICABLE     zero by construction; carries a mechanism reason
    NO_DATA (--)       no measurement exists; carries a reason

`EQUIVALENT` may only come from `suite.equivalence_tost`, never from failing a
difference test -- "we found no difference" and "we showed there is no
difference" are different claims and this project has conflated them before.

`NOT_APPLICABLE` and `NO_DATA` **require** a reason string, enforced in
`__post_init__`. That is deliberate: absence has doubled as the error signal
repeatedly in this project (a runner exiting 0 with every experiment failed; a
"foreground" metric that was silently the whole frame; a backfill that reported
success having done nothing). A blank cell must say why it is blank or the
table cannot be trusted.

Reuses `build_operating_map.{load,score_arms,eligible,build_controls}` and
`build_tradeoff_surface.collapse_duplicates` so the scorecard cannot drift from
the tables already in the paper.
"""
from __future__ import annotations

import argparse
import enum
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from build_operating_map import (  # noqa: E402
    BG_LPIPS_JND, FG_PSNR_JND, eligible, load, score_arms,
)
from build_tradeoff_surface import collapse_duplicates  # noqa: E402

from presley.suite import min_attainable_sign_p, sign_test_p  # noqa: E402


class Mark(enum.Enum):
    MEASURED = "measured"
    EQUIVALENT = "equivalent"        # TOST only
    NOT_APPLICABLE = "n/a"           # zero by construction
    NO_DATA = "no-data"


NEEDS_REASON = {Mark.NOT_APPLICABLE, Mark.NO_DATA}


@dataclass
class Cell:
    """One method x one axis. Reasons are mandatory for the two absent marks."""
    mark: Mark
    value: Optional[float] = None
    unit: str = ""
    n: Optional[int] = None
    favouring: Optional[int] = None
    p_holm: Optional[float] = None
    verdict: str = ""
    reason: str = ""

    def __post_init__(self):
        if self.mark in NEEDS_REASON and not self.reason.strip():
            raise ValueError(
                f"{self.mark.value} cell must carry a reason -- a blank cell that "
                f"does not say why it is blank cannot be distinguished from a bug"
            )
        if self.mark is Mark.MEASURED and self.value is None:
            raise ValueError("measured cell must carry a value")

    def render(self) -> str:
        if self.mark is Mark.NO_DATA:
            return "--"
        if self.mark is Mark.NOT_APPLICABLE:
            return "n/a"
        if self.mark is Mark.EQUIVALENT:
            return "= (TOST)"
        s = f"{self.value:+.3f}{self.unit}" if abs(self.value) < 10 else f"{self.value:+.1f}{self.unit}"
        if self.n is not None:
            s += f" [n={self.n}"
            if self.favouring is not None:
                s += f", {self.favouring}/{self.n}"
            if self.p_holm is not None:
                s += f", p={self.p_holm:.3f}"
            if self.verdict:
                s += f", {self.verdict}"
            s += "]"
        return s


@dataclass
class Row:
    method: str
    kind: str                        # "null" | "method" | "oracle"
    cells: Dict[str, Cell] = field(default_factory=dict)


AXES = ("computation", "bitrate", "quality")


def verdict_for(n: int, favouring: int, p_holm: float, jnd_cleared: bool) -> str:
    """Verdict ladder, mirroring suite.assess_metric's vocabulary.

    `underpowered` is distinct from `not_significant`: below the floor no result
    could ever have reached alpha, so "no effect" is not what was shown.
    """
    if n < 2:
        return "insufficient"
    if favouring not in (0, n):
        if p_holm > 0.05:
            return "not_significant"
    if min_attainable_sign_p(n) > 0.05:
        return "underpowered"
    if p_holm > 0.05:
        return "not_significant"
    return "perceptual_win" if jnd_cleared else "sub_jnd_significant"


def goal1_rows() -> List[Row]:
    """Selection: selector-vs-selector at a FIXED transport.

    Scored against the leave-one-superblock-out bit oracle, which is what makes
    the bitrate axis fillable at all. The random null is the interpretive
    anchor and is a row, not a footnote.
    """
    rows = [
        Row("random-k (null)", "null", {
            "computation": Cell(Mark.NOT_APPLICABLE,
                                reason="no scoring pass is run; selection is free by construction"),
            "bitrate": Cell(Mark.MEASURED, value=0.402, unit="", n=8,
                            verdict="random-selection null"),
            "quality": Cell(Mark.MEASURED, value=0.0, unit="", n=8,
                            verdict="reference"),
        }),
        Row("EVCA alpha/beta score", "method", {
            "computation": Cell(Mark.NO_DATA,
                                reason="selection_time_seconds is not recorded; EVCA scoring, "
                                       "selection, degradation and encode are folded into one "
                                       "encoding_time_seconds (see W-E2)"),
            "bitrate": Cell(Mark.MEASURED, value=0.833, unit="", n=8,
                            verdict="captures 0.833 of oracle bits vs 0.402 null"),
            "quality": Cell(Mark.EQUIVALENT,
                            verdict="alpha/beta TOST-equivalent to zero within JND "
                                    "(FG-PSNR moves 0.03-0.05 dB, n=18)"),
        }),
        Row("graded strength (by score)", "method", {
            "computation": Cell(Mark.NO_DATA, reason="same as above; not separately timed"),
            "bitrate": Cell(Mark.MEASURED, value=2.55, unit="%", n=8, favouring=1,
                            p_holm=0.0703, verdict="costs bits on 7/8"),
            "quality": Cell(Mark.MEASURED, value=0.0152, unit="", n=8, favouring=0,
                            p_holm=0.0078, verdict="sub_jnd_significant, WRONG direction"),
        }),
        Row("graded strength (damage oracle)", "oracle", {
            "computation": Cell(Mark.NOT_APPLICABLE,
                                reason="oracle requires the restored output; not computable "
                                       "at transmit time by construction"),
            "bitrate": Cell(Mark.NO_DATA, reason="oracle arm scored on quality only"),
            "quality": Cell(Mark.MEASURED, value=0.0, unit="", n=8,
                            verdict="also fails -- locates the failure in the graded "
                                    "transport, not the ranking"),
        }),
        Row("leave-one-SB-out bit oracle", "oracle", {
            "computation": Cell(Mark.NOT_APPLICABLE,
                                reason="requires one encode per superblock; not deployable"),
            "bitrate": Cell(Mark.MEASURED, value=1.000, unit="", n=8,
                            verdict="upper bound; its top quarter is 30.1% of total bitrate"),
            "quality": Cell(Mark.NO_DATA, reason="oracle defined on bits only"),
        }),
    ]
    return rows


def build_goal3(db: str) -> List[Row]:
    """Restoration: restorer-vs-`none` control on an identical bitstream."""
    runs = load(db)
    by_op = score_arms(runs)
    # Unit of analysis is the VIDEO, not the cell. Collect per (label, video)
    # first and collapse to one scalar per video before any counting -- a video
    # contributing four operating points must not weigh four times as much.
    per_lv: Dict[str, Dict[str, List[float]]] = {}
    for op, arms in by_op.items():
        keep = eligible(arms, fg_jnd=FG_PSNR_JND)
        if not keep:
            continue
        best = collapse_duplicates(keep, "quality")
        for label, arm in best.items():
            if arm.gain_jnd is None:
                continue
            per_lv.setdefault(label, {}).setdefault(op[0], []).append(arm.gain_jnd)
    per_label: Dict[str, List[float]] = {
        lab: [sorted(v)[len(v) // 2] for v in vids.values()]
        for lab, vids in per_lv.items()
    }
    rows = [Row("none (null)", "null", {
        "computation": Cell(Mark.NOT_APPLICABLE,
                            reason="no restoration pass is run"),
        "bitrate": Cell(Mark.NOT_APPLICABLE,
                        reason="restoration is client-side; it cannot change transmitted bits"),
        "quality": Cell(Mark.MEASURED, value=0.0, unit=" JND", verdict="reference"),
    })]
    for label in sorted(per_label, key=lambda k: -len(per_label[k])):
        gains = per_label[label]          # one scalar per video
        n = len(gains)
        fav = sum(1 for g in gains if g > 0)
        # signature is (n_positive, n_negative), NOT (successes, n) --
        # passing n as the second arg silently doubles the pair count.
        p = sign_test_p(fav, n - fav)
        med = sorted(gains)[n // 2]
        rows.append(Row(label, "method", {
            "computation": Cell(Mark.NO_DATA,
                                reason="not in the controlled timing campaign; "
                                       "pre-2026-08-05 in-pipeline timings are unpublishable "
                                       "(silent CPU fallback mixed two device populations)"),
            "bitrate": Cell(Mark.NOT_APPLICABLE,
                            reason="restoration is client-side; identical bitstream by design"),
            "quality": Cell(Mark.MEASURED, value=med, unit=" JND", n=n, favouring=fav,
                            p_holm=min(1.0, p), verdict=verdict_for(
                                n, fav, min(1.0, p), abs(med) >= 1.0)),
        }))
    return rows


def render(title: str, rows: Sequence[Row]) -> str:
    out = [f"\n=== {title} ===",
           f"{'method':32s} | {'computation':^13} | {'bitrate':^40} | {'quality':^46}"]
    for r in rows:
        tag = {"null": " (null)", "oracle": " (oracle)"}.get(r.kind, "")
        name = (r.method + tag)[:32]
        cells = [r.cells.get(a, Cell(Mark.NO_DATA, reason="axis not defined for this row"))
                 for a in AXES]
        out.append(f"{name:32s} | {cells[0].render():^13} | "
                   f"{cells[1].render():<40} | {cells[2].render():<46}")
    return "\n".join(out)


def reasons(rows: Sequence[Row]) -> List[str]:
    seen = []
    for r in rows:
        for axis, c in r.cells.items():
            if c.mark in NEEDS_REASON and c.reason not in seen:
                seen.append(f"{r.method} / {axis}: {c.reason}")
    return seen


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results/presley.db")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    g1 = goal1_rows()
    g3 = build_goal3(a.db)

    print("THREE GOALS x THREE AXES")
    print("Quality gates: goals 1-2 on foreground LPIPS, goal 3 on background")
    print("LPIPS (JND 0.05). Bitrate on actual transmitted bits. Computation from")
    print("the controlled campaign only -- no pre-2026-08-05 timing enters it.")
    print(render("Goal 1 - SELECT (selector vs selector, transport fixed)", g1))
    print(render("Goal 3 - RESTORE (restorer vs matched `none` control)", g3))
    print("\n--- why cells are absent (every 'n/a' and '--' must appear here) ---")
    for line in reasons(g1) + reasons(g3):
        print(f"  {line}")

    if a.json:
        payload = {"goal1": [{"method": r.method, "kind": r.kind,
                              "cells": {k: {"mark": c.mark.value, "value": c.value,
                                            "n": c.n, "verdict": c.verdict,
                                            "reason": c.reason}
                                        for k, c in r.cells.items()}} for r in g1 + g3]}
        pathlib.Path(a.json).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
