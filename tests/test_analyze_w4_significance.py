"""The significance layer's failure modes are the ones that already cost this
corpus a result, and every one of them is silent by nature.

  * **Cells voting instead of videos.** A video run at five operating points
    would otherwise outvote one run at a single cell, and coverage in this
    corpus is wildly uneven -- which is how a sign test starts measuring where
    the GPU time went.
  * **Filtering after collapsing.** Both orderings look reasonable; only
    filter-then-collapse reproduces the published table, because collapsing
    first can hand a cell to a run that then fails the eligibility filter and
    the cell is lost entirely.
  * **Holm over the survivors.** Adjusting across only the rows that made it
    into the write-up shrinks the correction every time a losing arm is dropped.
    The family is every candidate ever compared, losers padded in at p=1.0.
  * **A p below 0.05 on too few videos.** Hard rule 2b's n>=8 restorer floor is
    not advisory; the verdict must say underpowered however small the p.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools"
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(_ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "analyze_w4_significance", _TOOLS / "analyze_w4_significance.py")
w4 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = w4
_spec.loader.exec_module(w4)

BASE = w4.BASE_LABEL


def arm(label, bg, dbits=-10.0, dfg=0.0, tag=""):
    transport, _, fill = label.partition("+")
    return w4.Arm(hash=f"{label}{tag}-{bg}-{dbits}", label=label,
                  component="presley_ai", transport=transport, fill=fill,
                  dbits_pct=dbits, dfg_db=dfg, bg_lpips=bg,
                  config=(("degradation", transport), ("tag", tag)))


def op(video, qp=43):
    return (video, "svtav1", qp, 640, 360)


# --------------------------------------------------------------------------
# The unit of analysis is the video, not the cell
# --------------------------------------------------------------------------

def test_a_video_with_many_cells_still_casts_one_vote():
    """Three cells on `bear` where the candidate wins, one cell on `camel`
    where it loses. Counting cells reads 3-1 for the candidate; counting videos
    reads 1-1, which is the honest answer at n=2 videos."""
    by_op = {}
    for qp in (43, 50, 58):
        by_op[op("bear", qp)] = [arm(BASE, 0.30), arm("blur+nafnet", 0.20)]
    by_op[op("camel")] = [arm(BASE, 0.20), arm("blur+nafnet", 0.30)]

    row = w4.Row("blur+nafnet", "quality",
                 w4.paired_deltas(by_op, "blur+nafnet", "quality"))

    assert row.n == 2
    assert (row.base_wins, row.cand_wins) == (1, 1)


def test_a_videos_cells_are_averaged_not_taken_best_of():
    by_op = {op("bear", 43): [arm(BASE, 0.30), arm("blur+nafnet", 0.10)],
             op("bear", 50): [arm(BASE, 0.30), arm("blur+nafnet", 0.50)]}

    row = w4.Row("blur+nafnet", "quality",
                 w4.paired_deltas(by_op, "blur+nafnet", "quality"))

    assert row.n == 1
    assert row.means == [pytest.approx(0.0)]


def test_cells_without_the_baseline_arm_contribute_nothing():
    """Pairing is within an operating point. A cell where only the candidate ran
    is not evidence about the comparison, however good the candidate looks."""
    by_op = {op("bear"): [arm("blur+nafnet", 0.10)],
             op("camel"): [arm(BASE, 0.30), arm("blur+nafnet", 0.20)]}

    row = w4.Row("blur+nafnet", "quality",
                 w4.paired_deltas(by_op, "blur+nafnet", "quality"))

    assert row.videos == ["camel"]


# --------------------------------------------------------------------------
# Eligibility, and the order it is applied in
# --------------------------------------------------------------------------

def test_an_arm_that_saves_no_bits_is_not_a_deployable_comparison():
    by_op = {op("bear"): [arm(BASE, 0.30), arm("blur+nafnet", 0.20, dbits=+5.0)],
             op("camel"): [arm(BASE, 0.30), arm("blur+nafnet", 0.20)]}

    row = w4.Row("blur+nafnet", "quality",
                 w4.paired_deltas(by_op, "blur+nafnet", "quality"))

    assert row.videos == ["camel"]


def test_an_ineligible_baseline_blocks_the_cell_for_every_candidate():
    """Named in the published table as videos "blocked by an ineligible
    baseline" -- no new run fixes them, so they must never be scheduled as
    though they were a coverage gap."""
    by_op = {op("bear"): [arm(BASE, 0.30, dfg=-2.0), arm("blur+nafnet", 0.20)]}

    assert w4.paired_deltas(by_op, "blur+nafnet", "quality") == {}


def test_eligibility_is_applied_before_duplicates_collapse():
    """The ordering that reproduces the published table. `bear` carries two
    `blur+nafnet` runs: the one with the better background is not deployable
    (it loses visible foreground), the other is. Filtering first keeps the cell
    through the deployable run; collapsing first picks the pretty one, then
    drops it, and the video disappears without a word."""
    by_op = {op("bear"): [arm(BASE, 0.30),
                          arm("blur+nafnet", 0.10, dfg=-2.0, tag="pretty"),
                          arm("blur+nafnet", 0.25, tag="deployable")]}

    deltas = w4.paired_deltas(by_op, "blur+nafnet", "quality")

    assert list(deltas) == ["bear"]
    assert deltas["bear"] == [pytest.approx(0.25 - 0.30)]


def test_duplicates_collapse_under_the_objective_being_reported():
    """Best-of on quality and best-of on bitrate select different runs of one
    arm. Reporting a bitrate column off the quality-best run is the silent
    version of comparing two different experiments."""
    by_op = {op("bear"): [arm(BASE, 0.30, dbits=-10.0),
                          arm("blur+nafnet", 0.10, dbits=-2.0, tag="a"),
                          arm("blur+nafnet", 0.25, dbits=-30.0, tag="b")]}

    quality = w4.paired_deltas(by_op, "blur+nafnet", "quality")["bear"]
    bitrate = w4.paired_deltas(by_op, "blur+nafnet", "bitrate")["bear"]

    assert quality == [pytest.approx(0.10 - 0.30)]
    assert bitrate == [pytest.approx(-20.0)]


# --------------------------------------------------------------------------
# Holm, and the n floor
# --------------------------------------------------------------------------

def test_holm_pads_the_family_with_the_candidates_that_were_never_tabulated():
    """m=14 is the number of arms ever compared against the baseline, not the
    number of rows that survived into the report."""
    rows = [w4.Row("blur+nafnet", "quality", {f"v{i}": [-0.1] for i in range(11)})]

    w4.holm_within_family(rows, family_size=14)

    assert rows[0].p_raw == w4.sign_test_p(0, 11)
    assert rows[0].p_holm == pytest.approx(rows[0].p_raw * 14)


def test_a_tiny_p_on_too_few_videos_is_reported_as_underpowered():
    """Hard rule 2b's n>=8 restorer floor. 6/6 gives raw p=0.031, which would
    read as a result if the floor were only advisory."""
    row = w4.Row("freeze+propainter", "quality",
                 {f"v{i}": [0.1] for i in range(6)})
    w4.holm_within_family([row], family_size=1)

    assert row.p_raw < 0.05
    assert row.verdict.startswith("underpowered")


def test_the_significant_verdict_needs_both_the_floor_and_holm():
    row = w4.Row("blur+nafnet", "quality", {f"v{i}": [-0.1] for i in range(11)})
    w4.holm_within_family([row], family_size=14)

    assert row.n >= w4.N_FLOOR and row.p_holm < 0.05
    assert row.verdict == "SIGNIFICANT"


def test_excluded_runs_are_reported_by_reason():
    """The trap this suite exists around: a wave of clean runs lands and n does
    not move, because the arms carry no region LPIPS yet and are skipped in
    silence. It happened twice -- 2026-08-03 and again 2026-08-05 -- so the
    exclusion counts are printed rather than available on a flag."""
    w4.DROPPED.clear()
    w4.DROPPED["no BG-LPIPS (needs presley-evaluate --backfill-lpips)"].extend(
        ["aaaa", "bbbb", "cccc"])

    out = w4.format_drops()

    assert "3" in out and "backfill-lpips" in out
    w4.DROPPED.clear()


def test_no_drops_says_so_rather_than_printing_nothing():
    w4.DROPPED.clear()
    assert w4.format_drops() == "no runs dropped"
