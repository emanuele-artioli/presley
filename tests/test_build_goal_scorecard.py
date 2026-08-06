"""Tests for the three-goal scorecard.

These pin the two defects that were live in the first working version of the
tool, because both produced plausible, well-formatted, wrong output rather than
an error:

1. `sign_test_p` takes `(n_positive, n_negative)`, not `(successes, n)`.
   Calling it the second way silently doubles the pair count and turned a 4/4
   unanimous result into p=1.000.
2. The unit of analysis is the **video**. Counting arms instead of videos let
   `favouring` exceed `n` (a rendered "5/3"), because a video contributing
   several operating points was counted several times.

Plus the invariant the table's trustworthiness rests on: an absent cell must say
why it is absent.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from build_goal_scorecard import (  # noqa: E402
    Cell, Mark, Row, goal1_rows, reasons, verdict_for,
)
from presley.suite import sign_test_p  # noqa: E402


class TestAbsentCellsMustExplainThemselves:
    """A blank cell that does not say why is indistinguishable from a bug."""

    @pytest.mark.parametrize("mark", [Mark.NOT_APPLICABLE, Mark.NO_DATA])
    def test_absent_mark_without_reason_raises(self, mark):
        with pytest.raises(ValueError, match="must carry a reason"):
            Cell(mark)

    @pytest.mark.parametrize("mark", [Mark.NOT_APPLICABLE, Mark.NO_DATA])
    def test_whitespace_reason_is_not_a_reason(self, mark):
        with pytest.raises(ValueError, match="must carry a reason"):
            Cell(mark, reason="   ")

    @pytest.mark.parametrize("mark", [Mark.NOT_APPLICABLE, Mark.NO_DATA])
    def test_absent_mark_with_reason_is_fine(self, mark):
        assert Cell(mark, reason="because X").reason == "because X"

    def test_measured_cell_requires_a_value(self):
        with pytest.raises(ValueError, match="must carry a value"):
            Cell(Mark.MEASURED)

    def test_every_absent_cell_in_goal1_is_listed_in_the_reasons_block(self):
        rows = goal1_rows()
        absent = sum(1 for r in rows for c in r.cells.values()
                     if c.mark in (Mark.NOT_APPLICABLE, Mark.NO_DATA))
        assert len(reasons(rows)) == absent, (
            "every n/a and -- must surface in the reasons block, or the table "
            "hides its own gaps"
        )


class TestSignTestArgumentOrder:
    """Defect 1: (n_positive, n_negative), never (successes, n)."""

    def test_unanimous_four_of_four(self):
        assert sign_test_p(4, 0) == pytest.approx(0.125)

    def test_the_wrong_call_is_visibly_different(self):
        # What the buggy version computed: n=8 pairs, k=4 -> p=1.0
        assert sign_test_p(4, 4) == pytest.approx(1.0)
        assert sign_test_p(4, 0) != sign_test_p(4, 4)

    @pytest.mark.parametrize("n,expected", [(3, 0.25), (4, 0.125), (6, 0.03125)])
    def test_unanimous_floor_matches_hard_rule_2b(self, n, expected):
        assert sign_test_p(n, 0) == pytest.approx(expected)

    def test_n6_is_the_first_size_that_can_reach_alpha(self):
        assert sign_test_p(5, 0) > 0.05
        assert sign_test_p(6, 0) <= 0.05


class TestVerdictLadder:
    """`underpowered` and `not_significant` are different claims."""

    def test_below_the_floor_is_underpowered_not_no_effect(self):
        # 4/4 unanimous: p=0.125, but no n=4 result could ever reach 0.05
        assert verdict_for(4, 4, sign_test_p(4, 0), True) == "underpowered"

    def test_split_result_is_not_significant(self):
        assert verdict_for(10, 7, 0.344, True) == "not_significant"

    def test_significant_and_jnd_clearing_is_a_win(self):
        assert verdict_for(10, 10, 0.002, True) == "perceptual_win"

    def test_significant_but_sub_jnd_is_not_dressed_up_as_a_win(self):
        assert verdict_for(10, 10, 0.002, False) == "sub_jnd_significant"


class TestUnitOfAnalysis:
    """Defect 2: favouring counts videos, so it can never exceed n."""

    def test_favouring_never_exceeds_n_in_goal1(self):
        for r in goal1_rows():
            for axis, c in r.cells.items():
                if c.favouring is not None and c.n is not None:
                    assert c.favouring <= c.n, f"{r.method}/{axis}: {c.favouring}/{c.n}"

    def test_null_row_is_present_and_carries_the_random_baseline(self):
        rows = goal1_rows()
        nulls = [r for r in rows if r.kind == "null"]
        assert nulls, "the null row is the interpretive anchor; it must be a row"
        # 0.833 is meaningless without 0.402 beside it
        assert nulls[0].cells["bitrate"].value == pytest.approx(0.402)

    def test_equivalent_mark_renders_as_tost_not_as_a_null_result(self):
        c = Cell(Mark.EQUIVALENT, verdict="TOST")
        assert "TOST" in c.render()
