"""The tradeoff surface replaces a claim that died, so its own failure modes are
the ones that killed the predecessor.

  * **Pooling across cells without matching.** An arm run at 97 operating points
    and one run at 8 are not comparable by their medians. Doing it anyway
    produced a withdrawn motion correlation (Wave 2A), a physically impossible
    throughput ordering (Wave 2C), and here moves blackout's measured bitrate
    saving from -10.0% to -25.5%.
  * **Calling a swap "free" while measuring the wrong axis.** The first cost
    analysis found arms that were faster at indistinguishable *quality* and
    never checked the *bitrate* -- the axis a rate-first deployer optimises.
  * **An unstated duplicate-aggregation rule.** Wave 2B's sign tests read 9/9 or
    8/9 purely on best-of vs first-of, with the rule written down nowhere.
  * **Treating missing data as a bad score.** An arm with no timing must not be
    deleted from the frontier for it.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))
_spec = importlib.util.spec_from_file_location(
    "build_tradeoff_surface", _TOOLS / "build_tradeoff_surface.py")
bts = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bts
_spec.loader.exec_module(bts)

OP = ("bear", "svtav1", 50, 640, 360)
OP2 = ("camel", "svtav1", 50, 640, 360)


def arm(label, bg, dbits=-10.0, dfg=0.0, fps=None, config=None):
    transport, _, fill = label.partition("+")
    if config is None:
        config = (("degradation", transport), ("marker", (bg, dbits, fps)))
    return bts.Arm(hash=f"{label}-{bg}", label=label, component="presley_ai",
                   transport=transport, fill=fill, dbits_pct=dbits, dfg_db=dfg,
                   bg_lpips=bg, fps=fps, config=config)


# --------------------------------------------------------------------------
# The conflict measure -- the headline claim
# --------------------------------------------------------------------------

def test_faster_at_equal_quality_but_costing_bits_is_not_free():
    """The exact error being corrected. `bear`@QP58 in the real corpus: the rate
    winner saves 30.3% at 1.46 fps; a 6x faster arm ties on quality but saves
    only 8.2%. Calling that "replaceable for free" measures quality and reports
    it as though it were bitrate."""
    by_op = {OP: [arm("blackout+propainter", 0.432, dbits=-30.3, fps=1.46),
                  arm("freeze+telea", 0.475, dbits=-8.2, fps=8.78)]}

    cells = bts.conflict_scan(by_op)

    assert len(cells) == 1
    cell = cells[0]
    assert cell.rate_winner == "blackout+propainter"
    assert cell.faster_alt == "freeze+telea"
    assert cell.free is False
    assert cell.bits_given_up_pp == pytest.approx(22.1, abs=0.1)
    assert cell.speedup == pytest.approx(6.01, abs=0.01)


def test_faster_at_equal_quality_AND_equal_bits_is_free():
    """The genuine Pareto improvement, which does exist -- `drift-turn`-shaped:
    same bits, same-or-better quality, several times faster."""
    by_op = {OP: [arm("blur+nafnet", 0.266, dbits=-11.2, fps=0.52),
                  arm("downsample+realesrgan", 0.262, dbits=-11.3, fps=5.09)]}

    cell = bts.conflict_scan(by_op)[0]

    assert cell.rate_winner == "downsample+realesrgan"  # it also saves most bits
    assert cell.faster_alt is None                      # nothing beats it on speed


def test_quality_difference_beyond_jnd_is_not_an_alternative_at_all():
    """A faster arm that is visibly worse is a different operating point, not a
    swap. Counting it would turn the conflict measure into a speed ranking."""
    by_op = {OP: [arm("blackout+propainter", 0.30, dbits=-30.0, fps=1.0),
                  arm("freeze+telea", 0.50, dbits=-8.0, fps=9.0)]}

    cell = bts.conflict_scan(by_op)[0]

    assert cell.faster_alt is None


def test_a_swap_from_an_arm_to_itself_is_not_reported():
    """Two configs of one label -- blur_kernel 7 vs 31 -- differ in within-arm
    tuning, not in the choice a deployer makes between arms. Before duplicates
    were collapsed here, the report showed rows reading
    `blur+nafnet -> blur+nafnet`."""
    by_op = {OP: [arm("blur+nafnet", 0.30, dbits=-12.0, fps=1.0,
                      config=(("blur_kernel", 7),)),
                  arm("blur+nafnet", 0.31, dbits=-9.0, fps=4.0,
                      config=(("blur_kernel", 31),))]}

    assert bts.conflict_scan(by_op) == []


# --------------------------------------------------------------------------
# Matched vs naive pooling
# --------------------------------------------------------------------------

def test_matched_pooling_uses_only_cells_carrying_every_arm():
    """The methodological core. Arm B appears in one cell where it looks good;
    arm A appears in both. Pooling naively credits B with its lucky cell."""
    by_op = {
        OP: [arm("a+f", 0.20, dbits=-10.0), arm("b+f", 0.10, dbits=-30.0)],
        OP2: [arm("a+f", 0.22, dbits=-12.0)],
    }

    matched = bts.matched_summary(by_op, ["a+f", "b+f"])

    assert matched["cells"] == 1
    assert matched["arms"]["a+f"]["bg_lpips"] == pytest.approx(0.20)


def test_matched_summary_is_none_when_no_cell_carries_all_arms():
    """Better to say the comparison does not exist than to quietly compare on
    whatever overlap happens to be there."""
    by_op = {OP: [arm("a+f", 0.2)], OP2: [arm("b+f", 0.3)]}

    assert bts.matched_summary(by_op, ["a+f", "b+f"]) is None


def test_naive_and_matched_can_disagree_and_both_are_reported():
    """Not a bug -- the gap between them IS the coverage confound, and it is the
    reason both tables are printed. On the real corpus this gap moves blackout
    from -10.0% to -25.5%."""
    by_op = {
        OP: [arm("a+f", 0.20, dbits=-10.0), arm("b+f", 0.30, dbits=-25.0)],
        OP2: [arm("a+f", 0.20, dbits=-10.0)],
        ("dog", "svtav1", 50, 640, 360): [arm("a+f", 0.20, dbits=-1.0)],
    }

    naive = bts.naive_summary(by_op, min_cells=1)
    matched = bts.matched_summary(by_op, ["a+f", "b+f"])

    naive_a = next(r for r in naive if r["arm"] == "a+f")
    assert naive_a["dbits_pct"] == pytest.approx(-10.0)   # median over 3 cells
    assert matched["arms"]["a+f"]["dbits_pct"] == pytest.approx(-10.0)
    assert matched["cells"] == 1


# --------------------------------------------------------------------------
# The aggregation rule, stated and pinned
# --------------------------------------------------------------------------

def test_duplicates_collapse_best_of_under_the_stated_objective():
    """Leaving this implicit already changed published counts (9/9 vs 8/9). The
    rule is best-of, and "best" must follow the objective being reported --
    otherwise a bitrate column silently gets the quality-best row."""
    arms = [arm("blur+nafnet", 0.20, dbits=-5.0, fps=1.0, config=(("k", 7),)),
            arm("blur+nafnet", 0.40, dbits=-25.0, fps=9.0, config=(("k", 31),))]

    assert bts.collapse_duplicates(arms, "quality")["blur+nafnet"].bg_lpips == 0.20
    assert bts.collapse_duplicates(arms, "bitrate")["blur+nafnet"].dbits_pct == -25.0
    assert bts.collapse_duplicates(arms, "speed")["blur+nafnet"].fps == 9.0


def test_collapse_keeps_distinct_labels_apart():
    arms = [arm("blur+nafnet", 0.20), arm("downsample+realesrgan", 0.30)]

    assert set(bts.collapse_duplicates(arms, "quality")) == {
        "blur+nafnet", "downsample+realesrgan"}


# --------------------------------------------------------------------------
# Dominance
# --------------------------------------------------------------------------

def test_worse_on_all_three_axes_is_dominated():
    good = arm("a+f", 0.20, dbits=-20.0, fps=5.0)
    bad = arm("b+f", 0.30, dbits=-10.0, fps=1.0)

    assert bts.dominates(good, bad)
    assert not bts.dominates(bad, good)
    assert bts.pareto_front([good, bad]) == [good]


def test_better_on_one_axis_survives_on_the_frontier():
    """The tradeoff claim depends on this: an arm that is worse on quality but
    saves far more bits is a legitimate choice, not a dominated one."""
    quality = arm("a+f", 0.20, dbits=-10.0, fps=4.0)
    rate = arm("b+f", 0.35, dbits=-30.0, fps=1.0)

    assert bts.pareto_front([quality, rate]) == [quality, rate]


def test_missing_timing_is_not_treated_as_being_slow():
    """Absent data is not a bad score. Treating it as one would silently delete
    every untimed arm from the frontier."""
    timed = arm("a+f", 0.20, dbits=-20.0, fps=5.0)
    untimed = arm("b+f", 0.20, dbits=-20.0, fps=None)

    # Equal on the two axes both carry -> neither dominates the other.
    assert not bts.dominates(timed, untimed)
    assert untimed in bts.pareto_front([timed, untimed])
