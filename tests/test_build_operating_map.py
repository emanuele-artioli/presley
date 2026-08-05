"""The operating map is a *decision rule*, so its failure modes are claims the
article would then make and could not defend.

Four of them, each with a test below:

  * **Naming a winner that is not separable.** The map's honest core is that
    most cells have no winner. A tie silently resolved by float ordering turns
    "these are indistinguishable" into a published recommendation.
  * **Recommending an arm that is not deployable** -- one that spends more bits
    than the baseline, or costs visible foreground quality. Those fail the
    map's own feasibility test and can never be an answer to "what do I deploy".
  * **Ranking on restoration gain instead of absolute quality.** This is the
    standing blackout error: the corpus's largest gain and its worst absolute
    result, because it starts from the worst place.
  * **Scoring residuals against a line that is not a ladder.** A positive-slope
    fit has not found the rate-damage tradeoff; residuals from it are distances
    from an arbitrary line dressed up as "best of both worlds" candidates.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "build_operating_map.py"
_spec = importlib.util.spec_from_file_location("build_operating_map", _TOOL)
bom = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves annotations through
# sys.modules[cls.__module__], so a module loaded purely by path raises.
sys.modules[_spec.name] = bom
_spec.loader.exec_module(bom)

OP = ("bear", "svtav1", 50, 640, 360)


def arm(label, bg, dbits=-10.0, dfg=0.0, fps=None, unrestored=None, config=None):
    """`config` is the config-minus-fill signature -- the transmitted bitstream.

    It defaults to something unique per (transport, bits) so each arm is its own
    ladder point. Pass the SAME config to two arms to model what really happens
    in the corpus: several fills sharing one `none` control.
    """
    transport, _, fill = label.partition("+")
    if config is None:
        config = (("degradation", transport), ("bits_marker", dbits))
    return bom.Arm(hash=label, label=label, component="presley_ai",
                   transport=transport, fill=fill, dbits_pct=dbits, dfg_db=dfg,
                   bg_lpips=bg, unrestored_bg_lpips=unrestored, fps=fps,
                   config=config)


# --------------------------------------------------------------------------
# Separability -- refusing to name a winner is the map's most important output
# --------------------------------------------------------------------------

def test_sub_jnd_lead_is_a_tie_not_a_win():
    """0.04 of BG-LPIPS is below the 0.05 JND: imperceptible, so not a
    recommendation. This is the "never dress up imperceptible deltas" rule
    expressed as the map's core refusal."""
    cell = bom.decide(OP, [arm("downsample+realesrgan", 0.20),
                           arm("blur+nafnet", 0.24)], "quality_first")

    assert cell.verdict == "tie_within_threshold"
    assert cell.separable is False
    assert cell.winner is None
    assert set(cell.tied) == {"downsample+realesrgan", "blur+nafnet"}


def test_lead_of_exactly_one_jnd_separates():
    """The boundary is inclusive -- one full JND is the stated threshold, and a
    strict comparison here would quietly discard every borderline cell."""
    cell = bom.decide(OP, [arm("downsample+realesrgan", 0.20),
                           arm("blur+nafnet", 0.25)], "quality_first")

    assert cell.verdict == "separable"
    assert cell.winner == "downsample+realesrgan"
    assert cell.runner_up == "blur+nafnet"
    assert cell.margin == pytest.approx(1.0)


def test_every_tied_arm_is_listed_not_just_the_runner_up():
    """A three-way tie reported as a two-way one understates how undetermined
    the cell is, which is exactly the thing the map exists to be honest about."""
    cell = bom.decide(OP, [arm("a+x", 0.20), arm("b+x", 0.22),
                           arm("c+x", 0.23)], "quality_first")

    assert cell.tied == ["a+x", "b+x", "c+x"]


def test_single_eligible_arm_is_not_a_separable_winner():
    """One arm cannot beat a field of one. It is reported as the only option,
    never as evidence that transport choice matters."""
    cell = bom.decide(OP, [arm("downsample+realesrgan", 0.20),
                           arm("blur+nafnet", 0.24, dbits=+5.0)], "quality_first")

    assert cell.verdict == "single_eligible_arm"
    assert cell.separable is False
    assert cell.n_eligible == 1


# --------------------------------------------------------------------------
# Feasibility -- a recommendation has to be deployable
# --------------------------------------------------------------------------

def test_arm_that_spends_bits_is_never_recommended():
    """Beating the baseline on background quality while costing more bits is
    not a win for a compression pipeline; it is the pipeline not running."""
    cell = bom.decide(OP, [arm("freeze+propainter", 0.10, dbits=+4.0),
                           arm("downsample+realesrgan", 0.30)], "quality_first")

    assert cell.winner_hash == "downsample+realesrgan"
    assert cell.n_eligible == 1


def test_arm_costing_visible_foreground_quality_is_excluded():
    """FG-PSNR beyond the 0.5 dB JND is a visible cost, so background quality
    bought with it is not free and the arm is not a candidate."""
    cell = bom.decide(OP, [arm("blackout+propainter", 0.10, dfg=-0.8),
                           arm("downsample+realesrgan", 0.30, dfg=-0.4)],
                      "quality_first")

    assert cell.winner_hash == "downsample+realesrgan"
    assert cell.n_eligible == 1


def test_no_eligible_arm_reports_that_rather_than_the_least_bad_one():
    cell = bom.decide(OP, [arm("a+x", 0.10, dbits=+1.0),
                           arm("b+x", 0.20, dfg=-2.0)], "quality_first")

    assert cell.verdict == "no_eligible_arm"
    assert cell.winner is None and cell.winner_hash is None


# --------------------------------------------------------------------------
# The blackout trap -- gain flatters, absolute quality decides
# --------------------------------------------------------------------------

def test_largest_restoration_gain_does_not_win_on_worst_absolute_quality():
    """Blackout, in miniature: it posts the corpus's largest gain (it starts
    from the worst place) and ends far worse than downsample. The objective
    ranks on the absolute restored value, so the flattering gain loses."""
    blackout = arm("blackout+propainter", 0.374, unrestored=0.480)
    downsample = arm("downsample+realesrgan", 0.220, unrestored=0.297)
    assert blackout.gain_jnd > downsample.gain_jnd     # the flattering figure

    cell = bom.decide(OP, [blackout, downsample], "quality_first")

    assert cell.winner == "downsample+realesrgan"
    assert cell.winner_bg_lpips == pytest.approx(0.220)


def test_gain_is_none_without_a_control_rather_than_computed_from_the_baseline():
    """Restoration gain is only meaningful against the arm's *own* unrestored
    control. Falling back to anything else would compare across transports and
    manufacture a gain out of a transport difference."""
    assert arm("blur+nafnet", 0.30).gain_jnd is None


# --------------------------------------------------------------------------
# rate_first -- a different objective, and not a JND question
# --------------------------------------------------------------------------

def test_rate_first_ranks_on_bits_and_can_disagree_with_quality_first():
    """The two objectives disagreeing is a finding, not a bug: a recommendation
    that survives both scalarizations is the strong one."""
    arms = [arm("blackout+propainter", 0.374, dbits=-24.6),
            arm("downsample+realesrgan", 0.220, dbits=-14.8)]

    quality = bom.decide(OP, arms, "quality_first")
    rate = bom.decide(OP, arms, "rate_first")

    assert quality.winner == "downsample+realesrgan"
    assert rate.winner == "blackout+propainter"
    assert rate.margin_unit == "pp_bits"
    assert rate.margin == pytest.approx(9.8)


def test_sub_point_bitrate_lead_is_a_tie():
    """Under a percentage point of bitrate is inside this corpus's measurement
    tolerance, so it is called a tie rather than a win."""
    cell = bom.decide(OP, [arm("a+x", 0.20, dbits=-15.0),
                           arm("b+x", 0.20, dbits=-14.5)], "rate_first")

    assert cell.verdict == "tie_within_threshold"


def test_unknown_objective_is_refused_rather_than_silently_defaulted():
    with pytest.raises(ValueError):
        bom.decide(OP, [arm("a+x", 0.2), arm("b+x", 0.3)], "lowest_bitrate_ever")


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------

def test_positive_slope_fit_is_not_a_ladder():
    """More bits ought to mean less damage. A fit saying otherwise has not
    found the tradeoff, and residuals from it are meaningless."""
    fit = bom.fit_ladder([(-20.0, 0.20), (-10.0, 0.30), (0.0, 0.40)])

    assert fit.slope > 0
    assert fit.is_ladder is False


def test_ladder_needs_three_points():
    """Two points fit a line exactly, so every residual would be zero and every
    cell would falsely report 'no off-ladder arm'."""
    assert bom.fit_ladder([(-20.0, 0.40), (-10.0, 0.30)]) is None


def test_residual_is_positive_when_an_arm_beats_the_ladder():
    """Sign convention, and the reason the whole scalar exists: positive means
    less damage than the arm's bit saving should have cost -- a 'best of both
    worlds' candidate."""
    on_ladder = [arm("a+f", 0.5, dbits=-30.0, unrestored=0.40),
                 arm("b+f", 0.5, dbits=-20.0, unrestored=0.30),
                 arm("c+f", 0.5, dbits=-10.0, unrestored=0.20)]
    off = arm("d+f", 0.5, dbits=-25.0, unrestored=0.25)   # 0.10 under the line
    by_op = {OP: on_ladder + [off]}

    fits = bom.attach_residuals(by_op)

    assert fits[OP].is_ladder
    assert off.residual_jnd > 1.0
    # The off-ladder arm is in the fit it is scored against, so it drags the
    # line toward itself: 0.10 of raw distance scores 1.37x JND rather than the
    # naive 2.0x. Understating is the safe direction for a "best of both
    # worlds" claim, and with 3-6 points per cell leave-one-out would do the
    # opposite. If this number moves, the fit's leverage handling changed.
    assert off.residual_jnd == pytest.approx(1.371, abs=1e-3)
    for a in on_ladder:
        assert a.residual_jnd < off.residual_jnd


def test_several_fills_of_one_transport_count_as_one_ladder_point():
    """Otherwise a transport that happened to be run with six restorers would
    drag the fit six times harder than one run with a single restorer -- the
    same coverage confound the map avoids by never pooling across cells."""
    # One bitstream, five restorers run on it -> one control, one ladder point.
    one_bitstream = (("degradation", "a"), ("blur_kernel", 7))
    shared = [arm(f"a+fill{i}", 0.5, dbits=-30.0, unrestored=0.40,
                  config=one_bitstream) for i in range(5)]
    by_op = {OP: shared + [arm("b+f", 0.5, dbits=-20.0, unrestored=0.30),
                           arm("c+f", 0.5, dbits=-10.0, unrestored=0.20)]}

    fits = bom.attach_residuals(by_op)

    assert fits[OP].n_points == 3
    assert fits[OP].slope == pytest.approx(-0.01)


def test_residual_from_a_cloud_is_kept_out_of_the_gate(capsys):
    """A cell whose arms do not form a ladder (real: pigs@QP50, R^2=0.016) still
    produces residuals, and they can top the table. The falsification gate must
    not pass on one -- 'best of both worlds exists' cannot rest on a distance
    from a cloud."""
    scatter = [arm("a+f", 0.5, dbits=-30.0, unrestored=0.30),
               arm("b+f", 0.5, dbits=-20.0, unrestored=0.45),
               arm("c+f", 0.5, dbits=-10.0, unrestored=0.29)]
    by_op = {OP: scatter}

    fits = bom.attach_residuals(by_op)
    assert fits[OP].r2 < bom.LADDER_R2_MIN
    out = bom.report_ladder(by_op, fits, verbose=False)

    assert out["best_deployable_residual_any_fit"] is not None
    assert out["best_deployable_residual_jnd"] is None
    assert "!" in capsys.readouterr().out


def test_same_transport_at_different_strengths_gets_different_controls():
    """The bug this pins, found on real data: `dancing`@QP43 carries three
    `blur` runs at blur_kernel 7 / 15 / 31. Under the old key --
    (component, transport, block_size, shrink_amount) -- all three collapsed to
    one entry, the controls dict kept whichever was inserted last, and two of
    the three arms were scored against *another run's* damage while keeping
    their own bitrate. Damage and bits from different experiments is exactly how
    a rate-damage ladder acquires a positive slope."""
    def run(blur_kernel, bg, fill):
        return bom.Run(
            hash=f"k{blur_kernel}-{fill}", component="presley_ai", video="dancing",
            codec="svtav1", qp=43, width=640, height=360, transport="blur",
            fill=fill, block_size=16, shrink_amount=0.25, bits=1_000_000,
            frames=80, restore_s=20.0, bg_lpips=bg, fg_psnr=30.0,
            raw_config={"degradation": "blur", "blur_kernel": blur_kernel,
                        "block_size": 16, "shrink_amount": 0.25,
                        "restorer": fill},
        )
    runs = [run(k, bg, fill)
            for k, bg, fill in ((7, 0.30, "none"), (15, 0.40, "none"),
                                (31, 0.50, "none"), (7, 0.25, "nafnet"),
                                (15, 0.35, "nafnet"), (31, 0.45, "nafnet"))]

    controls = bom.build_controls(runs)

    assert len(controls) == 3, "each blur strength is its own control"
    # And each restored arm must draw damage from its OWN strength.
    by_kernel = {r.raw_config["blur_kernel"]: r for r in runs
                 if r.fill == "nafnet"}
    for kernel, expected_damage in ((7, 0.30), (15, 0.40), (31, 0.50)):
        ctrl = controls[(by_kernel[kernel].op, by_kernel[kernel].config)]
        assert ctrl.bg_lpips == expected_damage


def test_two_indistinguishable_controls_are_refused_not_guessed_between():
    """A collision under the full-config key means two runs are identical in
    config yet hashed differently. Silently keeping the last one is what caused
    the original defect, so the pair is dropped and recorded instead."""
    bom.AMBIGUOUS_CONTROLS.clear()
    cfg = {"degradation": "blur", "block_size": 8, "restorer": "none"}
    dupes = [bom.Run(hash=f"dup{i}", component="presley_ai", video="bear",
                     codec="svtav1", qp=50, width=640, height=360,
                     transport="blur", fill="none", block_size=8,
                     shrink_amount=0.25, bits=1e6, frames=80, restore_s=20.0,
                     bg_lpips=0.3 + 0.1 * i, fg_psnr=30.0, raw_config=dict(cfg))
             for i in range(2)]

    controls = bom.build_controls(dupes)

    assert controls == {}
    assert len(bom.AMBIGUOUS_CONTROLS) == 1
    bom.AMBIGUOUS_CONTROLS.clear()


def test_arms_without_a_control_still_enter_the_map():
    """The matched `none` control is far sparser than the baselines. Requiring
    it for the map (rather than only for the ladder) would drop about three
    quarters of the operating points for a scalar the map does not need."""
    a = arm("downsample+realesrgan", 0.20)
    by_op = {OP: [a, arm("blur+nafnet", 0.30)]}

    bom.attach_residuals(by_op)

    assert a.residual_jnd is None
    assert bom.decide(OP, by_op[OP], "quality_first").winner == "downsample+realesrgan"


# --------------------------------------------------------------------------
# Cost, and the threshold sensitivity pass
# --------------------------------------------------------------------------

def test_slower_and_worse_arm_is_marked_dominated():
    better = arm("a+x", 0.20, fps=4.0)
    worse = arm("b+x", 0.30, fps=1.0)
    untimed = arm("c+x", 0.40)

    bom.mark_cost_dominated([better, worse, untimed])

    assert worse.cost_dominated and not better.cost_dominated
    # No timing is not evidence of being dominated; it is missing data.
    assert not untimed.cost_dominated


def test_faster_but_worse_arm_is_on_the_frontier():
    """Cost is a real axis, so trading quality for speed is a legitimate
    operating point rather than a dominated one."""
    quality = arm("a+x", 0.20, fps=0.8)
    fast = arm("b+x", 0.30, fps=4.5)

    bom.mark_cost_dominated([quality, fast])

    assert not quality.cost_dominated and not fast.cost_dominated


def test_threshold_only_recommendation_is_visible_in_the_sensitivity_pass(capsys):
    """A 0.04 lead is a winner at a 0.03 threshold and a tie at 0.05/0.08.
    Publishing it as a decision rule would be publishing the constant."""
    by_op = {OP: [arm("a+x", 0.20), arm("b+x", 0.24)]}

    out = bom.report_sensitivity(by_op, [0.03, 0.05, 0.08])

    assert out["per_threshold"]["0.03"]["separable"] == 1
    assert out["per_threshold"]["0.05"]["separable"] == 0
    assert out["tight_only_cells"] == 1
    assert out["flipping_cells"] == 0        # the ranking cannot move; the verdict does


def test_bounds_check_fires_an_alarm_outside_the_pre_registered_range():
    """A fired bound must read as ALARM, not as a number to quote. The whole
    point of pre-registering is that the tool says so without being asked."""
    status, _ = bom.bounds_check("separable_pct", 5.0, "separable cells = 5.0%")
    assert status == "ALARM"

    status, _ = bom.bounds_check("separable_pct", 41.3, "separable cells = 41.3%")
    assert status == "in band"

    # Between plausible and alarm: not an alarm, but not in band either.
    status, _ = bom.bounds_check("separable_pct", 70.0, "separable cells = 70.0%")
    assert status == "outside plausible"


def test_gate_reports_a_stop_when_no_cell_has_a_winner(capsys):
    """The gate's job is to end the plan cheaply. 'No separable winner
    anywhere' must come out as a STOP, not as an empty table."""
    empty = {"separable": 0, "distinct_winners": 0, "separable_pct": 0.0}
    ladder = {"n": 0, "best_deployable_residual_jnd": None}
    sens = {"flipping_cells": 0}

    gate = bom.report_gate(empty, empty, ladder, sens)

    assert gate["no_separable_winner"] is True
    assert gate["passes"] is False
    assert "STOPS the plan" in capsys.readouterr().out
