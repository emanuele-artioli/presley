"""The sweep finder exists because an analysis key that omitted a parameter made
paid-for experiments invisible. Its own failure modes are variants of that.

  * **Attributing a two-parameter difference to one parameter.** Two runs
    differing in `blur_kernel` AND `block_size` are not a kernel sweep; calling
    them one hands the block-size effect to the kernel -- the same shape as the
    control-matching bug that revealed these sweeps.
  * **Reporting a comparison between different arms as a sweep.** The keys that
    name the arm (transport, fill, video, codec) can never be the swept
    parameter.
  * **Calling a single run a sweep.** One value is not a comparison.
  * **Getting the direction backwards on an axis where lower is better.** Two of
    the three axes are lower-better and one is higher-better, which is exactly
    the kind of asymmetry that silently inverts a recommendation.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools"
sys.path.insert(0, str(_TOOLS))
_spec = importlib.util.spec_from_file_location(
    "analyze_parameter_sweeps", _TOOLS / "analyze_parameter_sweeps.py")
aps = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = aps
_spec.loader.exec_module(aps)

OP = ("bear", "svtav1", 43, 640, 360)
OP2 = ("camel", "svtav1", 43, 640, 360)


def arm(label, bg, config, dbits=-10.0, fps=None):
    transport, _, fill = label.partition("+")
    return aps.Arm(hash=f"{label}-{sorted(config.items())}", label=label,
                   component="presley_ai", transport=transport, fill=fill,
                   dbits_pct=dbits, dfg_db=0.0, bg_lpips=bg, fps=fps,
                   config=tuple(sorted(config.items())))


def test_a_swept_key_is_found_when_everything_else_matches():
    by_op = {OP: [arm("blur+nafnet", 0.29, {"blur_kernel": 7, "block_size": 16}),
                  arm("blur+nafnet", 0.36, {"blur_kernel": 31, "block_size": 16})]}

    sweeps = aps.find_sweeps(by_op)

    assert list(sweeps) == ["blur_kernel"]
    assert [v for v, _ in sweeps["blur_kernel"][0].points] == [7, 31]


def test_two_keys_differing_at_once_is_not_a_sweep_of_either():
    """The error being guarded: this pair would otherwise be reported as a
    blur_kernel sweep whose effect is really block size."""
    by_op = {OP: [arm("blur+nafnet", 0.29, {"blur_kernel": 7, "block_size": 8}),
                  arm("blur+nafnet", 0.36, {"blur_kernel": 31, "block_size": 16})]}

    assert aps.find_sweeps(by_op) == {}


def test_a_third_run_rescues_the_pair_it_can_be_matched_with():
    """Adding a run that differs only in the kernel makes that pair a sweep,
    while the block-size-confounded pair stays excluded."""
    by_op = {OP: [arm("blur+nafnet", 0.29, {"blur_kernel": 7, "block_size": 8}),
                  arm("blur+nafnet", 0.36, {"blur_kernel": 31, "block_size": 16}),
                  arm("blur+nafnet", 0.33, {"blur_kernel": 31, "block_size": 8})]}

    sweeps = aps.find_sweeps(by_op)

    assert len(sweeps["blur_kernel"]) == 1
    assert {v for v, _ in sweeps["blur_kernel"][0].points} == {7, 31}


def test_different_arms_are_never_a_sweep():
    by_op = {OP: [arm("blur+nafnet", 0.29, {"degradation": "blur"}),
                  arm("freeze+propainter", 0.36, {"degradation": "freeze"})]}

    assert aps.find_sweeps(by_op) == {}


def test_one_run_is_not_a_sweep():
    assert aps.find_sweeps({OP: [arm("blur+nafnet", 0.29, {"blur_kernel": 7})]}) == {}


def test_the_best_value_is_the_lowest_on_lower_better_axes():
    group = aps.find_sweeps(
        {OP: [arm("blur+nafnet", 0.29, {"blur_kernel": 7}, dbits=-5.0),
              arm("blur+nafnet", 0.36, {"blur_kernel": 31}, dbits=-30.0)]}
    )["blur_kernel"][0]

    assert group.best("quality") == 7      # lower BG-LPIPS
    assert group.best("bitrate") == 31     # more negative delta = fewer bits


def test_the_best_value_is_the_highest_on_speed():
    """The one higher-better axis. Getting this backwards would recommend the
    slowest setting as the fast one."""
    group = aps.find_sweeps(
        {OP: [arm("blur+nafnet", 0.29, {"blur_kernel": 7}, fps=1.0),
              arm("blur+nafnet", 0.36, {"blur_kernel": 31}, fps=9.0)]}
    )["blur_kernel"][0]

    assert group.best("speed") == 31


def test_an_axis_with_no_timing_yields_no_speed_summary():
    """Absent timing is not a bad score; the axis is simply not reported."""
    groups = aps.find_sweeps(
        {OP: [arm("blur+nafnet", 0.29, {"blur_kernel": 7}),
              arm("blur+nafnet", 0.36, {"blur_kernel": 31})]}
    )["blur_kernel"]

    assert "speed" not in aps.summarise("blur_kernel", groups)["axes"]


def test_the_summary_counts_videos_separately_from_groups():
    """Two cells of one video are two groups but one video, and the n that
    matters for any later significance claim is the video count."""
    by_op = {}
    for op in (OP, OP2):
        by_op[op] = [arm("blur+nafnet", 0.29, {"blur_kernel": 7}),
                     arm("blur+nafnet", 0.36, {"blur_kernel": 31})]
    by_op[("bear", "svtav1", 50, 640, 360)] = [
        arm("blur+nafnet", 0.29, {"blur_kernel": 7}),
        arm("blur+nafnet", 0.36, {"blur_kernel": 31})]

    summary = aps.summarise("blur_kernel", aps.find_sweeps(by_op)["blur_kernel"])

    assert summary["n_groups"] == 3
    assert summary["n_videos"] == 2


def test_modal_share_separates_a_default_from_a_knob():
    """A parameter whose best value is the same everywhere is a default worth
    adopting; one whose best value moves per cell cannot be quoted as a single
    setting."""
    same = [arm("blur+nafnet", 0.29, {"blur_kernel": 7}),
            arm("blur+nafnet", 0.36, {"blur_kernel": 31})]
    flipped = [arm("blur+nafnet", 0.36, {"blur_kernel": 7}),
               arm("blur+nafnet", 0.29, {"blur_kernel": 31})]

    consistent = aps.summarise("blur_kernel", aps.find_sweeps(
        {OP: list(same), OP2: list(same)})["blur_kernel"])
    mixed = aps.summarise("blur_kernel", aps.find_sweeps(
        {OP: same, OP2: flipped})["blur_kernel"])

    assert consistent["axes"]["quality"]["modal_share"] == pytest.approx(1.0)
    assert mixed["axes"]["quality"]["modal_share"] == pytest.approx(0.5)


def test_an_absent_key_is_a_value_and_can_win():
    """`None` means the key is absent from the config, which is a real setting:
    no `downsample_uniform_level` is the graded mode, and it wins quality 8 of 8
    in the corpus. The first version of this tool used None to mean both "absent"
    and "no winner", and reported that as 1 of 8."""
    by_op = {OP: [arm("downsample+realesrgan", 0.15, {"downsample_levels": 3}),
                  arm("downsample+realesrgan", 0.24,
                      {"downsample_levels": 3, "downsample_uniform_level": 3})]}

    groups = aps.find_sweeps(by_op)["downsample_uniform_level"]
    summary = aps.summarise("downsample_uniform_level", groups)

    assert groups[0].best("quality") is None          # the absent-key variant
    assert summary["axes"]["quality"]["best_value_counts"] == {"None": 1}


def test_no_best_is_distinguishable_from_a_none_valued_winner():
    group = aps.find_sweeps(
        {OP: [arm("blur+nafnet", 0.29, {"blur_kernel": 7}),
              arm("blur+nafnet", 0.36, {"blur_kernel": 31})]}
    )["blur_kernel"][0]

    assert group.best("speed") is aps.NO_BEST          # no timing on either run
    assert group.best("quality") == 7
