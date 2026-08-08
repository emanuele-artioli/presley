"""Per-stage timing is a measurement, so its failure modes are measurement bugs.

The three that would silently corrupt the cost figure this feeds:

* a stage name that differs between components, which puts two different
  quantities on one axis and cannot be seen in the output;
* a stage recorded twice, or overlapping stages, which makes the parts sum to
  more than the whole;
* a stage that did not run reported as 0.0 rather than absent, which reads as
  "restoration was free" instead of "there was no restoration".

The GPU-level behaviour (that `restore` dominates, that selection is cheap)
belongs to the campaign, not here -- these tests are pure and run in the fast
tier.
"""

import time

import pytest

from presley.stagetiming import CACHED_STAGES, STAGES, StageTimer


def test_records_a_stage():
    timer = StageTimer()
    with timer('encode'):
        time.sleep(0.01)
    assert timer.as_dict()['encode'] >= 0.01


def test_stages_that_did_not_run_are_absent_not_zero():
    """`baselines` has no restoration step; 0.0 would misreport that as free."""
    timer = StageTimer()
    with timer('encode'):
        pass
    recorded = timer.as_dict()
    assert 'restore' not in recorded
    assert set(recorded) == {'encode'}


def test_re_entering_a_stage_accumulates():
    """Several stages run once per frame, inside a loop over the sequence."""
    timer = StageTimer()
    for _ in range(3):
        with timer('degrade'):
            time.sleep(0.005)
    assert timer.as_dict()['degrade'] >= 0.015


def test_unknown_stage_is_rejected():
    """A typo would otherwise appear as an extra stage nobody notices."""
    timer = StageTimer()
    with pytest.raises(KeyError):
        with timer('encoding'):   # the real name is 'encode'
            pass
    with pytest.raises(KeyError):
        timer.add('restoration', 1.0)


def test_output_is_in_pipeline_order():
    """The figure this feeds is a stacked bar; insertion order would scramble it."""
    timer = StageTimer()
    for stage in ('restore', 'encode', 'preprocess'):
        timer.add(stage, 1.0)
    assert list(timer.as_dict()) == ['preprocess', 'encode', 'restore']


def test_add_records_a_hand_timed_stage():
    timer = StageTimer()
    timer.add('restore', 12.5)
    assert timer.as_dict() == {'restore': 12.5}


def test_total_is_the_sum_of_the_parts():
    timer = StageTimer()
    timer.add('encode', 2.0)
    timer.add('restore', 3.0)
    assert timer.total() == pytest.approx(5.0)


def test_cached_stages_are_a_subset_of_the_vocabulary():
    """A cached stage naming something not in STAGES could never be reported."""
    assert CACHED_STAGES <= set(STAGES)


def test_every_component_uses_the_shared_vocabulary():
    """The point of a shared vocabulary is that it is actually shared.

    A component inventing its own stage name is invisible at runtime and only
    shows up as a missing bar in the figure, so it is checked structurally.
    """
    import ast
    import pathlib

    components = pathlib.Path(__file__).resolve().parent.parent / "src" / "presley" / "components"
    used = set()
    for path in components.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # stages('name') and stages.add('name', ...)
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                func = node.func
                is_call = isinstance(func, ast.Name) and func.id == 'stages'
                is_add = (isinstance(func, ast.Attribute) and func.attr == 'add'
                          and isinstance(func.value, ast.Name) and func.value.id == 'stages')
                if is_call or is_add:
                    used.add(node.args[0].value)

    assert used, "no stage timing found in any component"
    unknown = used - set(STAGES)
    assert not unknown, f"components use stage names outside STAGES: {sorted(unknown)}"
