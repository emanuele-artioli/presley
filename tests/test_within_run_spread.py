"""The within-run spread tool's two ways of being silently wrong.

The tool exists because a pooled dispersion was published as a within-run one.
It would be a poor failsafe if it could make the same class of mistake itself:
double-counting a run (which pools without saying so) or losing one (which
changes the mean nobody would check).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "analyze_within_run_spread.py"
_spec = importlib.util.spec_from_file_location("analyze_within_run_spread", _TOOL)
awr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(awr)


def test_the_probe_table_has_no_duplicate_run():
    """One hash in two levels would double-count it and quietly pool the levels."""
    k2, k3 = set(awr.PROBES[2]), set(awr.PROBES[3])
    assert not (k2 & k3)
    assert len(k2) == len(k3) == 8


def test_every_probe_appears_once_per_level():
    """Eight distinct videos per level — a repeat would weight one video double."""
    for level in (2, 3):
        videos = list(awr.PROBES[level].values())
        assert sorted(videos) == sorted(set(videos))
        assert len(videos) == 8


def test_spread_is_the_p90_p10_range():
    assert awr.spread(np.arange(0.0, 101.0)) == pytest.approx(80.0)


def test_pooling_inflates_a_spread_when_groups_differ():
    """The defect this tool exists to measure, in miniature.

    Two runs with identical internal dispersion but different centres pool to a
    wider spread than either has — which is exactly how a between-group
    difference gets read as within-run headroom.
    """
    a = np.linspace(0.0, 10.0, 500)
    b = a + 20.0

    within = np.mean([awr.spread(a), awr.spread(b)])
    pooled = awr.spread(np.concatenate([a, b]))

    assert pooled > within * 2


def test_pooling_is_harmless_when_groups_coincide():
    """And the converse, which is why the S1b figure was only mildly wrong.

    Pooling is not a fixed penalty: when the groups sit on top of each other it
    costs almost nothing, so "it was pooled" is not by itself proof a number is
    badly inflated. It has to be measured.
    """
    a = np.linspace(0.0, 10.0, 500)
    b = np.linspace(0.0, 10.0, 500)

    within = np.mean([awr.spread(a), awr.spread(b)])
    pooled = awr.spread(np.concatenate([a, b]))

    assert pooled == pytest.approx(within, rel=0.01)
