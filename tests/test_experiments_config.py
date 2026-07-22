"""The real experiments.yaml must stay loadable and dispatchable.

This is the guard that catches a refactor renaming a component, or a hand-edit
introducing a typo, before it costs a GPU run. It reads the repo's actual file
rather than a fixture, because the file itself is what the runner consumes.
"""

import json
from pathlib import Path

import pytest
import yaml

from presley.runner import COMPONENT_RUNNERS, compute_experiment_hash

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments.yaml"


@pytest.fixture(scope="module")
def experiments():
    if not EXPERIMENTS.is_file():
        pytest.skip("experiments.yaml not present")
    parsed = yaml.safe_load(EXPERIMENTS.read_text()) or {}
    return parsed.get("experiments") or []


def test_experiments_yaml_parses(experiments):
    assert experiments, "experiments.yaml has no entries"


def test_every_entry_names_a_known_component(experiments):
    unknown = {
        e.get("component")
        for e in experiments
        if e.get("component") not in COMPONENT_RUNNERS
    }
    assert not unknown, f"experiments.yaml names components the runner cannot dispatch: {unknown}"


def test_entries_are_hashable(experiments):
    """Every entry must survive json serialization — the hash depends on it."""
    for entry in experiments:
        compute_experiment_hash(entry)


def test_no_two_different_configs_share_a_hash(experiments):
    """A real collision would make two experiments write to one results/ dir.

    The runner skips any hash that already has a result.json, so the second
    config would never run and would silently inherit the first one's numbers —
    and the paper would cite that directory for both.

    Entries that are *identical* are fine and deliberately allowed: same config,
    same result, and the skip is correct. Only differing configs matter here.
    """
    by_hash: dict[str, str] = {}
    collisions = []
    for entry in experiments:
        h = compute_experiment_hash(entry)
        canonical = json.dumps(entry, sort_keys=True)
        if h in by_hash and by_hash[h] != canonical:
            collisions.append(h)
        by_hash[h] = canonical
    assert not collisions, f"different configs sharing a hash: {collisions[:3]}"
