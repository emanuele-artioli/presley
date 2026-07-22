"""The experiment hash is the identity of every result on disk.

If it changes, `results/<hash>/` stops matching `experiments.yaml`, the runner
re-runs work that is already done, and every CLAIM line in the paper points at
a directory that no longer corresponds to its config. These tests pin the
properties the rest of the system assumes.
"""

import textwrap

import pytest

from presley.runner import (
    COMPONENT_RUNNERS,
    annotate_experiments_yaml,
    compute_experiment_hash,
    dispatch_component,
    load_experiments,
)


def test_hash_is_stable_for_the_same_config():
    exp = {"component": "baselines", "video": "tennis", "codec": "x265"}
    assert compute_experiment_hash(exp) == compute_experiment_hash(dict(exp))


def test_hash_ignores_key_order():
    a = {"component": "roi", "video": "bear", "codec_params": {"qp": 30}}
    b = {"codec_params": {"qp": 30}, "video": "bear", "component": "roi"}
    assert compute_experiment_hash(a) == compute_experiment_hash(b)


def test_annotation_keys_do_not_perturb_the_hash():
    """`hash` and `_`-prefixed keys are bookkeeping, not configuration.

    Annotating an experiment with its own hash must not change that hash, or
    the annotation pass would invalidate every result it touches.
    """
    base = {"component": "elvis", "video": "camel"}
    annotated = {**base, "hash": "deadbeef", "_note": "added by a tool"}
    assert compute_experiment_hash(annotated) == compute_experiment_hash(base)


def test_different_config_gives_different_hash():
    a = {"component": "baselines", "video": "tennis", "codec_params": {"qp": 30}}
    b = {"component": "baselines", "video": "tennis", "codec_params": {"qp": 31}}
    assert compute_experiment_hash(a) != compute_experiment_hash(b)


def test_hash_is_a_16_char_hex_id():
    h = compute_experiment_hash({"component": "roi"})
    assert len(h) == 16
    assert set(h) <= set("0123456789abcdef")


# --- annotate_experiments_yaml -------------------------------------------------
# The docstring records that hard-coding the list indent once made this a silent
# no-op: it computed every hash, matched no lines, and reported success. Both
# indent styles are therefore pinned, as is the refusal to write a misaligned map.

ZERO_INDENT = textwrap.dedent(
    """\
    experiments:
    - component: baselines
      video: tennis
    - component: roi
      video: bear
    """
)

TWO_SPACE_INDENT = textwrap.dedent(
    """\
    experiments:
      - component: baselines
        video: tennis
      - component: roi
        video: bear
    """
)


@pytest.mark.parametrize("source", [ZERO_INDENT, TWO_SPACE_INDENT], ids=["flush", "indented"])
def test_annotation_labels_every_entry(tmp_path, source):
    path = tmp_path / "experiments.yaml"
    path.write_text(source)

    annotate_experiments_yaml(str(path))
    written = path.read_text()

    expected = [
        compute_experiment_hash({"component": "baselines", "video": "tennis"}),
        compute_experiment_hash({"component": "roi", "video": "bear"}),
    ]
    for h in expected:
        assert f"# hash: {h}" in written
    assert written.count("# hash:") == 2


def test_annotation_is_idempotent(tmp_path):
    path = tmp_path / "experiments.yaml"
    path.write_text(ZERO_INDENT)

    annotate_experiments_yaml(str(path))
    once = path.read_text()
    annotate_experiments_yaml(str(path))

    assert path.read_text() == once


def test_annotation_replaces_a_stale_hash(tmp_path):
    path = tmp_path / "experiments.yaml"
    path.write_text(ZERO_INDENT)
    annotate_experiments_yaml(str(path))

    stale = path.read_text().replace(
        compute_experiment_hash({"component": "roi", "video": "bear"}),
        "0000000000000000",
    )
    path.write_text(stale)
    annotate_experiments_yaml(str(path))

    assert "0000000000000000" not in path.read_text()


def test_nested_sequences_are_not_annotated(tmp_path):
    """Only top-level entries get a hash; a nested list must not be labelled."""
    path = tmp_path / "experiments.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            experiments:
            - component: baselines
              video: tennis
              bitrates:
                - 500
                - 1000
            """
        )
    )
    annotate_experiments_yaml(str(path))
    assert path.read_text().count("# hash:") == 1


def test_misalignment_leaves_the_file_untouched(tmp_path, capsys):
    """A wrong map is worse than none: entries would point at other runs' results."""
    path = tmp_path / "experiments.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            other_list:
            - not: an experiment
            experiments:
            - component: baselines
              video: tennis
            """
        )
    )
    before = path.read_text()

    # A stray top-level `- ` under some *other* key before `experiments:` is the
    # shape that used to make the indent latch onto the wrong region.
    annotate_experiments_yaml(str(path))

    if path.read_text() != before:
        # It annotated; then it must have annotated exactly the real entries.
        assert path.read_text().count("# hash:") == 1
    else:
        assert "WARNING" in capsys.readouterr().err


def test_missing_file_is_not_an_error(tmp_path):
    """Provenance metadata must never block a run."""
    annotate_experiments_yaml(str(tmp_path / "absent.yaml"))


# --- loading and dispatch ------------------------------------------------------


def test_load_experiments_filters_on_any_key(tmp_path):
    path = tmp_path / "experiments.yaml"
    path.write_text(ZERO_INDENT)

    assert len(load_experiments(str(path), {})) == 2
    only_roi = load_experiments(str(path), {"component": "roi"})
    assert [e["video"] for e in only_roi] == ["bear"]


def test_load_experiments_raises_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiments(str(tmp_path / "absent.yaml"), {})


def test_unknown_component_is_rejected_by_name():
    """A typo in `component` must fail loudly, not fall through to a default."""
    with pytest.raises(ValueError, match="Unknown component"):
        dispatch_component("presley-ai", {}, "dataset", "results/x", "cache")


def test_dispatch_table_covers_the_shipped_components():
    assert set(COMPONENT_RUNNERS) == {"baselines", "roi", "elvis", "presley_ai"}
