"""The results index is a materialized view; if it lies, every query built on
it lies silently.

These tests pin the properties downstream analysis assumes: that heterogeneous
configs across components don't drop columns, that a run's identity comes from
its directory, that non-scalar artifact pointers survive as loadable paths
rather than being flattened into nonsense, and that a corrupt result.json
degrades to a skipped row instead of taking the whole index down.
"""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "index_results.py"
_spec = importlib.util.spec_from_file_location("index_results", _TOOL)
index_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(index_results)


def _write(results_dir: Path, hash_name: str, data: dict) -> None:
    d = results_dir / hash_name
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(data))


def _rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn.execute("select * from results").fetchall()


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "results"


def test_heterogeneous_configs_keep_all_columns(results_dir, tmp_path):
    """baselines/elvis/presley_ai set disjoint config keys. A schema derived
    from only the first row would silently drop the others' columns."""
    _write(results_dir, "a" * 16, {"config": {"component": "baselines", "video": "bear"}})
    _write(results_dir, "b" * 16, {"config": {"component": "presley_ai", "restorer": "realesrgan"}})
    _write(results_dir, "c" * 16, {"config": {"component": "elvis", "inpainter": "propainter"}})

    db = tmp_path / "i.db"
    assert index_results.build_index(results_dir, db) == 0

    rows = {r["experiment_hash"]: r for r in _rows(db)}
    assert rows["b" * 16]["config_restorer"] == "realesrgan"
    assert rows["c" * 16]["config_inpainter"] == "propainter"
    # A row that never set a key gets NULL, not a missing column.
    assert rows["a" * 16]["config_restorer"] is None


def test_directory_name_wins_over_in_file_hash(results_dir, tmp_path):
    """result.json carries its own experiment_hash. The directory is the real
    identity on disk, so a stale in-file copy must not become the key."""
    _write(results_dir, "d" * 16, {"experiment_hash": "STALE", "config": {"video": "bear"}})

    db = tmp_path / "i.db"
    assert index_results.build_index(results_dir, db) == 0

    rows = _rows(db)
    assert len(rows) == 1
    assert rows[0]["experiment_hash"] == "d" * 16


def test_block_level_pointers_survive_as_json(results_dir, tmp_path):
    """metrics.block_level is {shape, path} pointers to .npz files -- the whole
    substrate for the damage-mining work. Flattening it into scalar columns
    would lose the path/shape pairing."""
    block = {"psnr": {"shape": [82, 135, 240], "path": "block_psnr.npz"}}
    _write(results_dir, "e" * 16, {"metrics": {"block_level": block, "overall": {"psnr_mean": 30.0}}})

    db = tmp_path / "i.db"
    assert index_results.build_index(results_dir, db) == 0

    row = _rows(db)[0]
    assert json.loads(row["metrics_block_level_json"]) == block
    assert row["metrics_overall_psnr_mean"] == 30.0


def test_invariant_failures_are_queryable_as_count_and_detail(results_dir, tmp_path):
    """A run with non-empty invariant_failures is never citable, so filtering
    on it must be a cheap column, with the reasons still recoverable."""
    _write(results_dir, "f" * 16, {"invariant_failures": []})
    _write(results_dir, "0" * 16, {"invariant_failures": ["fg_psnr_regression", "bitrate"]})

    db = tmp_path / "i.db"
    assert index_results.build_index(results_dir, db) == 0

    rows = {r["experiment_hash"]: r for r in _rows(db)}
    assert rows["f" * 16]["invariant_failures_n"] == 0
    assert rows["0" * 16]["invariant_failures_n"] == 2
    assert json.loads(rows["0" * 16]["invariant_failures_json"]) == ["fg_psnr_regression", "bitrate"]


def test_corrupt_result_json_is_skipped_not_fatal(results_dir, tmp_path):
    """One truncated file (an interrupted run) must not cost the whole index."""
    _write(results_dir, "1" * 16, {"config": {"video": "bear"}})
    bad = results_dir / ("2" * 16)
    bad.mkdir(parents=True)
    (bad / "result.json").write_text("{not valid json")

    db = tmp_path / "i.db"
    assert index_results.build_index(results_dir, db) == 0

    rows = _rows(db)
    assert [r["experiment_hash"] for r in rows] == ["1" * 16]


def test_empty_results_dir_is_an_error_not_an_empty_index(results_dir, tmp_path):
    """Silently producing an empty index would make every downstream query
    return 'no such experiment' instead of surfacing the wrong path."""
    results_dir.mkdir(parents=True)
    assert index_results.build_index(results_dir, tmp_path / "i.db") == 1


def test_rebuild_is_idempotent(results_dir, tmp_path):
    """The index is regenerable by contract; a refresh must not accumulate
    duplicate rows from a previous build."""
    _write(results_dir, "3" * 16, {"config": {"video": "camel"}})
    db = tmp_path / "i.db"

    assert index_results.build_index(results_dir, db) == 0
    assert index_results.build_index(results_dir, db) == 0

    assert len(_rows(db)) == 1
