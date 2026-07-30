"""The results DB is the source of truth for run metadata — these tests gate that.

Two properties carry everything else. **Round-trip fidelity**: a run must come
back out of the DB exactly as it went in, or promoting the DB above result.json
silently loses data that cannot be recovered (`results/` is gitignored). And
**citability enforcement**: the rules that decide whether a number may reach the
paper used to live as prose in a skill file, where they were followed by hand and
occasionally not. They are views now, so these tests check that a wrong query is
impossible rather than merely discouraged.

Deliberately not tested: SQLite's own durability, and the exact float formatting
of the JSON mirror (the round-trip assertion covers what matters). Nothing here
touches the real results/ tree — every test builds its own under tmp_path.
"""
import argparse
import json
import multiprocessing
import os

import pytest

from presley import db


def _result(h="abc123", video="bear", failures=None, metrics=True, **cfg):
    doc = {
        "experiment_hash": h,
        "video_frames": 82,
        "actual_bitrate_bps": 535733.8,
        "transmitted_size_bytes": 228803,
        "rate_control": "cqp",
        "output_video": "restored.mp4",
        "invariant_failures": [] if failures is None else failures,
        "config": {"component": "presley_ai", "video": video, "width": 640, "height": 360,
                   "codec": "svtav1", "codec_params": {"qp": 43, "preset": "8"},
                   "block_size": 16, "degradation": "downsample", "restorer": "realesrgan",
                   **cfg},
    }
    if metrics:
        doc["metrics"] = {
            "foreground": {"psnr_mean": 30.5, "lpips_mean": 0.11, "dists_fg": 0.04,
                           "fid_fg_bbox": 12.0},
            "background": {"psnr_mean": 28.1, "lpips_mean": 0.22, "dists_bg": 0.07},
            "overall": {"psnr_mean": 29.0, "vmaf_mean": 71.2},
            "transmitted": {"foreground": {"psnr_mean": 31.0},
                            "background": {"psnr_mean": 25.0}},
            "block_level": {"psnr": {"shape": [82, 22, 40], "path": "block_psnr.npz"}},
        }
    return doc


def test_round_trip_is_exact(tmp_path):
    """The gate for using the DB as source of truth: nothing may be lost."""
    conn = db.connect(str(tmp_path))
    doc = _result()
    db.upsert_run(conn, doc)

    assert db.get_run(conn, "abc123") == doc


def test_round_trip_survives_a_json_tree_import_export(tmp_path):
    """import -> export must reproduce the mirror, so either side can rebuild."""
    results = tmp_path / "results"
    for i, video in enumerate(["bear", "camel", "mosev2/8i1uo3x9"]):
        h = f"hash{i}"
        (results / h).mkdir(parents=True)
        (results / h / "result.json").write_text(json.dumps(_result(h=h, video=video), indent=2))
    originals = {p.parent.name: json.loads(p.read_text())
                 for p in results.glob("*/result.json")}

    conn = db.connect(str(results))
    assert db.import_json_tree(conn, str(results))["imported"] == 3
    db.export_json_tree(conn, str(results))

    after = {p.parent.name: json.loads(p.read_text()) for p in results.glob("*/result.json")}
    assert after == originals


def test_a_run_with_invariant_failures_is_not_citable(tmp_path):
    """A non-empty invariant_failures makes a run uncitable — the view enforces it."""
    conn = db.connect(str(tmp_path))
    db.upsert_run(conn, _result(h="clean"))
    db.upsert_run(conn, _result(h="unsound", failures=["vbr rate control"]))

    citable = {r["hash"] for r in conn.execute("SELECT hash FROM v_citable")}
    assert citable == {"clean"}


def test_an_unevaluated_run_is_not_citable(tmp_path):
    """No metrics at all means unevaluated, which is not the same as clean."""
    conn = db.connect(str(tmp_path))
    db.upsert_run(conn, _result(h="nometrics", metrics=False))

    assert conn.execute("SELECT count(*) FROM v_citable").fetchone()[0] == 0


@pytest.mark.parametrize("banned", db.BANNED_FG_METRICS)
def test_banned_bbox_metrics_are_unreachable_as_foreground(tmp_path, banned):
    """These are union-bbox artifacts, not foreground signals.

    The results-report skill bans them in prose. If the view could still return
    one, that prose is the only thing standing between a bbox artifact and a
    foreground claim in the paper.
    """
    conn = db.connect(str(tmp_path))
    doc = _result()
    doc["metrics"]["foreground"][banned] = 42.0
    db.upsert_run(conn, doc)

    rows = conn.execute("SELECT metric FROM v_fg_metrics WHERE metric = ?", (banned,)).fetchall()
    assert rows == []
    # ...but it is still stored, so it remains auditable rather than deleted.
    assert conn.execute(
        "SELECT value FROM metrics WHERE region='foreground' AND metric=?",
        (banned,)).fetchone()["value"] == 42.0


def test_true_foreground_metrics_are_reachable(tmp_path):
    """The ban must not be so broad that the real FG metrics disappear too."""
    conn = db.connect(str(tmp_path))
    db.upsert_run(conn, _result())

    got = {r["metric"] for r in conn.execute("SELECT metric FROM v_fg_metrics")}
    assert {"lpips_mean", "dists_fg", "psnr_mean"} <= got


def test_nested_transmitted_region_keeps_its_path(tmp_path):
    """`transmitted` nests one level deeper; flattening must not collide with
    the top-level regions of the same name."""
    conn = db.connect(str(tmp_path))
    db.upsert_run(conn, _result())

    regions = {r["region"] for r in conn.execute("SELECT DISTINCT region FROM metrics")}
    assert "transmitted.foreground" in regions
    fg = conn.execute("SELECT value FROM metrics WHERE region='foreground' AND metric='psnr_mean'").fetchone()
    tx = conn.execute("SELECT value FROM metrics WHERE region='transmitted.foreground' AND metric='psnr_mean'").fetchone()
    assert (fg["value"], tx["value"]) == (30.5, 31.0)


def test_block_level_becomes_an_artifact_not_a_metric(tmp_path):
    """Block arrays live on disk; the DB indexes them and must not try to hold them."""
    conn = db.connect(str(tmp_path))
    db.upsert_run(conn, _result())

    assert conn.execute("SELECT count(*) FROM metrics WHERE region LIKE 'block_level%'").fetchone()[0] == 0
    art = conn.execute("SELECT relpath, shape FROM artifacts WHERE kind='block_psnr'").fetchone()
    assert art["relpath"] == "block_psnr.npz"
    assert json.loads(art["shape"]) == [82, 22, 40]


def test_dataset_is_derived_from_the_video_id(tmp_path):
    """Non-DAVIS clips carry their dataset as a path prefix; queries need it split."""
    conn = db.connect(str(tmp_path))
    db.upsert_run(conn, _result(h="a", video="bear"))
    db.upsert_run(conn, _result(h="b", video="mosev2/8i1uo3x9"))
    db.upsert_run(conn, _result(h="c", video="youtube_vos/0e4068b53f"))

    got = dict(conn.execute("SELECT hash, dataset FROM runs").fetchall())
    assert got == {"a": "davis", "b": "mosev2", "c": "youtube_vos"}


def test_upsert_replaces_derived_rows_rather_than_accumulating(tmp_path):
    """Re-running evaluation must not leave stale metric rows behind."""
    conn = db.connect(str(tmp_path))
    db.upsert_run(conn, _result())
    before = conn.execute("SELECT count(*) FROM metrics WHERE hash='abc123'").fetchone()[0]

    doc = _result()
    doc["metrics"]["background"] = {"psnr_mean": 99.0}   # fewer metrics than before
    db.upsert_run(conn, doc)

    after = conn.execute("SELECT count(*) FROM metrics WHERE hash='abc123'").fetchone()[0]
    assert after < before
    assert conn.execute(
        "SELECT value FROM metrics WHERE hash='abc123' AND region='background' AND metric='psnr_mean'"
    ).fetchone()["value"] == 99.0
    assert db.get_run(conn, "abc123")["metrics"]["background"] == {"psnr_mean": 99.0}


def test_has_run_is_the_already_ran_check(tmp_path):
    conn = db.connect(str(tmp_path))
    assert not db.has_run(conn, "abc123")
    db.upsert_run(conn, _result())
    assert db.has_run(conn, "abc123")


def _writer(results_dir, lo, hi):
    conn = db.connect(results_dir)
    for i in range(lo, hi):
        db.upsert_run(conn, _result(h=f"h{i}", video=f"v{i}"))
    conn.close()


def test_two_concurrent_writers_both_succeed(tmp_path):
    """Two runners working one tree is a real workflow here, and the previous
    file-based scheme died on it (research-log/bugs.md, the result.json.tmp
    race). WAL plus a busy timeout is what replaces that temp-file juggling."""
    d = str(tmp_path)
    db.connect(d).close()          # create schema before forking
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=(d, 0, 25)),
             ctx.Process(target=_writer, args=(d, 25, 50))]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    assert [p.exitcode for p in procs] == [0, 0]
    conn = db.connect(d)
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 50


def test_write_mirror_uses_a_unique_temp_name(tmp_path):
    """The bug this replaces: a deterministic <path>.tmp is atomic against a
    crash but not against a second process writing the same target."""
    target = tmp_path / "result.json"
    db.write_mirror({"experiment_hash": "x"}, str(target))

    assert json.loads(target.read_text()) == {"experiment_hash": "x"}
    assert not list(tmp_path.glob("*.tmp")), "temp file leaked"
    assert not (tmp_path / "result.json.tmp").exists()


def test_schema_version_mismatch_is_loud(tmp_path):
    """A silently-diverged schema is how a query starts returning wrong rows."""
    conn = db.connect(str(tmp_path))
    conn.execute("UPDATE schema_version SET version = 999")
    conn.commit()

    with pytest.raises(RuntimeError, match="schema v999"):
        db.init_schema(conn)


# --- the query tool's ambiguity guard -----------------------------------------

def test_compare_refuses_an_ambiguous_pairing(tmp_path, monkeypatch):
    """A --where matching two runs for one video must fail, not emit duplicates.

    This is not hypothetical: the first real invocation of `compare` silently
    paired a 640x360 arm against a 1280x720 run of the same video because the
    filter did not constrain resolution, producing n=9 with `bear` twice and a
    -57% rate delta. A table generator that can duplicate a key is worse than
    no generator, because the output looks authoritative.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qr", os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "query_results.py"))
    qr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qr)

    conn = db.connect(str(tmp_path))
    db.upsert_run(conn, _result(h="small", video="bear", width=640))
    db.upsert_run(conn, _result(h="large", video="bear", width=1280))
    db.upsert_run(conn, _result(h="armb", video="bear", downsample_level_map="cache/x"))

    args = argparse.Namespace(
        a_where="json_extract(doc,'$.config.downsample_level_map') IS NULL",
        b_where="json_extract(doc,'$.config.downsample_level_map') IS NOT NULL",
        metric="lpips_mean", region="background", pair_by="video")

    with pytest.raises(SystemExit, match="multiple runs for the same video"):
        qr.cmd_compare(conn, args)

    # Narrowed to one run per side, it works.
    args.a_where += " AND width = 640"
    rows = qr.cmd_compare(conn, args)
    assert len(rows) == 1
