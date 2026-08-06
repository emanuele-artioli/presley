"""Tests for the ELVIS replay path in the timing campaign (E1).

`blackout+propainter` had no measured restoration speed at all -- not a slow
number, no number -- because the harness looked only for presley_ai's artefact
names. That is why `tab:speed-scaling` carries a structural gap rather than a
slow row.

Three things are pinned here, each of which failed silently rather than loudly
during implementation:

1. ELVIS artefacts (`encoded_shrunk.mp4` + `removal_masks.npz`) are accepted.
2. `removal_mode='shrink'` is REFUSED. Its transmitted frames are repacked into
   a smaller rectangle and need the stretch step, so replaying the decode as if
   it were the restorer's input would time the wrong operation while appearing
   to succeed. A wrong number is worse than a missing one.
3. The arm label comes from `removal_mode` when `degradation` is NULL. Every
   ELVIS row has `degradation = NULL`, so labelling from that column alone
   produces "None+propainter", which matches no requested arm and drops the
   whole component without a word.
"""
import json
import pathlib
import sqlite3
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import timing_campaign as tc  # noqa: E402


def _make_db(tmp_path, rows):
    db = tmp_path / "presley.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE runs (hash TEXT, doc TEXT, component TEXT, video TEXT,"
        " width INT, height INT, degradation TEXT, restorer TEXT,"
        " inpainter TEXT, n_invariant_failures INT, has_metrics INT)"
    )
    for r in rows:
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,0,1)", r)
    con.commit()
    con.close()
    return str(db)


def _elvis_row(h, removal_mode="blackout"):
    doc = json.dumps({"config": {"component": "elvis", "removal_mode": removal_mode,
                                 "block_size": 8, "width": 640, "height": 360}})
    return (h, doc, "elvis", "bear", 640, 360, None, None, "propainter")


def _artefacts(tmp_path, h, elvis=True):
    d = tmp_path / h
    d.mkdir()
    names = (("encoded_shrunk.mp4", "removal_masks.npz") if elvis
             else ("encoded_degraded.mp4", "strength_maps.npz"))
    for n in names:
        (d / n).write_bytes(b"x")
    return d


class TestElvisRunsAreDiscoverable:
    def test_elvis_artefacts_are_accepted(self, tmp_path):
        db = _make_db(tmp_path, [_elvis_row("aaa")])
        _artefacts(tmp_path, "aaa", elvis=True)
        got = tc.pick_configurations(db, ["blackout+propainter"], [(640, 360)])
        assert [c.hash for c in got] == ["aaa"], (
            "ELVIS runs must be replayable; excluding them is why "
            "blackout+propainter had no speed number at all"
        )

    def test_label_uses_removal_mode_when_degradation_is_null(self, tmp_path):
        db = _make_db(tmp_path, [_elvis_row("bbb")])
        _artefacts(tmp_path, "bbb", elvis=True)
        # the buggy label would have been "None+propainter"
        assert tc.pick_configurations(db, ["None+propainter"], [(640, 360)]) == []
        assert len(tc.pick_configurations(db, ["blackout+propainter"], [(640, 360)])) == 1

    def test_freeze_is_also_discoverable(self, tmp_path):
        db = _make_db(tmp_path, [_elvis_row("ccc", removal_mode="freeze")])
        _artefacts(tmp_path, "ccc", elvis=True)
        assert len(tc.pick_configurations(db, ["freeze+propainter"], [(640, 360)])) == 1

    def test_a_run_with_neither_artefact_set_is_skipped(self, tmp_path):
        db = _make_db(tmp_path, [_elvis_row("ddd")])
        (tmp_path / "ddd").mkdir()          # no artefacts at all
        assert tc.pick_configurations(db, ["blackout+propainter"], [(640, 360)]) == []


class TestShrinkIsRefusedRatherThanMisreplayed:
    def test_shrink_raises(self, tmp_path):
        d = tmp_path / "eee"
        d.mkdir()
        (d / "result.json").write_text(json.dumps({"config": {
            "component": "elvis", "removal_mode": "shrink", "inpainter": "propainter",
            "block_size": 8, "width": 640, "height": 360}}))
        with pytest.raises(ValueError, match="not replayable"):
            tc.replay_once(str(d), str(tmp_path / "out"))

    @pytest.mark.parametrize("mode", ["blackout", "freeze"])
    def test_non_shrink_modes_get_past_the_guard(self, tmp_path, mode):
        """They fail later for lack of a real video, not at the shrink guard."""
        d = tmp_path / f"f_{mode}"
        d.mkdir()
        (d / "result.json").write_text(json.dumps({"config": {
            "component": "elvis", "removal_mode": mode, "inpainter": "propainter",
            "block_size": 8, "width": 640, "height": 360}}))
        with pytest.raises(Exception) as exc:
            tc.replay_once(str(d), str(tmp_path / "out"))
        assert "not replayable" not in str(exc.value)
