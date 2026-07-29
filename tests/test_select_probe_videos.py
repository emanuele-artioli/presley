"""The probe-set selector decides which videos every downstream conclusion is
screened on, so a quiet failure here biases the whole campaign.

The risks worth pinning: an attribute on a large scale (bg_texture ~96)
swamping one on a small scale (hole_churn ~0.005) if standardization breaks; a
"representative" video that is not actually a member of the cluster it
represents; and a silently-truncated attribute set if audit_videos.py changes
its columns.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "select_probe_videos.py"
_spec = importlib.util.spec_from_file_location("select_probe_videos", _TOOL)
spv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spv)


def _csv(tmp_path: Path, rows: list[dict]) -> Path:
    import csv as _csv_mod
    p = tmp_path / "attrs.csv"
    cols = ["video", "status"] + spv.ATTRS
    with p.open("w", newline="") as f:
        w = _csv_mod.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _row(name: str, value: float, status: str = "ok") -> dict:
    r = {"video": name, "status": status}
    r.update({a: value for a in spv.ATTRS})
    return r


def test_standardize_equalizes_wildly_different_scales():
    """bg_texture is ~4 orders of magnitude larger than hole_churn. Without
    z-scoring, distance would be a function of bg_texture alone."""
    x = np.array([[0.001, 100.0], [0.002, 200.0], [0.003, 300.0]])
    z = spv.standardize(x)
    assert np.allclose(z.std(axis=0), 1.0)
    # Both columns now contribute identically to distance.
    assert np.allclose(z[:, 0], z[:, 1])


def test_standardize_survives_a_constant_attribute():
    """A zero-variance column would divide by zero and poison every distance."""
    x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    z = spv.standardize(x)
    assert np.all(np.isfinite(z))
    assert np.allclose(z[:, 1], 0.0)


def test_medoid_belongs_to_the_cluster_it_represents():
    """A centroid can be a point no video occupies; a medoid must be a real
    video from that cluster, or 'run the probe on it' is not executable."""
    rng = np.random.default_rng(0)
    z = np.vstack([rng.normal(-5, 0.1, (6, 3)), rng.normal(5, 0.1, (6, 3))])
    labels, medoids = spv.pick_medoids(z, 2)

    assert len(medoids) == 2
    for m in medoids:
        assert 0 <= m < len(z)
    # The two medoids must come from different clusters.
    assert labels[medoids[0]] != labels[medoids[1]]


def test_medoid_count_never_exceeds_k_and_medoids_are_distinct():
    """scipy's maxclust yields AT MOST k clusters, so the contract is an upper
    bound, not equality. Over-reporting would be the dangerous direction: it
    would claim coverage the probe set does not have."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(5, 3))

    _, one = spv.pick_medoids(z, 1)
    assert len(one) == 1

    for k in range(1, 6):
        _, medoids = spv.pick_medoids(z, k)
        assert 1 <= len(medoids) <= k
        assert len(set(medoids)) == len(medoids)


def test_shortfall_in_cluster_count_is_reported(capsys, tmp_path, monkeypatch):
    """Asking for more probe videos than the data can distinguish must warn,
    not silently return a smaller subset -- that would overstate coverage."""
    rows = [_row(f"v{i}", 1.0) for i in range(4)]  # all coincident
    p = _csv(tmp_path, rows)
    monkeypatch.setattr(sys, "argv", ["prog", "--attributes", str(p), "-k", "4", "--db", str(tmp_path / "none.db")])
    spv.main()
    assert "smaller than requested" in capsys.readouterr().err


def test_selection_is_deterministic():
    """A probe set that changes between invocations would silently change what
    every claim was screened on."""
    rng = np.random.default_rng(2)
    z = rng.normal(size=(12, 4))
    assert spv.pick_medoids(z, 3)[1] == spv.pick_medoids(z, 3)[1]


def test_missing_attribute_column_is_a_clean_error(tmp_path):
    """If audit_videos.py drops a column, fail loudly rather than cluster on a
    silently smaller feature set."""
    import csv as _csv_mod
    p = tmp_path / "attrs.csv"
    with p.open("w", newline="") as f:
        w = _csv_mod.DictWriter(f, fieldnames=["video", "status", "fg_frac"])
        w.writeheader()
        w.writerow({"video": "bear", "status": "ok", "fg_frac": "0.1"})

    with pytest.raises(SystemExit, match="missing columns"):
        spv.load_attributes(p)


def test_failed_audit_rows_are_excluded(tmp_path):
    """A video whose audit errored has garbage attributes; clustering on it
    would place a phantom point in the space."""
    p = _csv(tmp_path, [_row("good", 1.0), _row("bad", 0.0, status="error")])
    videos, x = spv.load_attributes(p)
    assert videos == ["good"]
    assert x.shape[0] == 1
