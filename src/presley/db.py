"""SQLite store for experiment results — the source of truth for run metadata.

Why this exists
---------------
`results/<hash>/result.json` was the only home for run metadata: 837 documents,
3.7 MB, with sparse and heterogeneous keys across four components. Reading them
is fast (a full parse is ~0.35 s) so this is **not** a performance change. It is
a queryability change: answering "which arms lack LPIPS", "what exists for this
video at this QP", or "give me the matched pairs for this comparison" meant
writing a bespoke scan every time, and every table that reached the paper was
assembled by hand from those scans. Two numbers in one session's write-up were
transcription slips, not measurement errors.

Design
------
**The DB is authoritative; `result.json` is a derived mirror.** `results/` is
gitignored and unrecoverable, so making a single `.db` the *only* home for 837
runs' metadata would replace 837 independent failure domains with one. The mirror
costs 3.7 MB and either side rebuilds the other (`import_json_tree` /
`export_json_tree`), so the durability of per-directory files is kept without
giving up the query layer.

**Within the DB, one canonical document plus derived rows.** `runs.doc` holds the
run exactly as canonical JSON; `runs`' other columns, `metrics` and `artifacts`
are *derived* from it by `_reindex`. This is what makes round-trip fidelity
provable rather than hoped for — the derived side can be dropped and rebuilt at
any time, and a schema change never risks the data. `import → export` is asserted
byte-identical against all real results in the test suite.

**Metrics are narrow, not 155 sparse columns.** `tools/index_results.py` created a
column per metric on first sight, so querying required knowing column names.
Here `metrics(hash, region, metric, value)` makes region and metric *values you
filter on*. `region` is a dotted path, because `transmitted` nests one level
deeper than the other regions (`transmitted.foreground.psnr_mean`).

**Citability is enforced by views, not by prose.** The rules in the
`results-report` skill — a non-empty `invariant_failures` makes a run uncitable;
the bbox "foreground" metrics are not foreground metrics; bitrate is
`actual_bitrate_bps` and never `file_size_bytes` — are encoded as views here, so
a wrong query is impossible rather than merely discouraged. See `V_CITABLE`,
`V_FG_METRICS`, `V_RATE`.

**Concurrency.** WAL mode plus a busy timeout, because two runners working the
same tree is a real workflow here and the previous file-based scheme broke on it
(`research-log/bugs.md`, the `result.json.tmp` race).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional

SCHEMA_VERSION = 1
DB_FILENAME = "presley.db"

# Metrics that must never be read as a foreground signal. These are union-bbox
# artifacts (100% of frame on some videos vs a measured 4.0% true FG), and the
# rule that bans them lives in the results-report skill as prose. Encoding it
# here means `v_fg_metrics` cannot return them at all.
BANNED_FG_METRICS = ("vmaf_fg_bbox", "vmaf_neg_fg_bbox", "fvmd", "fid_fg_bbox")
# The only true foreground region metrics (spatial UFO-mask based).
TRUE_FG_METRICS = ("lpips_mean", "dists_fg", "psnr_mean", "ssim_mean", "mse_mean")

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
    hash                TEXT PRIMARY KEY,
    doc                 TEXT NOT NULL,           -- canonical JSON, the truth
    component           TEXT,
    video               TEXT,
    dataset             TEXT,                    -- derived: 'davis' or the prefix before '/'
    width               INTEGER,
    height              INTEGER,
    codec               TEXT,
    qp                  INTEGER,
    rate_control        TEXT,
    degradation         TEXT,
    restorer            TEXT,
    inpainter           TEXT,
    block_size          INTEGER,
    actual_bitrate_bps  REAL,
    transmitted_size_bytes INTEGER,
    video_frames        INTEGER,
    total_time_seconds  REAL,
    n_invariant_failures INTEGER NOT NULL DEFAULT 0,
    invariant_failures  TEXT,                    -- JSON list
    has_metrics         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_runs_video     ON runs(video);
CREATE INDEX IF NOT EXISTS ix_runs_component ON runs(component);
CREATE INDEX IF NOT EXISTS ix_runs_dataset   ON runs(dataset);
CREATE INDEX IF NOT EXISTS ix_runs_codec_qp  ON runs(codec, qp);

CREATE TABLE IF NOT EXISTS metrics (
    hash   TEXT NOT NULL REFERENCES runs(hash) ON DELETE CASCADE,
    region TEXT NOT NULL,      -- 'foreground' | 'background' | 'overall' | 'transmitted.foreground' | ...
    metric TEXT NOT NULL,      -- 'psnr_mean' | 'lpips_mean' | 'dists_fg' | ...
    value  REAL,
    PRIMARY KEY (hash, region, metric)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_metrics_lookup ON metrics(region, metric);

CREATE TABLE IF NOT EXISTS artifacts (
    hash    TEXT NOT NULL REFERENCES runs(hash) ON DELETE CASCADE,
    kind    TEXT NOT NULL,     -- 'block_psnr' | 'output_video' | ...
    relpath TEXT NOT NULL,
    shape   TEXT,              -- JSON list, when known
    PRIMARY KEY (hash, kind)
) WITHOUT ROWID;
"""

# A run is citable only with an empty invariant_failures AND actual metrics.
# `invariant_failures` absent entirely means unevaluated, not clean — which is
# why n_invariant_failures is NOT NULL DEFAULT 0 but has_metrics gates too.
V_CITABLE = """
CREATE VIEW IF NOT EXISTS v_citable AS
SELECT * FROM runs WHERE n_invariant_failures = 0 AND has_metrics = 1;
"""

V_FG_METRICS = """
CREATE VIEW IF NOT EXISTS v_fg_metrics AS
SELECT m.hash, m.metric, m.value
FROM metrics m
JOIN v_citable r ON r.hash = m.hash
WHERE m.region = 'foreground'
  AND m.metric NOT IN ({banned});
""".format(banned=", ".join(f"'{m}'" for m in BANNED_FG_METRICS))

V_RATE = """
CREATE VIEW IF NOT EXISTS v_rate AS
SELECT hash, video, dataset, component, codec, qp, actual_bitrate_bps
FROM v_citable;
"""


def db_path(results_dir: str) -> str:
    return os.path.join(results_dir, DB_FILENAME)


def _ensure_wal(conn: sqlite3.Connection, attempts: int = 60) -> None:
    """Put the database in WAL mode, tolerating a concurrent connection doing
    the same.

    `PRAGMA journal_mode=WAL` needs a brief exclusive lock, and SQLite does
    **not** route that acquisition through the busy handler -- so it raises
    "database is locked" immediately rather than waiting out `busy_timeout`,
    however generous that is. Two tree sweeps opening a connection per run hit
    this constantly on a fresh DB.

    Two things make the retry safe. The mode is a persistent property of the
    file, so the check is a no-op on every connection after the first. And a
    loser of the race only has to wait for the winner: once anyone succeeds,
    the read below sees `wal` and returns.
    """
    for attempt in range(attempts):
        if (conn.execute("PRAGMA journal_mode").fetchone()[0] or "").lower() == "wal":
            return
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


def connect(results_dir: str = "results", *, read_only: bool = False) -> sqlite3.Connection:
    """Open (creating if needed) the results DB with concurrency-safe pragmas."""
    path = db_path(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # busy_timeout FIRST. `PRAGMA journal_mode=WAL` needs a brief exclusive lock,
    # and a pragma issued before the busy handler is installed fails outright
    # with "database is locked" instead of waiting — which is exactly how two
    # concurrent tree sweeps broke once the readers started opening a connection
    # per run.
    conn.execute("PRAGMA busy_timeout=30000")
    _ensure_wal(conn)
    conn.execute("PRAGMA foreign_keys=ON")
    if not read_only:
        init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    # Fast path: the DDL is idempotent but still takes a write lock, and every
    # connect() would otherwise pay it and contend with a concurrent writer.
    try:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is not None and row["version"] == SCHEMA_VERSION:
            return
    except sqlite3.OperationalError:
        pass                                    # no schema yet — fall through and create it
    conn.executescript(SCHEMA)
    conn.executescript(V_CITABLE + V_FG_METRICS + V_RATE)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    elif row["version"] != SCHEMA_VERSION:
        raise RuntimeError(
            f"results DB is schema v{row['version']} but this code expects "
            f"v{SCHEMA_VERSION}; migrate explicitly rather than letting the two diverge")
    conn.commit()


def _canonical(result: Dict[str, Any]) -> str:
    """One byte-stable serialization, so round-trip equality is well defined."""
    return json.dumps(result, sort_keys=True, indent=2)


def _dataset_of(video: Optional[str]) -> Optional[str]:
    if not video:
        return None
    return video.split("/", 1)[0] if "/" in video else "davis"


def _flatten_metrics(node: Any, prefix: str = "") -> Iterator[tuple]:
    """Yield (region, metric, value) for every numeric leaf under metrics.

    Regions are dotted because `transmitted` nests one level deeper than the
    other regions. `block_level` is skipped — it holds artifact pointers, not
    measurements, and is indexed into `artifacts` instead.
    """
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key == "block_level":
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            # A dict whose values are all scalars is a region; otherwise recurse.
            if any(isinstance(v, dict) for v in value.values()):
                yield from _flatten_metrics(value, path + ".")
            else:
                for metric, v in value.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        yield (path, metric, float(v))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # A scalar directly under metrics (rare); file it under region ''.
            yield ("", path, float(value))


def _reindex(conn: sqlite3.Connection, run_hash: str, result: Dict[str, Any]) -> None:
    """Rebuild every derived row for one run from its canonical document."""
    conn.execute("DELETE FROM metrics WHERE hash = ?", (run_hash,))
    conn.execute("DELETE FROM artifacts WHERE hash = ?", (run_hash,))

    metrics = result.get("metrics") or {}
    rows = [(run_hash, region, metric, value)
            for region, metric, value in _flatten_metrics(metrics)]
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO metrics(hash, region, metric, value) VALUES (?,?,?,?)", rows)

    arts = []
    for kind, spec in (metrics.get("block_level") or {}).items():
        if isinstance(spec, dict) and spec.get("path"):
            arts.append((run_hash, f"block_{kind}", spec["path"],
                         json.dumps(spec.get("shape")) if spec.get("shape") else None))
    for kind in ("output_video", "transmitted_video"):
        if result.get(kind):
            arts.append((run_hash, kind, result[kind], None))
    if arts:
        conn.executemany(
            "INSERT OR REPLACE INTO artifacts(hash, kind, relpath, shape) VALUES (?,?,?,?)", arts)


def upsert_run(conn: sqlite3.Connection, result: Dict[str, Any],
               run_hash: Optional[str] = None, *, commit: bool = True) -> str:
    """Write one run. The document is stored verbatim; everything else derives."""
    run_hash = run_hash or result.get("experiment_hash")
    if not run_hash:
        raise ValueError("cannot store a run without an experiment_hash")
    cfg = result.get("config") or {}
    failures = result.get("invariant_failures")
    codec_params = cfg.get("codec_params") or {}

    conn.execute("""
        INSERT INTO runs (hash, doc, component, video, dataset, width, height, codec, qp,
                          rate_control, degradation, restorer, inpainter, block_size,
                          actual_bitrate_bps, transmitted_size_bytes, video_frames,
                          total_time_seconds, n_invariant_failures, invariant_failures,
                          has_metrics)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(hash) DO UPDATE SET
            doc=excluded.doc, component=excluded.component, video=excluded.video,
            dataset=excluded.dataset, width=excluded.width, height=excluded.height,
            codec=excluded.codec, qp=excluded.qp, rate_control=excluded.rate_control,
            degradation=excluded.degradation, restorer=excluded.restorer,
            inpainter=excluded.inpainter, block_size=excluded.block_size,
            actual_bitrate_bps=excluded.actual_bitrate_bps,
            transmitted_size_bytes=excluded.transmitted_size_bytes,
            video_frames=excluded.video_frames, total_time_seconds=excluded.total_time_seconds,
            n_invariant_failures=excluded.n_invariant_failures,
            invariant_failures=excluded.invariant_failures, has_metrics=excluded.has_metrics
    """, (
        run_hash, _canonical(result), cfg.get("component"), cfg.get("video"),
        _dataset_of(cfg.get("video")), cfg.get("width"), cfg.get("height"),
        cfg.get("codec"), codec_params.get("qp"), result.get("rate_control"),
        cfg.get("degradation"), cfg.get("restorer"), cfg.get("inpainter"),
        cfg.get("block_size"), result.get("actual_bitrate_bps"),
        result.get("transmitted_size_bytes"), result.get("video_frames"),
        result.get("total_time_seconds"),
        len(failures) if isinstance(failures, list) else 0,
        json.dumps(failures) if failures is not None else None,
        1 if result.get("metrics") else 0,
    ))
    _reindex(conn, run_hash, result)
    if commit:
        conn.commit()
    return run_hash


def has_run(conn: sqlite3.Connection, run_hash: str) -> bool:
    return conn.execute("SELECT 1 FROM runs WHERE hash = ?", (run_hash,)).fetchone() is not None


def get_run(conn: sqlite3.Connection, run_hash: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT doc FROM runs WHERE hash = ?", (run_hash,)).fetchone()
    return json.loads(row["doc"]) if row else None


def iter_runs(conn: sqlite3.Connection, where: str = "", params: Iterable = ()) -> List[Dict[str, Any]]:
    sql = "SELECT doc FROM runs" + (f" WHERE {where}" if where else "") + " ORDER BY hash"
    return [json.loads(r["doc"]) for r in conn.execute(sql, tuple(params))]


def db_mtime(results_dir: str) -> float:
    """Newest mtime across the DB and its WAL sidecar.

    Under WAL a commit lands in `-wal` first, so the main file's mtime can lag
    behind the data. Taking the max keeps this conservative in the safe
    direction: a lagging estimate causes a redundant re-import, never a missed
    one. Returns 0.0 when no DB exists yet.
    """
    newest = 0.0
    for suffix in ("", "-wal"):
        try:
            newest = max(newest, os.path.getmtime(db_path(results_dir) + suffix))
        except OSError:
            pass
    return newest


def load_run(results_dir: str, run_hash: str) -> Optional[Dict[str, Any]]:
    """Read one run, DB first, falling back to (and healing from) the mirror.

    Every read-modify-write site goes through this so the document being
    modified is the authoritative one. The mirror fallback matters for a run
    written before the DB existed, or by a tool that has not been migrated: a
    reader must never silently see an *older* document than the file on disk.
    """
    conn = connect(results_dir)
    try:
        doc = get_run(conn, run_hash)
        path = os.path.join(results_dir, run_hash, "result.json")
        if doc is not None and not (os.path.isfile(path)
                                    and os.path.getmtime(path) > db_mtime(results_dir)):
            return doc
        if not os.path.isfile(path):
            return doc
        with open(path) as fh:                       # mirror is ahead — adopt and heal
            doc = json.load(fh)
        upsert_run(conn, doc, doc.get("experiment_hash") or run_hash)
        return doc
    finally:
        conn.close()


def save_run(results_dir: str, run_hash: str, result: Dict[str, Any]) -> None:
    """Write one run to both stores: DB (truth) and result.json (mirror).

    Mirror first, DB second, matching the ordering argument at the evaluation
    write site: a crash between the two leaves the mirror ahead, which
    `load_run`/`import_json_tree` reconcile, whereas the reverse would leave
    the DB claiming metrics no file records.
    """
    write_mirror(result, os.path.join(results_dir, run_hash, "result.json"))
    conn = connect(results_dir)
    try:
        upsert_run(conn, result, result.get("experiment_hash") or run_hash)
    finally:
        conn.close()


def import_json_tree(conn: sqlite3.Connection, results_dir: str = "results") -> Dict[str, int]:
    """Load every results/<hash>/result.json into the DB. Idempotent."""
    imported = skipped = 0
    for entry in sorted(os.listdir(results_dir)):
        if entry.startswith("_") or entry.startswith("."):
            continue
        path = os.path.join(results_dir, entry, "result.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                result = json.load(fh)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        upsert_run(conn, result, result.get("experiment_hash") or entry, commit=False)
        imported += 1
    conn.commit()
    return {"imported": imported, "skipped": skipped}


def export_json_tree(conn: sqlite3.Connection, results_dir: str = "results") -> int:
    """Write the derived result.json mirror for every run. Inverse of import."""
    written = 0
    for row in conn.execute("SELECT hash, doc FROM runs"):
        d = os.path.join(results_dir, row["hash"])
        os.makedirs(d, exist_ok=True)
        write_mirror(json.loads(row["doc"]), os.path.join(d, "result.json"))
        written += 1
    return written


def write_mirror(result: Dict[str, Any], path: str) -> None:
    """Atomically write one result.json mirror.

    Uses a unique temp name in the target directory: a deterministic `<path>.tmp`
    is atomic against a crash but NOT against a second process doing the same
    thing, which is exactly the defect logged in research-log/bugs.md.
    """
    import tempfile
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".result-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(result, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
