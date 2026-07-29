#!/usr/bin/env python3
"""Build a queryable index over results/*/result.json.

694+ experiment directories, each a nested JSON of config + metrics, with no
way to ask "which experiments already exist for X" except grepping the tree by
hand. This script flattens every result.json into rows of a SQLite database —
a materialized view over files that remain the source of truth, not a second
copy of them. Delete the .db and re-run any time; it is fully regenerable and
gitignored (results/ itself is gitignored, so nothing here is committable
data).

Config keys are sparse and heterogeneous across components (baselines/roi/elvis
/presley_ai each set different keys — verified: 24 distinct config keys appear
across 694 dirs, most present in under half). Rather than force a rigid schema,
every scalar (str/int/float/bool/None) config and top-level metric value becomes
its own column, created on first sight; rows missing a column get NULL. Nested
dicts (metrics/foreground/*, metrics/background/*, metrics/overall/*,
config/codec_params/*, config/restorer_params/*, ...) are flattened with
`_`-joined paths. `metrics.block_level.*` is NOT flattened into columns (it's
{shape, path} pointers to per-run .npz files, not scalars) -- instead the raw
paths are kept in a `block_level_json` column so a caller can locate and load
them.

Usage:
    python tools/index_results.py                  # build/refresh results/index.db
    python tools/index_results.py --results-dir results_test
    python tools/index_results.py --db /tmp/foo.db --results-dir results

Query example:
    sqlite3 results/index.db "select experiment_hash, video, config_restorer, \\
        metrics_foreground_lpips_mean from results \\
        where config_component='presley_ai' and config_codec='svtav1' \\
        and invariant_failures_n=0"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keys whose values are themselves per-run artifact pointers, not scalars to
# flatten into columns. Kept verbatim as a JSON blob column instead.
_ARTIFACT_KEYS = {"block_level"}

_SQLITE_TYPES = {str: "TEXT", bool: "INTEGER", int: "INTEGER", float: "REAL"}


def _flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    """Recursively flatten nested dicts into prefix_joined scalar columns."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _ARTIFACT_KEYS and prefix in ("metrics",):
                out[f"{prefix}_{k}_json"] = json.dumps(v)
                continue
            _flatten(f"{prefix}_{k}" if prefix else k, v, out)
    elif isinstance(obj, list):
        # Only invariant_failures is a top-level list in practice; store the
        # count as a queryable column plus the raw list as JSON for detail.
        out[f"{prefix}_n"] = len(obj)
        out[f"{prefix}_json"] = json.dumps(obj)
    else:
        out[prefix] = obj


def flatten_result(hash_dir: str, data: dict) -> dict[str, Any]:
    row: dict[str, Any] = {"experiment_hash": hash_dir}
    for top_key, val in data.items():
        if top_key == "experiment_hash":
            # Already the row's primary key (from the directory name); the
            # in-file copy is redundant and would collide as a column.
            continue
        _flatten(top_key, val, row)
    return row


def _column_type(value: Any) -> str:
    for py_type, sql_type in _SQLITE_TYPES.items():
        if isinstance(value, py_type):
            return sql_type
    return "TEXT"


def build_index(results_dir: Path, db_path: Path) -> int:
    result_files = sorted(results_dir.glob("*/result.json"))
    if not result_files:
        print(f"error: no result.json files under {results_dir}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    skipped = 0
    for f in result_files:
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: skipping unreadable {f}: {e}", file=sys.stderr)
            skipped += 1
            continue
        rows.append(flatten_result(f.parent.name, data))

    if not rows:
        print("error: no readable result.json files", file=sys.stderr)
        return 1

    # Column set = union across all rows, typed from the first row that has
    # a non-null value for that column (sqlite is dynamically typed per-cell
    # anyway, so this only affects the declared column affinity).
    all_columns: dict[str, str] = {}
    for row in rows:
        for k, v in row.items():
            if k == "experiment_hash":
                continue  # already the table's declared PRIMARY KEY column
            if k not in all_columns or all_columns[k] == "TEXT":
                if v is not None:
                    all_columns[k] = _column_type(v)
            all_columns.setdefault(k, "TEXT")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cols_sql = ", ".join(f'"{c}" {t}' for c, t in sorted(all_columns.items()))
    conn.execute(f'CREATE TABLE results (experiment_hash TEXT PRIMARY KEY, {cols_sql})')

    col_names = ["experiment_hash"] + sorted(all_columns.keys())
    placeholders = ", ".join("?" for _ in col_names)
    quoted_names = ", ".join('"{}"'.format(c) for c in col_names)
    insert_sql = f'INSERT OR REPLACE INTO results ({quoted_names}) VALUES ({placeholders})'
    for row in rows:
        conn.execute(insert_sql, [row.get(c) for c in col_names])

    # A few indexes for the query patterns this project actually uses.
    for col in ("config_component", "config_video", "config_codec", "invariant_failures_n"):
        if col in all_columns:
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{col} ON results("{col}")')

    conn.commit()
    conn.close()

    print(f"indexed {len(rows)} results ({skipped} skipped) -> {db_path}")
    print(f"columns: {len(all_columns)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--db", default=None, help="output sqlite path (default: <results-dir>/index.db)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    db_path = Path(args.db) if args.db else results_dir / "index.db"
    return build_index(results_dir, db_path)


if __name__ == "__main__":
    sys.exit(main())
