#!/usr/bin/env python3
"""Query the results DB and emit tables — including paper-ready LaTeX + CLAIM lines.

The point of this tool is that a table in the paper and its provenance comment
should come from the same query, not from a human reading numbers off one screen
and typing them into another. Two figures in the 2026-07-30 write-up were
transcription slips (0.09 vs 0.095; a mean quoted from two different aggregations
of the same quantity). `--format latex --claim` removes that class of error: the
tabular body and the `CLAIM(...)`/`src(...)` block are generated together from
the rows that were actually measured.

Every query runs against the citability views by default (`v_citable`), so a run
with a non-empty `invariant_failures` cannot reach a table by accident. Pass
`--include-uncitable` to see them, which prints a warning naming each one.

Examples
--------
  # what exists for a video
  tools/query_results.py exists --video bear

  # which runs are missing a perceptual metric (the usual pre-report question)
  tools/query_results.py missing-metric --metric lpips_mean --region background

  # a paired comparison, straight to LaTeX with its provenance block
  tools/query_results.py compare --a-where "config_downsample_level_map IS NULL" \
      --b-where "config_downsample_level_map IS NOT NULL" \
      --region background --metric lpips_mean --format latex --claim

  # anything else
  tools/query_results.py sql "SELECT video, qp, actual_bitrate_bps FROM v_rate LIMIT 5"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from presley import db  # noqa: E402


# --------------------------------------------------------------------------
# freshness — a stale index is worse than no index, because it answers wrongly
# --------------------------------------------------------------------------

def ensure_fresh(conn, results_dir: str, *, auto: bool = True) -> None:
    """Import any result.json the DB has not seen.

    The predecessor (`tools/index_results.py`) was a one-shot build that nobody
    re-ran, and it was found 104 runs behind the tree. Silently answering from a
    stale index is the failure mode this guards.
    """
    on_disk = {e for e in os.listdir(results_dir)
               if not e.startswith((".", "_"))
               and os.path.isfile(os.path.join(results_dir, e, "result.json"))}
    known = {r["hash"] for r in conn.execute("SELECT hash FROM runs")}
    missing = on_disk - known
    if not missing:
        return
    if not auto:
        print(f"warning: {len(missing)} run(s) on disk are not in the DB", file=sys.stderr)
        return
    print(f"(indexing {len(missing)} new run(s))", file=sys.stderr)
    for h in sorted(missing):
        try:
            with open(os.path.join(results_dir, h, "result.json")) as fh:
                db.upsert_run(conn, json.load(fh), h, commit=False)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {h}: {exc}", file=sys.stderr)
    conn.commit()


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def _fmt_value(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return "" if v is None else str(v)


def render_table(rows, cols) -> str:
    if not rows:
        return "(no rows)"
    data = [[_fmt_value(r[c]) for c in cols] for r in rows]
    w = [max(len(c), *(len(d[i]) for d in data)) for i, c in enumerate(cols)]
    out = ["  ".join(c.ljust(w[i]) for i, c in enumerate(cols)),
           "  ".join("-" * w[i] for i in range(len(cols)))]
    out += ["  ".join(d[i].ljust(w[i]) for i in range(len(cols))) for d in data]
    return "\n".join(out)


def render_csv(rows, cols) -> str:
    import csv
    import io
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(cols)
    for r in rows:
        wr.writerow([r[c] for c in cols])
    return buf.getvalue().rstrip("\n")


def render_latex(rows, cols, caption: str = "", label: str = "") -> str:
    """A tabular body, escaped, ready to paste. Numbers are never re-typed."""
    def esc(s):
        s = _fmt_value(s)
        for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
            s = s.replace(a, b)
        return s

    align = "l" + "r" * (len(cols) - 1)
    out = ["\\begin{table}[t]", "\\centering"]
    if caption:
        out.append("\\caption{%s}" % caption)
    if label:
        out.append("\\label{%s}" % label)
    out += ["\\small", "\\begin{tabular}{%s}" % align, "\\toprule",
            " & ".join(esc(c) for c in cols) + " \\\\", "\\midrule"]
    out += [" & ".join(esc(r[c]) for c in cols) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def render_claim(rows, label: str, conn, extra: str = "") -> str:
    """The provenance comment the paper convention requires, generated not typed.

    Emits the marker plus a `src(...)` list of the exact hashes behind the table,
    and states the citability check that was applied.
    """
    import datetime
    hashes = [r["hash"] for r in rows if "hash" in r.keys()]
    lines = [f"% CLAIM({label}): date={datetime.date.today().isoformat()} "
             f"generated by tools/query_results.py"]
    if extra:
        lines += [f"%   {ln}" for ln in extra.splitlines()]
    if hashes:
        lines.append("%   src:")
        for h in hashes:
            row = conn.execute(
                "SELECT video, component, codec, qp FROM runs WHERE hash=?", (h,)).fetchone()
            if row:
                lines.append(f"%     {row['video']}={h}  ({row['component']}, "
                             f"{row['codec']} qp={row['qp']})")
            else:
                lines.append(f"%     {h}")
    lines.append("%   All rows drawn from v_citable: empty invariant_failures and evaluated.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# canned queries
# --------------------------------------------------------------------------

def cmd_exists(conn, args):
    where, params = ["1=1"], []
    if args.video:
        where.append("video = ?"); params.append(args.video)
    if args.dataset:
        where.append("dataset = ?"); params.append(args.dataset)
    if args.component:
        where.append("component = ?"); params.append(args.component)
    view = "runs" if args.include_uncitable else "v_citable"
    sql = (f"SELECT hash, video, component, codec, qp, block_size, degradation, restorer, "
           f"actual_bitrate_bps, n_invariant_failures FROM {view} "
           f"WHERE {' AND '.join(where)} ORDER BY video, component, qp")
    return conn.execute(sql, params).fetchall()


def cmd_missing_metric(conn, args):
    sql = """
      SELECT r.hash, r.video, r.component, r.codec, r.qp
      FROM v_citable r
      WHERE NOT EXISTS (SELECT 1 FROM metrics m
                        WHERE m.hash=r.hash AND m.region=? AND m.metric=?)
      ORDER BY r.video, r.component
    """
    return conn.execute(sql, (args.region, args.metric)).fetchall()


def cmd_datasets(conn, args):
    return conn.execute("""
      SELECT dataset, component, count(*) AS n, count(DISTINCT video) AS clips
      FROM v_citable GROUP BY dataset, component ORDER BY dataset, component
    """).fetchall()


def cmd_compare(conn, args):
    """Paired A/B over a join key, with the delta computed in SQL."""
    key = args.pair_by
    sql = f"""
      WITH a AS (
        SELECT r.{key} AS k, r.hash AS hash, r.actual_bitrate_bps AS rate,
               MAX(CASE WHEN m.metric=? THEN m.value END) AS val
        FROM v_citable r LEFT JOIN metrics m ON m.hash=r.hash AND m.region=?
        WHERE {args.a_where} GROUP BY r.hash),
      b AS (
        SELECT r.{key} AS k, r.hash AS hash, r.actual_bitrate_bps AS rate,
               MAX(CASE WHEN m.metric=? THEN m.value END) AS val
        FROM v_citable r LEFT JOIN metrics m ON m.hash=r.hash AND m.region=?
        WHERE {args.b_where} GROUP BY r.hash)
      SELECT b.k AS {key}, b.hash AS hash,
             a.val AS baseline, b.val AS arm,
             (b.val - a.val) AS delta,
             ROUND(100.0*(b.rate-a.rate)/a.rate, 2) AS d_rate_pct
      FROM b JOIN a USING(k) ORDER BY delta DESC
    """
    # Refuse an ambiguous pairing rather than emitting a table with a duplicated
    # key. A --where that matches two runs for one video silently produces two
    # rows for it and a wrong n; caught in practice the first time this ran.
    for side, where in (("a", args.a_where), ("b", args.b_where)):
        dupes = conn.execute(
            f"SELECT {key} AS k, count(*) AS n, group_concat(hash) AS hashes "
            f"FROM v_citable WHERE {where} GROUP BY {key} HAVING count(*) > 1"
        ).fetchall()
        if dupes:
            detail = "; ".join(f"{d['k']}: {d['n']} runs ({d['hashes']})" for d in dupes)
            raise SystemExit(
                f"--{side}-where matches multiple runs for the same {key}, so the "
                f"comparison is ambiguous and would emit duplicate rows:\n  {detail}\n"
                f"Narrow the filter until each {key} selects exactly one run.")

    rows = conn.execute(sql, (args.metric, args.region, args.metric, args.region)).fetchall()
    if rows:
        deltas = [r["delta"] for r in rows if r["delta"] is not None]
        if deltas:
            worse = sum(1 for d in deltas if d > 0)
            print(f"# n={len(deltas)}  mean delta {sum(deltas)/len(deltas):+.4f}  "
                  f"{worse}/{len(deltas)} higher", file=sys.stderr)
            print("# Verdict wording must come from presley.suite.assess_metric, "
                  "not from this mean.", file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--format", choices=("table", "csv", "latex"), default="table")
    ap.add_argument("--claim", action="store_true",
                    help="also emit the CLAIM provenance block for the rows")
    ap.add_argument("--label", default="tab:generated", help="label for --claim/--format latex")
    ap.add_argument("--caption", default="")
    ap.add_argument("--include-uncitable", action="store_true",
                    help="include runs with invariant_failures (they are named)")
    ap.add_argument("--no-refresh", action="store_true", help="do not index new runs first")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("exists", help="what runs exist")
    p.add_argument("--video"); p.add_argument("--dataset"); p.add_argument("--component")
    p.set_defaults(fn=cmd_exists)

    p = sub.add_parser("missing-metric", help="citable runs lacking a metric")
    p.add_argument("--metric", default="lpips_mean")
    p.add_argument("--region", default="background")
    p.set_defaults(fn=cmd_missing_metric)

    p = sub.add_parser("datasets", help="coverage by dataset and component")
    p.set_defaults(fn=cmd_datasets)

    p = sub.add_parser("compare", help="paired A/B with deltas")
    p.add_argument("--a-where", required=True); p.add_argument("--b-where", required=True)
    p.add_argument("--metric", default="lpips_mean"); p.add_argument("--region", default="background")
    p.add_argument("--pair-by", default="video")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("sql", help="arbitrary SQL")
    p.add_argument("query")
    p.set_defaults(fn=lambda conn, a: conn.execute(a.query).fetchall())

    args = ap.parse_args()
    conn = db.connect(args.results_dir)
    if not args.no_refresh:
        ensure_fresh(conn, args.results_dir)

    rows = args.fn(conn, args)
    if not rows:
        print("(no rows)")
        return 0
    cols = list(rows[0].keys())

    if args.include_uncitable:
        bad = [r["hash"] for r in rows
               if "n_invariant_failures" in r.keys() and r["n_invariant_failures"]]
        if bad:
            print(f"warning: {len(bad)} row(s) are NOT citable: {', '.join(bad)}",
                  file=sys.stderr)

    if args.format == "latex":
        print(render_latex(rows, cols, args.caption, args.label))
    elif args.format == "csv":
        print(render_csv(rows, cols))
    else:
        print(render_table(rows, cols))

    if args.claim:
        print()
        print(render_claim(rows, args.label, conn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
