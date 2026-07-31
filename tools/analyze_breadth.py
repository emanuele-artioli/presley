#!/usr/bin/env python3
"""Does PRESLEY survive outside DAVIS? Compare presley_ai to baseline and ELVIS.

Bounds were pre-registered in docs/BREADTH_PRESLEY_AI.md before any of these
runs existed. This script reports each comparison split by dataset as well as
pooled, because MOSEv2 (5 clips) and YouTube-VOS (4) are different populations
and the pre-registration says not to pool them if they disagree in sign.

Verdicts come from presley.suite.assess_metric so the mandated wording is
quoted rather than paraphrased; a sub-JND gain is never a win (hard rule 2b).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from presley import db  # noqa: E402
from presley.compare import JND, REGION_METRIC_KEYS  # noqa: E402
from presley.suite import PairedDelta, assess_metric, sign_test_p  # noqa: E402

GATING = [m for m in ("lpips", "dists", "psnr", "ssim")]


def arms(conn, qp):
    """{component: {video: doc}} for the non-DAVIS clips at one QP."""
    out = {}
    for row in conn.execute(
            "SELECT hash, component, video FROM v_citable "
            "WHERE dataset IN ('mosev2','youtube_vos') AND qp = ? ORDER BY video", (qp,)):
        doc = db.get_run(conn, row["hash"])
        out.setdefault(row["component"], {})[row["video"]] = doc
    return out


def metric_value(doc, region, metric):
    key = REGION_METRIC_KEYS.get(region, {}).get(metric)
    if key is None:
        return None
    node = (doc.get("metrics") or {}).get(region) or {}
    v = node.get(key)
    return None if v is None else float(v)


def compare(base, test, label, region, videos=None):
    vids = sorted(set(base) & set(test) & (set(videos) if videos else set(base)))
    if not vids:
        print(f"  {label}: no paired videos")
        return
    rates = []
    for v in vids:
        a, b = base[v]["actual_bitrate_bps"], test[v]["actual_bitrate_bps"]
        rates.append(100.0 * (b - a) / a)
    npos = sum(1 for r in rates if r > 0)
    print(f"\n  {label}   n={len(vids)}")
    print(f"    bitrate: mean {sum(rates)/len(rates):+.2f}%  {npos}/{len(rates)} higher  "
          f"sign p={sign_test_p(npos, len(rates)-npos):.4f}")
    for metric in GATING:
        pairs = []
        for v in vids:
            va, vb = metric_value(base[v], region, metric), metric_value(test[v], region, metric)
            if va is None or vb is None:
                continue
            pairs.append(PairedDelta((v,), "", "", va, vb, vb - va,
                                     abs(vb - va) < JND[metric][0]))
        if len(pairs) < 2:
            print(f"    {metric:6}: only {len(pairs)} usable pairs (backfill missing)")
            continue
        r = assess_metric(pairs, metric, region, family_size=1,
                          is_primary=(metric == "lpips"))
        star = "*" if r.is_primary else " "
        print(f"    {metric:6}{star} mean {r.mean_delta:+.4f}  "
              f"{r.n_positive}+/{r.n_negative}-  p={r.sign_p}  {r.verdict}")
        if r.is_primary:
            print(f"            wording: {r.wording}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    conn = db.connect(args.results_dir)
    # Refresh before reading. Skipping this produced a silent "presley_ai=0" and
    # nine "no paired videos" lines on a set of runs that were complete on disk:
    # they predated the dual-write, so the DB simply had not seen them. A stale
    # index does not error, it answers wrongly, which is the worst failure a
    # results query can have.
    stats = db.import_json_tree(conn, args.results_dir)
    print(f"(indexed {stats['imported']} runs, {stats['skipped']} unreadable)")

    for qp in (32, 37):
        a = arms(conn, qp)
        pa, el, bl = a.get("presley_ai", {}), a.get("elvis", {}), a.get("baselines", {})
        print(f"\n{'='*78}\nQP {qp}   presley_ai={len(pa)} elvis={len(el)} baselines={len(bl)}\n{'='*78}")
        for region in ("foreground", "background"):
            print(f"\n--- region: {region} ---")
            for name, subset in (("ALL (9)", None),
                                 ("mosev2 (5)", [v for v in pa if v.startswith("mosev2/")]),
                                 ("youtube_vos (4)", [v for v in pa if v.startswith("youtube_vos/")])):
                print(f"\n  [{name}]")
                compare(bl, pa, "PRESLEY vs pristine baseline", region, subset)
                compare(el, pa, "PRESLEY vs ELVIS", region, subset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
