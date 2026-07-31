#!/usr/bin/env python3
"""Is PRESLEY better than ELVIS at MATCHED BITRATE, on the 9 non-DAVIS clips?

The same-QP comparison (tab:breadth-ext-presley) found a background win above JND
on 9/9 clips -- but PRESLEY spent +23.1%/+14.9% more bits doing it, and the
paper's chain `presley_ai > elvis` is specified at matched bitrate. This script
answers the matched-rate question the only way the project's rules allow.

Why BD-rate and not a QP search: hard rule 1 forbids VBR/bitrate-targeting rate
control for degradation comparisons, so the arms cannot simply be driven to the
same bitrate. Instead each arm gets >= 4 fixed-QP rungs and the curves are
interpolated -- scripts/bd_rate.py, which is deliberately strict about the two
ways this analysis lies (too few points, and non-overlapping curves).

Bounds were pre-registered in docs/RATEMATCHED_BREADTH.md BEFORE these runs
existed. Every headline here is printed next to its band, and a breach is
labelled a BREACH rather than silently absorbed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from bd_rate import BDError, bd_rate, overlap_fraction  # noqa: E402
from presley import db  # noqa: E402
from presley.compare import JND, REGION_METRIC_KEYS  # noqa: E402

CLIPS = ["mosev2/8i1uo3x9", "mosev2/fii86rku", "mosev2/jxmcdk8k", "mosev2/ptq7rtia",
         "mosev2/zofozj6l", "youtube_vos/0e4068b53f", "youtube_vos/282651c6f7",
         "youtube_vos/30fe0ed0ce", "youtube_vos/b1a8a404ad"]
RUNGS = [32, 37, 42, 47]

# From docs/RATEMATCHED_BREADTH.md, quoted rather than re-derived.
BOUNDS = {
    "bd_bg":    (-70.0, -20.0, "BD-rate BG-LPIPS, PRESLEY vs ELVIS"),
    "bd_fg":    (-15.0, +15.0, "BD-rate FG-LPIPS (both arms protect FG: expect a wash)"),
    "overlap":  (0.5, 1.01, "rate overlap_fraction"),
    "starved":  (0.0, 0.5, "QP47 bitrate as a fraction of QP32's"),
}


def metric_value(doc, region, metric):
    key = REGION_METRIC_KEYS.get(region, {}).get(metric)
    if key is None:
        return None
    node = (doc.get("metrics") or {}).get(region) or {}
    v = node.get(key)
    return None if v is None else float(v)


def curves(conn):
    """{(component, video): {qp: (rate, {metric: value})}} over citable runs only."""
    out = {}
    for row in conn.execute(
            "SELECT hash, component, video, qp FROM v_citable "
            "WHERE dataset IN ('mosev2','youtube_vos') AND component IN ('elvis','presley_ai') "
            "ORDER BY video, component, qp"):
        if row["video"] not in CLIPS or row["qp"] not in RUNGS:
            continue
        doc = db.get_run(conn, row["hash"])
        # keyed by region+metric so background/foreground lpips cannot collide
        vals = {
            "bg_lpips": metric_value(doc, "background", "lpips"),
            "fg_lpips": metric_value(doc, "foreground", "lpips"),
            "bg_dists": metric_value(doc, "background", "dists"),
            "bg_psnr": metric_value(doc, "background", "psnr"),
        }
        out.setdefault((row["component"], row["video"]), {})[row["qp"]] = (
            float(doc["actual_bitrate_bps"]), vals)
    return out


def band(name, value):
    lo, hi, label = BOUNDS[name]
    if value is None or not np.isfinite(value):
        return f"  [{label}: n/a]"
    ok = lo <= value <= hi
    return f"  [{label}: band {lo:+g}..{hi:+g} -> {'in bounds' if ok else '*** BREACH ***'}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    conn = db.connect(args.results_dir)
    cur = curves(conn)

    print("=" * 78)
    print("Rate-matched PRESLEY vs ELVIS -- 9 non-DAVIS clips, fixed QP 32/37/42/47")
    print("Bounds pre-registered in docs/RATEMATCHED_BREADTH.md before any run existed.")
    print("=" * 78)

    # --- completeness, before any number is read -----------------------------
    missing = []
    for comp in ("elvis", "presley_ai"):
        for v in CLIPS:
            have = sorted(cur.get((comp, v), {}))
            if have != RUNGS:
                missing.append(f"{comp}/{v}: have {have}")
    if missing:
        print("\nINCOMPLETE -- not analysing until every curve has all 4 rungs:")
        for m in missing:
            print("   ", m)
        print(f"\n({len(missing)} curve(s) short. Re-run once the campaign finishes.)")
        return

    # --- per-clip BD-rate ----------------------------------------------------
    rows = []
    print(f"\n{'clip':26}{'BD BG-LPIPS':>13}{'BD FG-LPIPS':>13}{'overlap':>9}{'QP47/QP32':>11}")
    for v in CLIPS:
        e, p = cur[("elvis", v)], cur[("presley_ai", v)]
        re_ = [e[q][0] for q in RUNGS]
        rp = [p[q][0] for q in RUNGS]
        try:
            bg = bd_rate(re_, [e[q][1]["bg_lpips"] for q in RUNGS],
                         rp, [p[q][1]["bg_lpips"] for q in RUNGS], lower_is_better=True)
        except BDError as exc:
            bg = float("nan")
            print(f"{v:26}  BD BG-LPIPS not computable: {exc}")
        try:
            fg = bd_rate(re_, [e[q][1]["fg_lpips"] for q in RUNGS],
                         rp, [p[q][1]["fg_lpips"] for q in RUNGS], lower_is_better=True)
        except BDError:
            fg = float("nan")
        ov = overlap_fraction(re_, rp)
        starve = rp[-1] / rp[0]
        rows.append((v, bg, fg, ov, starve,
                     [e[q][1]["bg_lpips"] for q in RUNGS],
                     [p[q][1]["bg_lpips"] for q in RUNGS]))
        print(f"{v:26}{bg:+13.2f}{fg:+13.2f}{ov:9.2f}{starve:11.2f}")

    # --- the way this analysis can silently lie (bound 2) --------------------
    print("\n--- quality-range overlap (bound 2: no BD number without it) ---")
    bad = []
    for v, _, _, _, _, eq, pq in rows:
        lo = max(min(eq), min(pq))
        hi = min(max(eq), max(pq))
        if lo > hi:
            bad.append(v)
            print(f"  {v:26} ELVIS BG-LPIPS [{min(eq):.3f},{max(eq):.3f}] vs "
                  f"PRESLEY [{min(pq):.3f},{max(pq):.3f}] -- DISJOINT")
    if bad:
        print(f"\n  *** {len(bad)} clip(s) have DISJOINT BG-LPIPS ranges. Their BD numbers")
        print("      are extrapolation and MUST NOT be quoted. Report non-overlap instead:")
        print("      'PRESLEY's background quality is outside ELVIS's reach at any rate here.'")
    else:
        print("  all clips overlap in BG-LPIPS -- BD numbers are interpolation, not extrapolation")

    # --- monotonicity (bound 4) ----------------------------------------------
    print("\n--- monotonicity (bound 4: a breach is a bug, not a finding) ---")
    nonmono = []
    for comp in ("elvis", "presley_ai"):
        for v in CLIPS:
            r = [cur[(comp, v)][q][0] for q in RUNGS]
            q_ = [cur[(comp, v)][q][1]["bg_lpips"] for q in RUNGS]
            if any(r[i + 1] >= r[i] for i in range(3)):
                nonmono.append(f"{comp}/{v}: bitrate not falling with QP: {[round(x) for x in r]}")
            if any(q_[i + 1] < q_[i] for i in range(3)):
                nonmono.append(f"{comp}/{v}: BG-LPIPS not worsening with QP: {[round(x,4) for x in q_]}")
    print("  clean" if not nonmono else "\n".join("  *** " + m for m in nonmono))

    # --- pooled, and split, per the n>=6 rule --------------------------------
    def summarise(label, subset):
        bg = np.array([r[1] for r in rows if r[0] in subset], dtype=float)
        fg = np.array([r[2] for r in rows if r[0] in subset], dtype=float)
        ov = np.array([r[3] for r in rows if r[0] in subset], dtype=float)
        st = np.array([r[4] for r in rows if r[0] in subset], dtype=float)
        n = np.isfinite(bg).sum()
        print(f"\n  [{label} n={len(subset)}]")
        print(f"    BD-rate BG-LPIPS: mean {np.nanmean(bg):+.2f}%  "
              f"({int((bg < 0).sum())}/{n} favour PRESLEY)")
        print(band("bd_bg", float(np.nanmean(bg))))
        print(f"    BD-rate FG-LPIPS: mean {np.nanmean(fg):+.2f}%")
        print(band("bd_fg", float(np.nanmean(fg))))
        print(f"    overlap_fraction: mean {np.nanmean(ov):.2f}")
        print(band("overlap", float(np.nanmean(ov))))
        print(f"    QP47/QP32 bitrate: mean {np.nanmean(st):.2f}")
        print(band("starved", float(np.nanmean(st))))

    print("\n--- results against the pre-registered bounds ---")
    summarise("ALL", CLIPS)
    mose = [c for c in CLIPS if c.startswith("mosev2/")]
    yt = [c for c in CLIPS if c.startswith("youtube_vos/")]
    summarise("mosev2", mose)
    summarise("youtube_vos", yt)

    bg_all = np.nanmean([r[1] for r in rows])
    print("\n--- decision rule (docs/RATEMATCHED_BREADTH.md), applied ---")
    if bad:
        verdict = ("BG-LPIPS ranges disjoint on some clips -> report NON-OVERLAP, "
                   "quote no BD number for those clips")
    elif bg_all <= -20:
        verdict = ("chain CONFIRMED at matched rate on non-DAVIS data; the paper may say "
                   "presley_ai > elvis outside DAVIS, quoting BD-rate and overlap")
    elif bg_all < 0:
        verdict = ("directionally consistent but MODEST -- report the number, do NOT word "
                   "the chain as confirmed")
    else:
        verdict = ("chain NOT confirmed at matched rate: the same-QP win was bought with "
                   "bits. Narrow the claim accordingly.")
    print(f"  mean BD-rate(BG-LPIPS) = {bg_all:+.2f}%  ->  {verdict}")
    print(f"\n  (JND reference: BG-LPIPS {JND['lpips'][0]}; BD-rate is a rate delta, not a "
          "quality delta, so JND gates the per-rung deltas, not this number.)")


if __name__ == "__main__":
    main()
