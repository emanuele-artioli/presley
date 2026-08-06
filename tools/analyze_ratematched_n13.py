#!/usr/bin/env python3
"""PRESLEY vs ELVIS at matched rate, extended from 9 clips to 13 ladders.

`tab:breadth-ratematched` established this on 9 non-DAVIS clips at x265 QP
32/37/42/47: BD-rate -51.4% on background LPIPS, 9/9, foreground a wash. That is
the strongest single result in the article and it currently sits inside a
"breadth" subsection answering a referee's dataset question.

Wave B added the ten bs8 ELVIS runs that were missing from the four DAVIS
SVT-AV1 ladders (ELVIS was bs16 there while PRESLEY was bs8, so the arms were
not block-size matched). With those, the same comparison runs on **13 ladders
across two codecs and three dataset families**.

Why that matters, in one line: at n=9 a unanimous two-tailed sign test gives
p=0.0039 and survives a Holm family of 12; at n=13 it gives **p=0.000244** and
survives a family of 204 -- i.e. it survives a referee who insists on folding in
every candidate this project has ever tried. The extension buys robustness to
the multiple-comparisons argument, not just a bigger number.

This is a SEPARATE tool from `analyze_ratematched.py` on purpose. That one
reproduces a published table and must keep doing so byte-for-byte; this one
cross-checks against it (`--verify-published`) rather than replacing it.

Ladders are per-video because rungs are recalibrated per video and codec -- the
DAVIS SVT-AV1 QP values are not the non-DAVIS x265 ones and must not be pooled
as if they were.
"""
from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from bd_rate import BDError, bd_rate, overlap_fraction  # noqa: E402
from presley import db  # noqa: E402
from presley.compare import REGION_METRIC_KEYS  # noqa: E402

# (video, codec) -> rungs. The nine non-DAVIS clips are the published set and
# their rungs must not change; the four DAVIS ladders are the Wave-B extension.
# Resolution is part of the key, not a global filter: `bear` and `camel` also
# have 1920x1080 and 1280x720 runs at the same QPs, and mixing a 1080p rung into
# a 360p ladder yields a meaningless BD-rate. It is in the key rather than a
# constant because `ptq7rtia` is portrait (360x640) and `0e4068b53f` is 640x480.
LADDERS = {
    ("mosev2/8i1uo3x9", "x265", 640, 360): [32, 37, 42, 47],
    ("mosev2/fii86rku", "x265", 640, 360): [32, 37, 42, 47],
    ("mosev2/jxmcdk8k", "x265", 640, 360): [32, 37, 42, 47],
    ("mosev2/ptq7rtia", "x265", 360, 640): [32, 37, 42, 47],
    ("mosev2/zofozj6l", "x265", 640, 360): [32, 37, 42, 47],
    ("youtube_vos/0e4068b53f", "x265", 640, 480): [32, 37, 42, 47],
    ("youtube_vos/282651c6f7", "x265", 640, 360): [32, 37, 42, 47],
    ("youtube_vos/30fe0ed0ce", "x265", 640, 360): [32, 37, 42, 47],
    ("youtube_vos/b1a8a404ad", "x265", 640, 360): [32, 37, 42, 47],
    ("bear", "svtav1", 640, 360): [43, 51, 58, 61],
    ("camel", "svtav1", 640, 360): [42, 50, 58, 62],
    ("dog", "svtav1", 640, 360): [50, 55, 60, 63],
    ("pigs", "svtav1", 640, 360): [50, 55, 60, 63],
}
PUBLISHED = {"n": 9, "bd_bg_mean": -51.4}   # tab:breadth-ratematched
PUBLISHED_TOL = 0.5                          # percentage points

# Pre-registered in docs/RATEMATCHED_BREADTH.md, quoted not re-derived.
BOUNDS = {
    "bd_bg":   (-70.0, -20.0, "BD-rate BG-LPIPS, PRESLEY vs ELVIS"),
    "bd_fg":   (-15.0, +15.0, "BD-rate FG-LPIPS (both arms protect FG: expect a wash)"),
    "overlap": (0.5, 1.01, "rate overlap_fraction"),
}


def sign_p(k, n):
    """Exact two-tailed sign test. One-tailed here would be the trap hard rule
    2b exists to stop: 5/5 is 0.0625, not 0.031."""
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) * 0.5 ** n for i in range(lo + 1)))


def metric_value(doc, region, metric):
    key = REGION_METRIC_KEYS.get(region, {}).get(metric)
    if key is None:
        return None
    v = ((doc.get("metrics") or {}).get(region) or {}).get(key)
    return None if v is None else float(v)


# Pin the exact arm on each side. Selecting by component alone is unsafe here:
# a single (video, qp, component, block_size) key can be shared by 20+ runs
# spanning different degradations and restorers, and a dict assignment then keeps
# whichever the DB happened to return last. That produced a non-monotonic ladder
# and a +711% BD-rate earlier this week. Ambiguity is an error, not a coin flip.
ARMS = {
    "elvis": {"removal_mode": "blackout", "inpainter": "propainter",
              "fg_protect": True, "shrink_amount": 0.25},
    "presley_ai": {"degradation": "downsample", "restorer": "realesrgan",
                   "fg_protect": True, "shrink_amount": 0.25},
}


def curves(conn):
    """{(component, video, codec): {qp: (rate, {metric: value})}}, citable only."""
    out = {}
    seen = {}
    for row in conn.execute(
            "SELECT hash, component, video, codec, qp FROM v_citable "
            "WHERE component IN ('elvis','presley_ai')"):
        doc = db.get_run(conn, row["hash"])
        cfg = doc.get("config") or {}
        key = (row["video"], row["codec"], cfg.get("width"), cfg.get("height"))
        if key not in LADDERS or row["qp"] not in LADDERS[key]:
            continue
        # Block size must match ACROSS arms or the comparison is not controlled.
        # This is exactly what Wave B existed to fix; assert it rather than hope.
        if cfg.get("block_size") != 8:
            continue
        spec = ARMS.get(row["component"], {})
        if any(cfg.get(k) != v for k, v in spec.items()):
            continue
        ident = (row["component"], *key, row["qp"])
        if ident in seen and seen[ident] != row["hash"]:
            raise SystemExit(
                f"AMBIGUOUS ARM for {ident}: {seen[ident]} and {row['hash']} both "
                f"match. Tighten ARMS rather than letting the DB order decide."
            )
        seen[ident] = row["hash"]
        out.setdefault((row["component"], *key), {})[row["qp"]] = (
            float(doc["actual_bitrate_bps"]),
            {"bg_lpips": metric_value(doc, "background", "lpips"),
             "fg_lpips": metric_value(doc, "foreground", "lpips")},
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--verify-published", action="store_true",
                    help="check the 9 published clips still reproduce -51.4%%")
    a = ap.parse_args()

    conn = db.connect(a.results_dir)
    cur = curves(conn)

    print("=" * 82)
    print("PRESLEY vs ELVIS at matched rate -- 13 ladders, 2 codecs, 3 dataset families")
    print("BD-rate is PRESLEY anchored on ELVIS; NEGATIVE = PRESLEY needs fewer bits.")
    print("=" * 82)

    incomplete = []
    for key, rungs in LADDERS.items():
        v, codec = key[0], key[1]
        for comp in ("elvis", "presley_ai"):
            have = sorted(cur.get((comp, *key), {}))
            if have != rungs:
                incomplete.append(f"{comp}/{v}/{codec}: have {have}, want {rungs}")
    if incomplete:
        print("\nINCOMPLETE -- refusing to analyse a partial set:")
        for m in incomplete:
            print("   ", m)
        return 1

    rows = []
    print(f"\n{'ladder':34}{'codec':>8}{'BD BG-LPIPS':>13}{'BD FG-LPIPS':>13}{'overlap':>9}")
    for key, rungs in LADDERS.items():
        v, codec = key[0], key[1]
        e = cur[("elvis", *key)]
        p = cur[("presley_ai", *key)]
        re_ = [e[q][0] for q in rungs]
        rp = [p[q][0] for q in rungs]
        try:
            bg = bd_rate(re_, [e[q][1]["bg_lpips"] for q in rungs],
                         rp, [p[q][1]["bg_lpips"] for q in rungs], lower_is_better=True)
            fg = bd_rate(re_, [e[q][1]["fg_lpips"] for q in rungs],
                         rp, [p[q][1]["fg_lpips"] for q in rungs], lower_is_better=True)
        except BDError as exc:
            print(f"{v:34}{codec:>8}  not computable: {exc}")
            continue
        ov = overlap_fraction(re_, rp)
        rows.append((v, codec, bg, fg, ov))
        print(f"{v:34}{codec:>8}{bg:>12.1f}%{fg:>12.1f}%{ov:>9.2f}")

    n = len(rows)
    wins = sum(1 for _, _, bg, _, _ in rows if bg < 0)
    fg_wash = sum(1 for _, _, _, fg, _ in rows if abs(fg) <= 15.0)
    p = sign_p(wins, n)
    mean_bg = float(np.mean([bg for _, _, bg, _, _ in rows]))

    print(f"\nBD-rate BG-LPIPS: PRESLEY ahead on {wins}/{n} ladders, mean {mean_bg:+.1f}%")
    print(f"  exact two-tailed sign p = {p:.6f}")
    print(f"  survives a Holm family of k <= {int(0.05 / p) if p > 0 else 'inf'}")
    print(f"FG is a wash (|BD| <= 15%) on {fg_wash}/{n} -- both arms protect the foreground")

    low = [(v, ov) for v, _, _, _, ov in rows if ov < 0.5]
    print(f"\nOverlap gate (>=0.50): {'all pass' if not low else low}")

    print("\nPre-registered bounds:")
    lo, hi, label = BOUNDS["bd_bg"]
    ok = lo <= mean_bg <= hi
    print(f"  {label}: band {lo:+g}..{hi:+g}, got {mean_bg:+.1f} -> "
          f"{'in bounds' if ok else '*** BREACH ***'}")

    # The FG band is PER LADDER and it is the one that fires. Report it as fired
    # rather than quoting only the axis that behaved -- the pre-registration said
    # "expect a wash", and on several ladders it is not a wash in either
    # direction. Bands are never re-fitted to accommodate a result.
    flo, fhi, flabel = BOUNDS["bd_fg"]
    out_of_band = [(v, fg) for v, _, _, fg, _ in rows if not (flo <= fg <= fhi)]
    print(f"  {flabel}: band {flo:+g}..{fhi:+g} per ladder -> "
          f"{len(out_of_band)}/{n} OUTSIDE"
          f"{' *** BREACH, recorded not re-fitted ***' if out_of_band else ''}")
    for v, fg in sorted(out_of_band, key=lambda t: -abs(t[1])):
        side = "PRESLEY worse on FG" if fg > 0 else "PRESLEY better on FG"
        print(f"      {v:32s}{fg:+7.1f}%   {side}")
    if out_of_band:
        print("    Reading: the foreground is NOT uniformly a wash. It moves in BOTH")
        print("    directions across ladders, so the correct statement is that the")
        print("    background win does not come at a systematic foreground cost --")
        print("    not that the foreground is unchanged.")

    if a.verify_published:
        pub = [r for r in rows if r[1] == "x265"]
        pm = float(np.mean([bg for _, _, bg, _, _ in pub]))
        ok = abs(pm - PUBLISHED["bd_bg_mean"]) <= PUBLISHED_TOL
        print(f"\nPublished-set check ({len(pub)} x265 clips): mean {pm:+.1f}% vs "
              f"landed {PUBLISHED['bd_bg_mean']:+.1f}% -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print("  MISMATCH means this tool disagrees with a landed table. Stop and")
            print("  reconcile before quoting anything from either.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
