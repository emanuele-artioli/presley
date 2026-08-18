#!/usr/bin/env python3
"""R1 — does PRESLEY's advantage depend on the operating point? Runs ONCE.

Pre-registered in `docs/PREREG_R1_REGIME_SCOPE.md` before Wave B existed. This
is the sixth attempt at a scope result here and the first that is properly
powered, so the design matters more than the number.

Why this shape. All five prior attempts correlated a VIDEO-LEVEL attribute
against a PER-VIDEO OUTCOME RATE, and that shape manufactures both confounds
that killed them:

  * per-video CELL COUNT -- it is the rate's own denominator, and ten of
    nineteen videos contributed a single contested cell, where a rate can only
    be 0 or 1;
  * DATASET PROVENANCE -- rho=0.907 with the outcome, because the non-DAVIS
    clips were *selected* to differ in kind.

R1's unit is the rate LADDER and its statistic is a WITHIN-LADDER CONTRAST.
Under that design each ladder contributes exactly one scalar however many rungs
it has, so cell count cannot enter; and dataset origin is constant within a
video, so provenance differences out exactly -- as do duration, foreground area,
motion and texture, every attribute already refuted. These are not controls that
can be argued with; the confounds are absent.

Regime coordinate is the matched PRISTINE BASELINE's own background LPIPS at
that rung: measured, codec-independent, and computed only from the baseline arm
so it cannot be contaminated by the outcome. Never QP (x265 32 is not svtav1
43); never absolute bitrate (that IS codec efficiency).

Stopping rule, quoted from the pre-registration: R1 runs once. If it does not
fire, STOP -- no second coordinate, no cell-level version. A properly powered
negative closes the question and is worth more than five underpowered ones.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from presley import db  # noqa: E402
from presley.compare import REGION_METRIC_KEYS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ratematched_n13 import ARMS, LADDERS, _DEFAULTS, _mismatch  # noqa: E402

HOLM_K = 4          # {quality, bitrate} x {presley vs elvis, presley vs baseline}
ALPHA = 0.05

# Pre-registered bounds, docs/PREREG_R1_REGIME_SCOPE.md.
CONTRAST_PLAUSIBLE = (0.00, 0.10)     # |per-ladder contrast|, BG-LPIPS
CONTRAST_ALARM = 0.25                 # 5x JND -- investigate a bug first
TRIVIAL_MAGNITUDE = 0.01              # unanimity below this is not an effect


def sign_p(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) * 0.5 ** n for i in range(lo + 1)))


def min_attainable(n):
    return sign_p(n, n) if n else 1.0


def metric(doc, region, name):
    key = REGION_METRIC_KEYS.get(region, {}).get(name)
    v = ((doc.get("metrics") or {}).get(region) or {}).get(key) if key else None
    return None if v is None else float(v)


def collect(conn):
    """{ladder: {qp: {'baseline'|'elvis'|'presley_ai': doc}}}"""
    out = {}
    for row in conn.execute(
            "SELECT hash, component, video, codec, qp FROM v_citable "
            "WHERE component IN ('baselines','elvis','presley_ai')"):
        doc = db.get_run(conn, row["hash"])
        cfg = doc.get("config") or {}
        key = (row["video"], row["codec"], cfg.get("width"), cfg.get("height"))
        if key not in LADDERS or row["qp"] not in LADDERS[key]:
            continue
        comp = row["component"]
        if comp in ARMS:
            if cfg.get("block_size") != 8:
                continue
            # Use the SAME matcher as analyze_ratematched_n13: ARMS values may be
            # whitelist tuples (restorer_params) and some keys are absent from
            # configs produced before the schema gained them (selection_rule), so
            # a naive `cfg.get(k) != v` rejects every arm -- which is why this
            # tool reported "no ladder has all three arms" on a tree that has them.
            if any(_mismatch(cfg.get(k, _DEFAULTS.get(k)), v)
                   for k, v in ARMS[comp].items()):
                continue
        out.setdefault(key, {}).setdefault(row["qp"], {})[comp] = doc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--coordinate", choices=["bg_lpips", "fg_psnr"],
                    default="bg_lpips",
                    help="regime coordinate; both must AGREE IN SIGN or R1 is withdrawn")
    a = ap.parse_args()

    conn = db.connect(a.results_dir)
    data = collect(conn)

    print("=" * 80)
    print("R1 -- does the outcome depend on operating point?  (runs ONCE)")
    print(f"coordinate: matched pristine baseline's {a.coordinate}")
    print("contrast: median(2 most-starved rungs) - median(2 most-comfortable)")
    print("=" * 80)

    contrasts, skipped = [], []
    for key, rungs in LADDERS.items():
        per = data.get(key, {})
        pts = []
        for qp in rungs:
            cell = per.get(qp, {})
            b, e, p = cell.get("baselines"), cell.get("elvis"), cell.get("presley_ai")
            if not (b and e and p):
                continue
            coord = (metric(b, "background", "lpips") if a.coordinate == "bg_lpips"
                     else metric(b, "foreground", "psnr"))
            pe, ee = metric(p, "background", "lpips"), metric(e, "background", "lpips")
            if coord is None or pe is None or ee is None:
                continue
            pts.append((coord, pe - ee))       # negative = PRESLEY better
        if len(pts) < 4:
            skipped.append((key, len(pts)))
            continue
        # More starved = worse baseline quality = HIGHER bg_lpips / LOWER fg_psnr.
        pts.sort(key=lambda t: t[0], reverse=(a.coordinate == "bg_lpips"))
        starved = statistics.median(v for _, v in pts[:2])
        comfy = statistics.median(v for _, v in pts[-2:])
        contrasts.append((key[0], key[1], starved - comfy))

    if skipped:
        print(f"\nladders skipped for incomplete 3-component rungs ({len(skipped)}):")
        for key, got in skipped:
            print(f"   {key[0]:30s}{key[1]:>8}  only {got}/4 rungs have all three arms")

    n = len(contrasts)
    if n == 0:
        print("\nno ladder has all three arms at four rungs -- R1 cannot run yet.")
        return 1

    print(f"\n{'ladder':30}{'codec':>8}{'contrast':>12}")
    for v, c, d in contrasts:
        print(f"{v:30}{c:>8}{d:>+12.4f}")

    neg = sum(1 for _, _, d in contrasts if d < 0)
    p_raw = sign_p(neg, n)
    p_holm = min(1.0, p_raw * HOLM_K)
    mags = [abs(d) for _, _, d in contrasts]
    med_mag = statistics.median(mags)

    print(f"\nn = {n} ladders")
    print(f"  contrast negative (PRESLEY relatively better when starved) on {neg}/{n}")
    print(f"  exact two-tailed sign p = {p_raw:.6f}")
    print(f"  Holm-corrected (k={HOLM_K})   = {p_holm:.6f}")
    print(f"  min attainable p at n={n}     = {min_attainable(n):.6f}")
    print(f"  median |contrast|             = {med_mag:.4f} BG-LPIPS")

    print("\nPre-registered bound status:")
    lo, hi = CONTRAST_PLAUSIBLE
    outside = [(v, d) for v, _, d in contrasts if not (lo <= abs(d) <= hi)]
    alarm = [(v, d) for v, _, d in contrasts if abs(d) > CONTRAST_ALARM]
    print(f"  |contrast| plausible {lo}-{hi}: {len(outside)}/{n} outside")
    for v, d in outside:
        print(f"      {v:30s}{d:+.4f}")
    print(f"  ALARM |contrast| > {CONTRAST_ALARM}: "
          f"{'none' if not alarm else alarm} "
          f"{'' if not alarm else '<-- investigate a bug BEFORE believing this'}")
    if neg in (0, n) and med_mag < TRIVIAL_MAGNITUDE:
        print(f"  *** ALARM: unanimous at median |contrast| {med_mag:.4f} < "
              f"{TRIVIAL_MAGNITUDE} -- the coordinate orders trivially; this is "
              f"NOT evidence of a regime effect")

    fires = p_holm <= ALPHA and med_mag >= TRIVIAL_MAGNITUDE
    print("\n" + "=" * 80)
    if fires:
        print("R1 FIRES: the outcome depends on the operating point.")
        print("Confirm with --coordinate fg_psnr; the two MUST agree in sign or")
        print("R1 is withdrawn per the pre-registration.")
    else:
        print("R1 DOES NOT FIRE.")
        print("Per the stopping rule: STOP. Do not try a second coordinate, and do")
        print("not fall back to a cell-level version -- cell-level correlations run")
        print("~4x more 'significant' and are exactly the inflation the ladder unit")
        print("exists to prevent.")
        print("")
        print("Report it as the properly powered negative it is: with 13 ladders and")
        print("a design immune by construction to the cell-count and provenance")
        print("confounds that invalidated five earlier attempts, we detect no")
        print("dependence of the outcome on operating point.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
