"""Does the starved/comfortable sign flip generalize? — H1b, HOLE(tab:av1).

`CLAIM(tab:av1)` is the paper's most quotable result: on SVT-AV1 the sign of
the bit relocation flips with regime -- ELVIS frees bits when starved and loses
outright when comfortable -- and it rests on two videos. H1b adds `dog` and
`pigs` over four recalibrated rungs each. This script reports, per video and
rung, the bitrate delta of each removal mode against the pristine baseline at
the same QP, and the foreground-PSNR cost of getting it.

**A sign that does not flip is a real outcome, not a failure**; the
pre-registered bounds below are checked so that outcome is reported rather than
explained away.

⚠ `pigs` cannot be starved as hard as bear/camel: at QP 63, the codec maximum,
its baseline is still ~1 dB above where the incumbents' ladders end. A weaker
regime effect on `pigs` must therefore not be read as content-dependence
without saying that first -- this script prints the caveat next to the numbers
so it cannot be quoted without it.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from bd_rate import BDError, bd_quality, bd_rate, overlap_fraction  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
VIDEOS = ("dog", "pigs")
RUNGS = (50, 55, 60, 63)
MODES = ("blackout", "freeze")
FG_PSNR_JND = 0.5

# Pre-registered in docs/HOLE_CLOSURE_WAVE.md, H1.
BOUNDS = {
    "blackout_starved": (-35.0, -10.0, "positive => no saving at all"),
    "freeze_starved": (-15.0, -3.0, "positive => no saving at all"),
    "fg_psnr_loss": (0.0, FG_PSNR_JND, f"> 1 dB loss; {FG_PSNR_JND} dB is JND"),
}
CAVEAT = ("pigs is less starved than bear/camel even at QP 63 (baseline 27.19 dB "
          "vs the incumbents' 26.2/25.6 dB floor) — a weaker effect here is "
          "partly that, not necessarily content-dependence.")


def load():
    base, elvis = {}, {}
    for p in RESULTS.glob("*/result.json"):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        c = d.get("config") or {}
        if c.get("video") not in VIDEOS or c.get("codec") != "svtav1":
            continue
        if c.get("height") != 360 or c.get("codec_params", {}).get("qp") not in RUNGS:
            continue
        key = (c["video"], c["codec_params"]["qp"])
        if c.get("component") == "baselines":
            base[key] = d
        elif c.get("component") == "elvis" and c.get("block_size") == 16:
            elvis[key + (c.get("removal_mode"),)] = d
    return base, elvis


def citable(d):
    return d is not None and not d.get("invariant_failures")


def main():
    base, elvis = load()
    rows = []
    for v in VIDEOS:
        print(f"\n=== {v} ===")
        print(f"  {'QP':>4}{'baseline kb/s':>15}{'mode':>10}{'kb/s':>10}"
              f"{'d-bits':>10}{'FG-PSNR base':>14}{'FG-PSNR arm':>13}{'d FG':>8}")
        for qp in RUNGS:
            b = base.get((v, qp))
            if not citable(b):
                print(f"  {qp:>4}  baseline missing or not citable")
                continue
            b_rate = b["actual_bitrate_bps"] / 1000.0
            b_fg = b["metrics"]["foreground"]["psnr_mean"]
            for mode in MODES:
                a = elvis.get((v, qp, mode))
                if not citable(a):
                    print(f"  {qp:>4}{b_rate:>15.1f}{mode:>10}"
                          f"{'(incomplete or not citable)':>45}")
                    continue
                a_rate = a["actual_bitrate_bps"] / 1000.0
                a_fg = a["metrics"]["foreground"]["psnr_mean"]
                dbits = 100.0 * (a_rate - b_rate) / b_rate
                dfg = a_fg - b_fg
                print(f"  {qp:>4}{b_rate:>15.1f}{mode:>10}{a_rate:>10.1f}"
                      f"{dbits:>+9.2f}%{b_fg:>14.2f}{a_fg:>13.2f}{dfg:>+8.2f}")
                rows.append((v, qp, mode, dbits, dfg))

    print("\n--- pre-registered bounds, checked at the most starved rung (QP 63) ---")
    print(f"  caveat: {CAVEAT}")
    fired = False
    for v in VIDEOS:
        for mode in MODES:
            hit = [r for r in rows if r[0] == v and r[2] == mode and r[1] == 63]
            if not hit:
                continue
            _, _, _, dbits, _ = hit[0]
            lo, hi, note = BOUNDS[f"{mode}_starved"]
            ok = lo <= dbits <= hi
            fired |= not ok
            print(f"  {v:6} {mode:9} d-bits {dbits:+7.2f}%  band {lo:+.0f}..{hi:+.0f}  "
                  f"{'in band' if ok else '*** OUT OF BAND — ALARM ***'}   [{note}]")

    # The FG bound is specified AT MATCHED RATE, so it must be checked there.
    # Comparing FG-PSNR at the same fixed QP measures a different quantity: the
    # arms sit at different bitrates, so a same-QP gap mixes the FG penalty with
    # whatever bits the transport moved. That mistake was made once here.
    print("\n--- FG-PSNR at MATCHED RATE (the quantity the bound names) ---")
    for v in VIDEOS:
        b_all = {qp: d for (vv, qp), d in base.items() if vv == v}
        for mode in MODES:
            a_all = {qp: d for (vv, qp, m), d in elvis.items() if vv == v and m == mode}
            have = [q for q in RUNGS if citable(b_all.get(q)) and citable(a_all.get(q))]
            if len(have) < 3:
                print(f"  {v:6} {mode:9} only {len(have)} usable rungs, no BD number")
                continue
            br = [b_all[q]["actual_bitrate_bps"] for q in have]
            ar = [a_all[q]["actual_bitrate_bps"] for q in have]
            qb = [b_all[q]["metrics"]["foreground"]["psnr_mean"] for q in have]
            qa = [a_all[q]["metrics"]["foreground"]["psnr_mean"] for q in have]
            try:
                r = bd_rate(br, qb, ar, qa, lower_is_better=False)
                dq = bd_quality(br, qb, ar, qa)
                ov = overlap_fraction(br, ar)
            except BDError as exc:
                print(f"  {v:6} {mode:9} refused: {exc}")
                continue
            lo, hi, note = BOUNDS["fg_psnr_loss"]
            loss = -dq
            ok = loss <= hi
            fired |= not ok
            print(f"  {v:6} {mode:9} BD-rate {r:+7.1f}%   FG-PSNR at equal rate "
                  f"{dq:+.2f} dB   overlap {ov:.2f}   "
                  f"{'within JND' if ok else '*** ABOVE JND ***'}   [{note}]")

    print("\n--- regime check: does the sign flip? ---")
    for v in VIDEOS:
        for mode in MODES:
            got = {qp: d for vv, qp, m, d, _ in rows if vv == v and m == mode for d in [d]}
            if len(got) < 2:
                continue
            most, least = got.get(63), got.get(50)
            if most is None or least is None:
                continue
            print(f"  {v:6} {mode:9} QP50 {least:+7.2f}%  ->  QP63 {most:+7.2f}%   "
                  f"{'sign flips across the ladder' if least * most < 0 else 'same sign at both ends'}")
    if fired:
        print("\nA bound fired. Investigate implementation / eval / data first; do not")
        print("cite until it is closed or explicitly revised with a stated reason.")


if __name__ == "__main__":
    main()
