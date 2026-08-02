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
            _, _, _, dbits, dfg = hit[0]
            lo, hi, note = BOUNDS[f"{mode}_starved"]
            ok = lo <= dbits <= hi
            fired |= not ok
            print(f"  {v:6} {mode:9} d-bits {dbits:+7.2f}%  band {lo:+.0f}..{hi:+.0f}  "
                  f"{'in band' if ok else '*** OUT OF BAND — ALARM ***'}   [{note}]")
            loss = -dfg
            ok_fg = loss <= FG_PSNR_JND
            fired |= not ok_fg
            print(f"  {v:6} {mode:9} FG-PSNR loss {loss:+6.2f} dB  "
                  f"{'within JND' if ok_fg else '*** ABOVE JND — ALARM ***'}")

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
