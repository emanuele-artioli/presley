"""BD-rate curves for the H3 set — HOLE(tab:conditioned) on dog and pigs.

`tab:conditioned` carries the Goal-2 headline from n=2 videos at a single QP.
H3 adds two more videos over four fixed-QP rungs each, which is what turns the
screen into a curve. This script reads the finished runs and reports, per video
and per arm, the BD-rate against two references:

  * the pristine SVT-AV1 baseline at the same rungs -- the reference
    `tab:priced-trade` already uses, so its numbers are comparable; and
  * the unrestored arm of the same transport -- what the restorer itself buys.

It computes no metrics. BD integration and the overlap guard come from
`scripts/bd_rate.py`, which refuses to quote a BD number across disjoint
quality ranges; per `docs/HOLE_CLOSURE_WAVE.md` we reuse that path rather than
writing a second one. Bounds below are the pre-registered ones from that
document and are checked, not eyeballed.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from bd_rate import BDError, bd_rate, overlap_fraction  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
VIDEOS = ("dog", "pigs")
RUNGS = (50, 55, 60, 63)
ARMS = (("downsample", "none"), ("downsample", "realesrgan"),
        ("blur", "none"), ("blur", "unsharp"))

# Pre-registered in docs/HOLE_CLOSURE_WAVE.md, H3. (low, high, alarm-note)
BOUNDS = {
    "bd_fg_downsample_realesrgan": (-25.0, 5.0, "< -40% suspects a rate-accounting error"),
    "bd_bg_downsample_realesrgan": (-10.0, 90.0, "wide band is the known scope-back"),
    "overlap": (0.6, 1.01, "< 0.3 => BD numbers are extrapolation, quote none"),
}


def load():
    """Return {(video, qp, degradation, restorer): doc} plus {(video, qp): baseline}."""
    arms, base = {}, {}
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
        qp = c["codec_params"]["qp"]
        if c.get("component") == "baselines":
            base[(c["video"], qp)] = d
        elif c.get("component") == "presley_ai" and c.get("block_size") == 8:
            arms[(c["video"], qp, c.get("degradation"), c.get("restorer"))] = d
    return arms, base


def series(docs):
    """(rates kbps, fg-lpips, bg-lpips) over RUNGS; None if any rung is missing
    or any run is not citable."""
    rate, fg, bg = [], [], []
    for d in docs:
        if d is None or d.get("invariant_failures"):
            return None
        m = d["metrics"]
        if m["foreground"].get("lpips_mean") is None or m["background"].get("lpips_mean") is None:
            return None
        rate.append(d["actual_bitrate_bps"] / 1000.0)
        fg.append(m["foreground"]["lpips_mean"])
        bg.append(m["background"]["lpips_mean"])
    return rate, fg, bg


def bd(ref, arm, idx):
    try:
        return bd_rate(ref[0], ref[idx], arm[0], arm[idx], lower_is_better=True)
    except BDError as exc:
        return f"refused: {exc}"


def fmt(v):
    return f"{v:+8.1f}%" if isinstance(v, float) else f"  {v}"


def check(name, value):
    lo, hi, note = BOUNDS[name]
    if not isinstance(value, float):
        return f"    {name}: {value}"
    verdict = "in band" if lo <= value <= hi else "*** OUT OF BAND — ALARM ***"
    return f"    {name}: {value:+.1f}  (band {lo:+.1f}..{hi:+.1f})  {verdict}   [{note}]"


def main():
    arms, base = load()
    missing = [(v, qp, dg, r) for v in VIDEOS for qp in RUNGS for dg, r in ARMS
               if (v, qp, dg, r) not in arms]
    if missing:
        print(f"incomplete: {len(missing)} of {len(VIDEOS) * len(RUNGS) * len(ARMS)} "
              f"arm-runs missing; first few: {missing[:4]}")

    alarms = []
    for v in VIDEOS:
        print(f"\n=== {v} ===")
        ref_base = series([base.get((v, qp)) for qp in RUNGS])
        if ref_base is None:
            print("  no citable pristine baseline over all four rungs")
        print(f"  {'arm':26}{'kb/s @rungs':>34}{'BD-FG vs base':>15}{'BD-BG vs base':>15}")
        curves = {}
        for dg, r in ARMS:
            s = series([arms.get((v, qp, dg, r)) for qp in RUNGS])
            curves[(dg, r)] = s
            if s is None:
                print(f"  {dg + '+' + r:26}{'(incomplete or not citable)':>34}")
                continue
            rates = " ".join(f"{x:7.1f}" for x in s[0])
            f_ = bd(ref_base, s, 1) if ref_base else "no baseline"
            b_ = bd(ref_base, s, 2) if ref_base else "no baseline"
            print(f"  {dg + '+' + r:26}{rates:>34}{fmt(f_):>15}{fmt(b_):>15}")
            if (dg, r) == ("downsample", "realesrgan"):
                alarms.append(check("bd_fg_downsample_realesrgan", f_))
                alarms.append(check("bd_bg_downsample_realesrgan", b_))
                if ref_base:
                    ov = overlap_fraction(ref_base[0], s[0])
                    alarms.append(check("overlap", float(ov)))

        print("  restorer effect within transport (unrestored -> restored):")
        for dg, r in (("downsample", "realesrgan"), ("blur", "unsharp")):
            a, b_c = curves.get((dg, "none")), curves.get((dg, r))
            if a is None or b_c is None:
                print(f"    {dg}: incomplete")
                continue
            print(f"    {dg + ' none->' + r:28}BD-FG {fmt(bd(a, b_c, 1))}  "
                  f"BD-BG {fmt(bd(a, b_c, 2))}  overlap {overlap_fraction(a[0], b_c[0]):.2f}")

    print(f"\n--- pre-registered bounds ({len(alarms)} checks) ---")
    for line in alarms:
        print(line)
    if any("ALARM" in a for a in alarms):
        print("\nAt least one bound fired. Per AGENTS.md: investigate implementation /")
        print("eval / data bugs FIRST. Do not cite until the alarm is closed or the")
        print("bound is explicitly revised with a stated reason.")


if __name__ == "__main__":
    main()
