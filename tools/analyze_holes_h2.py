"""Is the restorer-on-fill result stable? — H2, HOLE(tab:goal2).

`CLAIM(tab:goal2)` is deliberately weak already: restoration is a >1 JND win on
a zero-information fill (blackout) and video-dependent on freeze. Its own
`NOTE` records that the finding has been **mis-stated twice in opposite
directions**, so this script reports the direction per video rather than any
pooled verdict, and prints the cells that disagree.

H2 adds two videos at their starved QP and a second QP on the incumbents.
**n=4 videos is still under the n>=6 that hard rule 2b requires**, so nothing
here can be a significance verdict — it can only widen or narrow the
descriptive claim.

JND thresholds come from `presley.compare`, the project's single source of
truth, rather than being restated here.

⚠ `NOTE(tab:goal2)` warns that FG-LPIPS is mask-**weighted**, not
mask-isolated: a sub-JND FG-LPIPS move between restorers is background leaking
in, never an FG effect. The FG check below is therefore a guard on
`fg_protect`, not a finding, and is labelled that way in the output.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from presley.compare import JND  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
# video -> the QPs H2 covers (dog/pigs at their starved rung; a 2nd QP on the
# incumbents, alongside the QP each already had in CLAIM(tab:goal2)).
CELLS = {"dog": (50,), "pigs": (50,), "bear": (58,), "camel": (58,)}
FILLS = ("none", "telea", "e2fgvi", "propainter")
MODES = ("blackout", "freeze")
LPIPS_JND = JND["lpips"][0]

# Pre-registered in docs/HOLE_CLOSURE_WAVE.md, H2, in JND multiples.
BOUNDS = {
    "blackout": (1.0, 3.5, "no improvement on any video"),
    "freeze": (-1.2, 1.2, "> 2x JND either way"),
}


def load():
    """Index every H2 cell by (video, qp, mode, fill).

    ⚠ `tab:goal2`'s two halves live in different components, and reading only
    one of them is how 16 of this set's cells were silently queued wrong:

      * freeze  -> `presley_ai`, `degradation: freeze`, `restorer: <fill>`
      * blackout -> `elvis`, `removal_mode: blackout`, `inpainter: <fill>`

    blackout removes a block outright, which is ELVIS's job; `presley_ai`
    rejects it outright. Both shapes are normalised to the same key here.
    """
    out = {}
    for p in RESULTS.glob("*/result.json"):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        c = d.get("config") or {}
        v = c.get("video")
        if v not in CELLS or c.get("block_size") != 8 or c.get("height") != 360:
            continue
        qp = c.get("codec_params", {}).get("qp")
        if qp not in CELLS[v] or c.get("codec") != "svtav1":
            continue
        if c.get("component") == "presley_ai":
            mode, fill = c.get("degradation"), c.get("restorer")
        elif c.get("component") == "elvis":
            mode, fill = c.get("removal_mode"), c.get("inpainter")
        else:
            continue
        if mode in MODES and fill in FILLS:
            out[(v, qp, mode, fill)] = d
    return out


def lpips(d, region):
    if d is None or d.get("invariant_failures"):
        return None
    return d["metrics"][region].get("lpips_mean")


def main():
    runs = load()
    print(f"BG-LPIPS vs the matched `none` control. JND = {LPIPS_JND}. "
          "Negative delta = restoration helps.")
    verdicts = {"blackout": [], "freeze": []}
    fg_spreads = []
    for v, qps in CELLS.items():
        for qp in qps:
            for mode in MODES:
                base = runs.get((v, qp, mode, "none"))
                b = lpips(base, "background")
                if b is None:
                    print(f"\n{v} QP{qp} {mode}: `none` control missing or not citable")
                    continue
                print(f"\n{v} QP{qp} {mode}   none = {b:.4f}")
                best = None
                for fill in FILLS:
                    if fill == "none":
                        continue
                    a = lpips(runs.get((v, qp, mode, fill)), "background")
                    if a is None:
                        print(f"    {fill:11} (incomplete or not citable)")
                        continue
                    delta = a - b
                    mult = abs(delta) / LPIPS_JND
                    verdict = "helps" if delta < 0 else "HARMS"
                    print(f"    {fill:11} {a:.4f}  delta {delta:+.4f}  "
                          f"{mult:.2f}x JND  {verdict}"
                          f"{'' if mult >= 1 else '  (sub-JND)'}")
                    if best is None or delta < best[1]:
                        best = (fill, delta)
                if best:
                    verdicts[mode].append((v, qp, best[0], -best[1] / LPIPS_JND))

                # fg_protect guard, NOT a finding -- FG-LPIPS is mask-weighted.
                fgs = [lpips(runs.get((v, qp, mode, f)), "foreground") for f in FILLS]
                fgs = [x for x in fgs if x is not None]
                if len(fgs) > 1:
                    fg_spreads.append((v, qp, mode, max(fgs) - min(fgs)))

    print("\n--- pre-registered bounds (JND multiples, best restorer vs none) ---")
    fired = False
    for mode in MODES:
        lo, hi, note = BOUNDS[mode]
        if not verdicts[mode]:
            print(f"  {mode}: no complete cells yet")
            continue
        for v, qp, fill, mult in verdicts[mode]:
            ok = lo <= mult <= hi
            fired |= not ok
            print(f"  {mode:9} {v:6} QP{qp:<3} best={fill:11} {mult:+6.2f}x JND  "
                  f"band {lo:+.1f}..{hi:+.1f}  "
                  f"{'in band' if ok else '*** OUT OF BAND — ALARM ***'}   [{note}]")

    print("\n--- fg_protect guard (NOT a finding: FG-LPIPS is mask-weighted) ---")
    for v, qp, mode, spread in fg_spreads:
        ok = spread <= LPIPS_JND
        fired |= not ok
        print(f"  {v:6} QP{qp:<3} {mode:9} FG-LPIPS spread {spread:.4f}  "
              f"{'sub-JND, as fg_protect requires' if ok else '*** SUPRA-JND — would contradict fg_protect ***'}")

    print("\n--- direction per video (the thing that has been mis-stated twice) ---")
    for mode in MODES:
        helps = [f"{v}QP{qp}" for v, qp, _, m in verdicts[mode] if m > 0]
        harms = [f"{v}QP{qp}" for v, qp, _, m in verdicts[mode] if m <= 0]
        crossing = [f"{v}QP{qp}" for v, qp, _, m in verdicts[mode] if m >= 1.0]
        print(f"  {mode:9} helps {len(helps)}/{len(verdicts[mode])} cells {helps}; "
              f"harms {harms}; clears JND in {crossing}")
    print("\n  n=4 videos is below hard rule 2b's n>=6: descriptive only, never")
    print("  'significant'. Report the per-cell direction, not a pooled verdict.")
    if fired:
        print("\nA bound fired. Investigate implementation / eval / data first; do not")
        print("cite until it is closed or explicitly revised with a stated reason.")


if __name__ == "__main__":
    main()
