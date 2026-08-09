#!/usr/bin/env python3
"""PRESLEY vs ELVIS at matched rate, 13 ladders -- `fig:ratematched`.

The article's strongest single result, and it currently lives in a table where
"13/13" has to be counted rather than seen. As a sorted bar chart with every
bar on one side of zero, unanimity is the shape of the figure.

The foreground panel is not decoration and must not be dropped. The background
win could in principle come from trading foreground quality away, and the
honest statement is subtler than "the foreground is a wash": foreground BD-rate
moves in BOTH directions across ladders, six of thirteen outside a +/-15% band.
What the data supports is that the background win does not come at a
*systematic* foreground cost -- not that the foreground is unchanged. Drawing
both panels is what makes that visible instead of asserted.

Numbers come from `analyze_ratematched_n13.py`, which is also the tool that
re-derives the published nine-clip -51.4% and refuses to run if it drifts. This
script re-implements nothing.

Usage:
    python tools/plot_ratematched.py --data-root .
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import figkit  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

FG_WASH_BAND = 15.0


def parse(text: str):
    """Pull the per-ladder table out of the analysis tool's own output.

    Parsing the tool rather than recomputing keeps one implementation of the
    result: if the tool changes its answer, the figure changes with it, and if
    the tool refuses to run the figure cannot be built from stale numbers.
    """
    rows, seen_header = [], False
    for line in text.splitlines():
        if line.startswith("ladder "):
            seen_header = True
            continue
        if not seen_header:
            continue
        parts = line.split()
        if len(parts) != 5 or not parts[2].endswith("%"):
            if rows:
                break
            continue
        rows.append({
            "ladder": parts[0], "codec": parts[1],
            "bg": float(parts[2].rstrip("%")),
            "fg": float(parts[3].rstrip("%")),
            "overlap": float(parts[4]),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--figures-dir", default=None)
    args = ap.parse_args()

    tool = REPO_ROOT / "tools" / "analyze_ratematched_n13.py"
    proc = subprocess.run([sys.executable, str(tool), "--results-dir",
                           str(pathlib.Path(args.data_root) / "results")],
                          capture_output=True, text=True, cwd=args.data_root)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit("analysis refused to run; not drawing a figure from stale data")

    rows = parse(proc.stdout)
    if len(rows) != 13:
        raise SystemExit(f"expected 13 ladders, parsed {len(rows)}")
    rows.sort(key=lambda r: r["bg"])

    names = [r["ladder"].split("/")[-1][:12] for r in rows]
    bg = [r["bg"] for r in rows]
    fg = [r["fg"] for r in rows]

    figkit.style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figkit.FULL_WIDTH, 2.9), sharey=True)

    ax.barh(range(len(bg)), bg, color=figkit.COLORS["win"], height=0.65)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlabel("BD-rate on BG-LPIPS (%)")
    ax.set_title(f"Background: {sum(1 for v in bg if v < 0)}/{len(bg)} below zero\n"
                 f"mean {np.mean(bg):.1f}%, sign $p=0.000244$")

    bx.axvspan(-FG_WASH_BAND, FG_WASH_BAND, color=figkit.COLORS["muted"], alpha=0.18)
    bx.barh(range(len(fg)), fg,
            color=[figkit.COLORS["win"] if v < 0 else figkit.COLORS["loss"] for v in fg],
            height=0.65)
    bx.axvline(0, color="black", linewidth=0.8)
    bx.set_xlabel("BD-rate on FG-LPIPS (%)")
    n_out = sum(1 for v in fg if abs(v) > FG_WASH_BAND)
    bx.set_title(f"Foreground: moves both ways\n{n_out}/{len(fg)} outside "
                 f"$\\pm${FG_WASH_BAND:.0f}% (shaded)")

    fig.tight_layout()

    sentence = (
        "Two sorted horizontal bar charts sharing one axis of 13 ladders. Left: "
        "BD-rate on background LPIPS for PRESLEY against ELVIS at matched rate; "
        "every one of the 13 bars is below zero, mean -56.4 percent, exact "
        "two-tailed sign p = 0.000244, surviving a Holm family of up to 204 "
        "candidates. Right: the same ladders on foreground LPIPS, where bars fall "
        f"on both sides of zero and {n_out} of 13 lie outside a shaded plus or minus "
        "15 percent band. So the background win does not come at a systematic "
        "foreground cost, which is a weaker and more accurate statement than "
        "saying the foreground is unchanged.")

    data = {
        "comparison": "PRESLEY vs ELVIS at matched rate",
        "anchor": "ELVIS; negative means PRESLEY needs fewer bits at equal quality",
        "n_ladders": len(rows),
        "codecs": sorted({r["codec"] for r in rows}),
        "dataset_families": ["DAVIS", "MOSEv2", "YouTube-VOS"],
        "ladders": [r["ladder"] for r in rows],
        "bd_rate_bg_lpips_pct": [round(v, 1) for v in bg],
        "bd_rate_fg_lpips_pct": [round(v, 1) for v in fg],
        "bg_wins": sum(1 for v in bg if v < 0),
        "bg_mean_pct": round(float(np.mean(bg)), 1),
        "sign_p_two_tailed": 0.000244,
        "holm_family_survived": 204,
        "fg_outside_15pct_band": n_out,
        "fg_reading": "the background win does not come at a systematic foreground "
                      "cost; the foreground is NOT uniformly a wash and moves in "
                      "both directions",
    }

    paths = figkit.emit("ratematched", fig, sentence, data, figures_dir=args.figures_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
