#!/usr/bin/env python3
"""Where the time actually goes, per pipeline stage -- `fig:stage-timing`.

From `docs/w1e_stage_timings.json`: three trials per cell on a pinned GPU, run
into scratch directories so nothing under `results/` is touched, with the
device recorded on every trial and cells that mix device populations refused
rather than averaged.

Drawn as a stacked horizontal bar per arm, plus a log-scale inset for the
stages that are invisible at linear scale -- and they are invisible for a
reason worth showing: selection, the axis the article locates its remaining
headroom on, costs about a hundredth of a second against a pipeline of tens of
seconds.

Two readings the figure has to support, both of which are the point:

  * restoration dominates everything (70-89% of wall clock), so the cost of
    this class of system is the generative model and not the transport;
  * degradation costs more than encoding, so the server-side bottleneck is our
    own per-block Python loop rather than the codec.

Usage:
    python tools/plot_stage_timing.py --timings docs/w1e_stage_timings.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import figkit  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

STAGES = ["preprocess", "score", "select", "degrade", "encode",
          "sidechannel", "decode", "restore", "composite"]

# Sequential, so the eye reads the stack in pipeline order. Restoration gets
# the one saturated colour because it is the answer to "where does time go".
STAGE_COLOR = {
    "preprocess": "#cfd8dc", "score": "#b0bec5", "select": "#90a4ae",
    "degrade": "#78909c", "encode": "#546e7a", "sidechannel": "#455a64",
    "decode": "#37474f", "restore": "#c2545b", "composite": "#8d6e63",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", default=str(REPO_ROOT / "docs" / "w1e_stage_timings.json"))
    ap.add_argument("--figures-dir", default=None)
    args = ap.parse_args()

    rows = json.loads(pathlib.Path(args.timings).read_text())
    arms = sorted({r["arm"] for r in rows}, key=lambda a: -st.fmean(
        [r["wall_mean_s"] for r in rows if r["arm"] == a]))

    means = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        means[arm] = {s: st.fmean([r["stages"][s]["mean_s"] for r in rs if s in r["stages"]])
                      for s in STAGES if any(s in r["stages"] for r in rs)}

    figkit.style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figkit.FULL_WIDTH, 2.3),
                                 gridspec_kw={"width_ratios": [1.55, 1]})

    ypos = range(len(arms))
    for i, arm in enumerate(arms):
        left = 0.0
        for s in STAGES:
            v = means[arm].get(s, 0.0)
            if v <= 0:
                continue
            ax.barh(i, v, left=left, color=STAGE_COLOR[s], height=0.6,
                    label=s if i == 0 else None)
            left += v
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([a.replace(" + ", "\n+ ") for a in arms])
    ax.set_xlabel("seconds per 60-90 frame clip at 640$\\times$360")
    ax.set_ylim(len(arms) - 0.5, -0.5)
    ax.set_title("Restoration dominates")

    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLOR[s]) for s in STAGES]
    ax.legend(handles, STAGES, ncol=3, fontsize=5.5, loc="lower right", framealpha=0.9)

    # Log scale: the server-side stages are three orders of magnitude apart and
    # a linear axis renders selection as literally nothing.
    server = ["score", "select", "degrade", "encode", "sidechannel"]
    arm = "presley downsample + Real-ESRGAN"
    vals = [means[arm].get(s, 0.0) for s in server]
    bx.barh(range(len(server)), vals,
            color=[STAGE_COLOR[s] for s in server], height=0.6)
    bx.set_xscale("log")
    bx.set_yticks(range(len(server)))
    bx.set_yticklabels(server)
    bx.set_ylim(len(server) - 0.5, -0.5)
    bx.set_xlabel("seconds (log)")
    bx.set_title("Server side, PRESLEY arm")
    for i, v in enumerate(vals):
        if v > 0:
            bx.text(v * 1.25, i, f"{v:.3f}", va="center", fontsize=6)

    fig.tight_layout()

    per_arm = {a: {s: round(v, 3) for s, v in means[a].items()} for a in arms}
    wall = {a: round(st.fmean([r["wall_mean_s"] for r in rows if r["arm"] == a]), 2)
            for a in arms}
    restore_share = {a: round(100 * means[a].get("restore", 0.0) / wall[a], 1)
                     for a in arms if wall[a] > 0}

    sentence = (
        "Stacked horizontal bars, one per pipeline arm, showing mean seconds per "
        "stage over three trials on a pinned GPU. Restoration is the dominant "
        "stage in every arm that has one, taking between 70 and 89 percent of "
        "wall clock; ProPainter costs about three times Real-ESRGAN. Degradation "
        "costs more than encoding, so the server-side bottleneck is the per-block "
        "loop rather than the codec. A log-scale panel shows the server-side "
        "stages, where block selection costs about one hundredth of a second -- "
        "roughly a thousandth of the pipeline, and the axis on which the article "
        "locates its remaining headroom.")

    data = {
        "units": "seconds, mean of 3 trials, 640x360, 60-90 frames",
        "device": sorted({r["device"] for r in rows}),
        "trials_per_cell": sorted({r["trials"] for r in rows}),
        "videos": sorted({r["video"] for r in rows}),
        "stage_order": STAGES,
        "mean_seconds_by_arm": per_arm,
        "wall_seconds_by_arm": wall,
        "restoration_share_pct": restore_share,
        "method": "components called directly into scratch dirs; nothing written "
                  "under results/; device recorded per trial; cells mixing device "
                  "populations refused rather than averaged",
        "caveat": "Real-ESRGAN here runs full-frame (restorer_params {}), which is "
                  "faster than the tiled tile=400 configuration the published "
                  "corpus uses; these timings are not comparable to timings "
                  "quoted for that configuration",
    }

    paths = figkit.emit("stage_timing", fig, sentence, data,
                        figures_dir=args.figures_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
