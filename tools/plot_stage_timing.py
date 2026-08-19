"""Cost per pipeline stage — `fig:stage-timing`.

Stacked horizontal bars, one per arm, over the stage order the pipeline runs
in. Restoration dominates every arm that has one; the inset repeats the
server-side stages on a log axis, because block selection -- the axis this
article locates its headroom on -- is three orders below restoration and
vanishes on a linear scale.

Drawn at half-textwidth so it can sit beside another figure, and authored at
that width so its labels are not scaled down on inclusion.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figure_style import HALF, despine, use_paper_style, width_in  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1] / "68e8b6bb11d0dd9e62a67aef" / "Figures"
D = json.loads((HERE / "stage_timing.json").read_text())["data"]

use_paper_style()

SHORT = {
    "baseline (SVT-AV1)": "baseline",
    "elvis blackout + ProPainter": "ELVIS blackout\n+ ProPainter",
    "presley downsample + Real-ESRGAN": "PRESLEY downsample\n+ Real-ESRGAN",
    "presley freeze + ProPainter": "PRESLEY freeze\n+ ProPainter",
}
COLOURS = {
    "preprocess": "#b9c6d4", "score": "#8fa8c0", "select": "#6d8fae",
    "degrade": "#4878a8", "encode": "#2f5d86", "sidechannel": "#9ecae1",
    "decode": "#c7c7c7", "restore": "#d1603d", "composite": "#a8a8a8",
}
arms = list(SHORT)
stages = D["stage_order"]

fig, ax = plt.subplots(figsize=(width_in(HALF), 1.66))
y = range(len(arms))
left = [0.0] * len(arms)
for st in stages:
    vals = [D["mean_seconds_by_arm"][a].get(st, 0.0) for a in arms]
    if not any(vals):
        continue
    ax.barh(list(y), vals, left=left, color=COLOURS[st], edgecolor="white",
            linewidth=0.4, label=st)
    left = [l + v for l, v in zip(left, vals)]

for i, a in enumerate(arms):
    share = D["restoration_share_pct"][a]
    if share:
        ax.annotate(f"{share:.0f}% restore", xy=(left[i], i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.5,
                    color="0.25")

ax.set_yticks(list(y))
ax.set_yticklabels([SHORT[a] for a in arms], fontsize=6.5, linespacing=0.95)
ax.set_xlabel("seconds (mean of 3 trials)")
ax.set_xlim(0, max(left) * 1.46)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
          fontsize=6.2, handlelength=0.9, columnspacing=0.8,
          handletextpad=0.35, borderpad=0.1, labelspacing=0.25)
despine(ax)
fig.tight_layout(pad=0.25)
fig.savefig(HERE / "stage_timing.pdf", format="pdf")
print("wrote", HERE / "stage_timing.pdf")
