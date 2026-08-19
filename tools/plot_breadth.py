"""Bitrate against the pristine baseline on a matched corpus — `fig:breadth`.

One row per clip, two bars: ELVIS and PRESLEY on the same video at the same
QPs, against the same baseline. Grouping them per clip is the point --- the
article's content-dependence claim is that the sign splits *within* dataset
families rather than between them, and that only reads if both configurations
are shown on one corpus.

Authored at half textwidth so it pairs with fig:ratematched.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figure_style import HALF, despine, use_paper_style, width_in  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1] / "68e8b6bb11d0dd9e62a67aef" / "Figures"
D = json.loads((HERE / "breadth.json").read_text())["data"]

use_paper_style()
ORDER = ["DAVIS", "MOSEv2 (held out)", "YouTube-VOS (held out)"]
E, P = "ELVIS (block removal)", "PRESLEY (downsample)"

eb = {p["video"]: p for p in D["arms"][E]["per_clip"]}
pb = {p["video"]: p for p in D["arms"][P]["per_clip"]}

rows, seps, at = [], [], 0
for fam in ORDER:
    vs = sorted((v for v in eb if eb[v]["family"] == fam),
                key=lambda v: pb[v]["bitrate_delta_pct"])
    if not vs:
        continue
    if at:
        seps.append(at - 0.5)
    rows += vs
    at += len(vs)

fig, ax = plt.subplots(figsize=(width_in(HALF), 0.108 * len(rows) + 0.55))
h = 0.38
for k, (src, colour, lab) in enumerate(
        ((eb, "#b5443a", "ELVIS"), (pb, "#2e7d5b", "PRESLEY"))):
    ax.barh([i + (h / 2 if k else -h / 2) for i in range(len(rows))],
            [src[v]["bitrate_delta_pct"] for v in rows], height=h,
            color=colour, edgecolor="black", linewidth=0.3, label=lab)
for s in seps:
    ax.axhline(s, color="0.45", lw=0.5, ls=(0, (2, 2)))
ax.axvline(0, color="black", lw=0.7)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([v.split("/")[-1] for v in rows], fontsize=6.2)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.invert_yaxis()
ax.set_xlabel("bitrate vs pristine baseline (%)", fontsize=7.0, labelpad=1)
ax.tick_params(axis="x", labelsize=6.3)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.8,
          handlelength=1.0, columnspacing=1.2, borderpad=0.1)
despine(ax)
fig.tight_layout(pad=0.25)
fig.savefig(HERE / "breadth.pdf", format="pdf")
print("wrote", HERE / "breadth.pdf", f"({len(rows)} clips, both configurations)")
