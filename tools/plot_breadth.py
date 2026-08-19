"""Bitrate against the pristine baseline on every held-out clip — `fig:breadth`.

One bar per clip per arm, grouped by dataset family, because the article's
content-dependence claim is that the sign splits *within* families rather than
between them -- a win rate would hide exactly that. Crossed bars saved bits
while their foreground quality moved against the baseline, so the saving is
not free.

Authored at full textwidth. The clip roster is the point of the figure, so it
is dense by nature; labels are 7pt, between the 8pt of the other figures and
the 9pt caption, which is the smallest this roster fits at.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figure_style import FULL, despine, use_paper_style, width_in  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1] / "68e8b6bb11d0dd9e62a67aef" / "Figures"
ARMS = json.loads((HERE / "breadth.json").read_text())["data"]["arms"]

use_paper_style()

FAMILY_OF = lambda v: ("MOSEv2" if v.startswith("mosev2/")
                       else "YouTube-VOS" if v.startswith("youtube_vos/")
                       else "DAVIS")
ORDER = ["DAVIS", "MOSEv2", "YouTube-VOS"]
C_SAVE, C_COST = "#2e7d5b", "#b5443a"


def rows_for(arm_key):
    """Clips in family order, sorted within family, plus the family breaks."""
    arm = ARMS[arm_key]
    rows, seps, at = [], [], 0
    for fam in ORDER:
        clips = [c for c in arm["per_clip"] if FAMILY_OF(c["video"]) == fam]
        clips.sort(key=lambda c: c["bitrate_delta_pct"])
        if not clips:
            continue
        if at:
            seps.append(at - 0.5)
        rows.extend(clips)
        at += len(clips)
    return rows, seps


def draw(ax, rows, seps, flagged, title, xlim):
    y = range(len(rows))
    vals = [c["bitrate_delta_pct"] for c in rows]
    ax.barh(list(y), vals, color=[C_SAVE if v < 0 else C_COST for v in vals],
            edgecolor="black", linewidth=0.3, height=0.72)
    for i, c in enumerate(rows):
        if c["video"] in flagged:
            ax.plot(0, i, marker="x", ms=2.6, mew=0.8, color="black", zorder=4)
    for sp in seps:
        ax.axhline(sp, color="0.45", lw=0.5, ls=(0, (2, 2)))
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels([c["video"].split("/")[-1] for c in rows], fontsize=6.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(*xlim)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=7.0, pad=3, linespacing=1.05)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.tick_params(axis="y", length=0)
    despine(ax)


bridge_rows, bridge_seps = rows_for("bridge (ELVIS blackout)")
pres_rows, pres_seps = rows_for("PRESLEY (downsample)")
bridge = ARMS["bridge (ELVIS blackout)"]
pres = ARMS["PRESLEY (downsample)"]

# The bridge arm's 33 clips flow across two columns so the float is half as
# tall; the split falls inside DAVIS, and the family rules stay where they land.
cut = 12
cols = [bridge_rows[0:cut], bridge_rows[cut:2 * cut], bridge_rows[2 * cut:]]
seps = [[x - i * cut for x in bridge_seps
         if i * cut - 0.5 < x < (i + 1) * cut - 0.5] for i in range(3)]

allv = [c["bitrate_delta_pct"] for c in bridge_rows + pres_rows]
xlim = (min(allv) * 1.08, max(allv) * 1.12)

fig, axes = plt.subplots(1, 4, figsize=(width_in(FULL), 0.100 * cut + 0.68))
titles = [f"bridge (ELVIS blackout)\n{bridge['clips_saving_bits']}/{bridge['n_clips']} save bits",
          "bridge, continued", "bridge, continued"]
for ax, rows, sp, t in zip(axes, cols, seps, titles):
    draw(ax, rows, sp, set(bridge["saving_with_foreground_cost"]), t, xlim)
draw(axes[3], pres_rows, pres_seps, set(pres["saving_with_foreground_cost"]),
     f"PRESLEY\n{pres['clips_saving_bits']}/{pres['n_clips']} save bits", xlim)
axes[1].set_xlabel("bitrate vs pristine baseline (%)", fontsize=7.5, labelpad=1)
fig.text(0.5, 0.008, "green: saves bits   \u00b7   red: costs bits   \u00b7   "
         "\u2715: saved bits but foreground quality moved against the baseline",
         ha="center", fontsize=6.5)
fig.tight_layout(pad=0.25, w_pad=1.0, rect=(0, 0.05, 1, 1))
fig.savefig(HERE / "breadth.pdf", format="pdf")
print("wrote", HERE / "breadth.pdf")
