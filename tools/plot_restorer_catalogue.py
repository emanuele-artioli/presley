"""Every restorer, against its own matched unrestored control — `fig:restorers`.

Two panels that disagree on purpose. *Left:* gain over each arm's own control,
where hole-filling leads because it destroys more to begin with. *Right:* the
absolute background LPIPS a viewer receives, where the ordering inverts. The
article's verdict rests on the right panel and the diagnosis on the left, so
both have to be visible at once.

Authored at full textwidth, which is the width it is included at.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figure_style import FULL, despine, use_paper_style, width_in  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1] / "68e8b6bb11d0dd9e62a67aef" / "Figures"
ARMS = json.loads((HERE / "restorer_catalogue.json").read_text())["data"]["arms"]

use_paper_style()

COND = "conditioned (keeps the prior)"
C_COND, C_PAINT = "#4878a8", "#d1603d"

arms = sorted(ARMS, key=lambda a: a["median_absolute_bg_lpips"])
labels = [f"{a['transport']}+{a['restorer']}" for a in arms]
colours = [C_COND if a["family"] == COND else C_PAINT for a in arms]
y = range(len(arms))

fig, (axl, axr) = plt.subplots(1, 2, figsize=(width_in(FULL), 1.82), sharey=True)

gains = [a["median_gain_over_control"] for a in arms]
lo = [a["median_gain_over_control"] - a["min_gain"] for a in arms]
hi = [a["max_gain"] - a["median_gain_over_control"] for a in arms]
axl.barh(list(y), gains, color=colours, edgecolor="black", linewidth=0.4)
axl.errorbar(gains, list(y), xerr=[lo, hi], fmt="none", ecolor="0.35",
             elinewidth=0.6, capsize=1.5)
axl.axvline(0, color="black", lw=0.7)
axl.set_xlabel("gain over own control (BG-LPIPS)")
axl.set_title("recovers more", fontsize=7.5, pad=3)

absol = [a["median_absolute_bg_lpips"] for a in arms]
axr.barh(list(y), absol, color=colours, edgecolor="black", linewidth=0.4)
axr.set_xlabel("absolute BG-LPIPS delivered")
axr.set_title("delivers better (lower is better)", fontsize=7.5, pad=3)

axl.set_yticks(list(y))
axl.set_yticklabels(labels, fontsize=6.5)
for ax in (axl, axr):
    despine(ax)
    ax.tick_params(axis="y", length=0)

handles = [plt.Rectangle((0, 0), 1, 1, fc=C_COND, ec="black", lw=0.4),
           plt.Rectangle((0, 0), 1, 1, fc=C_PAINT, ec="black", lw=0.4)]
fig.legend(handles, ["conditioned (keeps the prior)", "in-painting (discards it)"],
           loc="lower center", ncol=2, fontsize=7, handlelength=1.1,
           bbox_to_anchor=(0.5, -0.02), columnspacing=1.2)
for ax in (axl, axr):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
fig.tight_layout(pad=0.3, w_pad=1.6, rect=(0, 0.06, 1, 1))
fig.savefig(HERE / "restorer_catalogue.pdf", format="pdf")
print("wrote", HERE / "restorer_catalogue.pdf")
