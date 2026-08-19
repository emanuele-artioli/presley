"""PRESLEY against ELVIS at matched rate — `fig:ratematched`.

One bar per rate ladder, BD-rate with ELVIS as the anchor, so a bar left of
zero means PRESLEY reaches the same quality for fewer bits. The background
panel is negative on every ladder; the foreground panel moves in both
directions, which is the weaker claim the article actually makes -- no
*systematic* foreground cost, rather than an unchanged foreground.

Authored at half textwidth so it can pair with another figure.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figure_style import HALF, despine, use_paper_style, width_in  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1] / "68e8b6bb11d0dd9e62a67aef" / "Figures"
D = json.loads((HERE / "ratematched.json").read_text())["data"]

use_paper_style()

def short(name: str) -> str:
    return name.split("/")[-1] if "/" in name else name

order = sorted(range(D["n_ladders"]), key=lambda i: D["bd_rate_bg_lpips_pct"][i])
labels = [short(D["ladders"][i]) for i in order]
bg = [D["bd_rate_bg_lpips_pct"][i] for i in order]
fg = [D["bd_rate_fg_lpips_pct"][i] for i in order]
y = range(len(order))

fig, (axb, axf) = plt.subplots(1, 2, figsize=(width_in(HALF), 1.78), sharey=True)

axb.barh(list(y), bg, color="#4878a8", edgecolor="black", linewidth=0.4)
axb.axvline(0, color="black", lw=0.7)
axb.set_xlabel("background", labelpad=2)
axb.set_title(f"{D['bg_wins']}/{D['n_ladders']} negative", fontsize=7, pad=3)

axf.barh(list(y), fg, color=["#7ba05b" if v < 0 else "#c0724a" for v in fg],
         edgecolor="black", linewidth=0.4)
axf.axvline(0, color="black", lw=0.7)
axf.axvspan(-15, 15, color="0.88", zorder=0)
axf.set_xlabel("foreground", labelpad=2)
axf.set_title("both directions", fontsize=7, pad=3)

axb.set_yticks(list(y))
axb.set_yticklabels(labels, fontsize=6.2)
for ax in (axb, axf):
    despine(ax)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=6.2)
fig.supxlabel("BD-rate vs ELVIS (%), negative = PRESLEY cheaper",
              fontsize=6.8, y=0.015)
fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0.055, 1, 1))
fig.savefig(HERE / "ratematched.pdf", format="pdf")
print("wrote", HERE / "ratematched.pdf")
