"""The removability score against a bit oracle — `fig:selection-axes`, top panel.

Authored at the width it is *rendered* at. Every other figure in this article
was drawn on a ~6.8in canvas and then included at ~3.3in, which scales its 8pt
labels down to under 5pt against 10pt body text. A figure is legible exactly
when its authored width equals its rendered width, so this script takes the
target column width as its canvas and sets fonts in points that survive
unscaled.

Data is read from the JSON emitted alongside the original figure, so the plot
cannot drift from the numbers the CLAIM lines cite.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1] / "68e8b6bb11d0dd9e62a67aef" / "Figures"
DATA = json.loads((HERE / "selection_cost.json").read_text())
D = DATA.get("data", DATA)

# Rendered at 0.48\textwidth on a 469pt text block => 225pt => 3.12in.
WIDTH_IN = 3.12
FS = 8.0

plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
    "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

vids = D["videos"]
null = D["random_null"]
cap = D["capture"]
order = sorted(range(len(vids)), key=lambda i: cap[i])
vids = [vids[i] for i in order]
null = [null[i] for i in order]
cap = [cap[i] for i in order]

fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.68))
y = range(len(vids))
ax.barh(y, null, color="0.75", edgecolor="black", linewidth=0.5,
        label="random (null)")
ax.barh(y, [c - n for c, n in zip(cap, null)], left=null, color="#4878a8",
        edgecolor="black", linewidth=0.5, label="score gain")
ax.barh(y, [1 - c for c in cap], left=cap, color="white", edgecolor="black",
        linewidth=0.5, hatch="///", label="headroom")

ax.axvline(D["mean_capture"], color="crimson", lw=1.0, ls="--", zorder=5)
ax.annotate(f"mean {D['mean_capture']:.3f}", xy=(D["mean_capture"], -0.95),
            xytext=(-3, 0), textcoords="offset points", ha="right", va="center",
            fontsize=FS - 1, color="crimson")
ax.set_ylim(-1.4, len(vids) - 0.4)

ax.set_yticks(list(y))
ax.set_yticklabels(vids)
ax.set_xlim(0, 1.0)
ax.set_xlabel("fraction of the oracle's freed bits")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3,
          handlelength=1.0, borderpad=0.1, columnspacing=0.8,
          handletextpad=0.4, fontsize=FS - 1.0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.25)
fig.savefig(HERE / "selection_cost.pdf", format="pdf")
print("wrote", HERE / "selection_cost.pdf", f"at {WIDTH_IN}in")
