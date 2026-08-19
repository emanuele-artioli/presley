"""Shared style for the manuscript's figures.

The rule this module exists to enforce: **a figure is authored at the width it
is rendered at.** Every figure in this article was originally drawn on a ~6.8in
canvas and included at ~3.3in, which scales its labels to under 5pt against
10pt body text. Nothing about the plotting was wrong; the mismatch was.

So each generator declares the LaTeX width it will be included at, converts
that to inches here, and draws on exactly that canvas. Fonts are then set in
real points and survive inclusion unscaled.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless host: never a GUI backend
import matplotlib.pyplot as plt  # noqa: E402

# acmart `manuscript` text block, measured from the rendered PDF.
TEXTWIDTH_PT = 469.0
PT_PER_IN = 72.27

#: body text renders at ~10.2pt and captions at ~9pt, so 8pt figure text reads
#: as "about the same size as its description" without dominating it.
FONT_PT = 8.0

FULL = 1.00   # \includegraphics[width=\textwidth]
HALF = 0.48   # two panels side by side in one float


def width_in(frac: float) -> float:
    """Canvas width, in inches, for a figure included at `frac`\\textwidth."""
    return frac * TEXTWIDTH_PT / PT_PER_IN


def use_paper_style(font_pt: float = FONT_PT) -> None:
    plt.rcParams.update({
        "font.size": font_pt,
        "axes.labelsize": font_pt,
        "axes.titlesize": font_pt,
        "xtick.labelsize": font_pt - 0.5,
        "ytick.labelsize": font_pt - 0.5,
        "legend.fontsize": font_pt - 0.5,
        "figure.titlesize": font_pt,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": None,     # honour the declared canvas exactly
        "pdf.fonttype": 42,       # embed TrueType, not Type 3
    })


def despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
