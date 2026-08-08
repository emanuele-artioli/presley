"""Every figure emits three artefacts from one run. This is the shared plumbing.

    Figures/<name>.pdf        the figure
    Figures/<name>.json       the underlying values
    Figures/<name>.desc.tex   a \\Description{} block for \\input

The third one is the point, and it is worth explaining because it looks like
redundancy and is not.

ACM requires a `\\Description{}` on every float. It is alt text: it does not
render on the page, it is what a screen reader speaks, and it is machine
readable. So it is the one place in a LaTeX document where a figure's
*underlying numbers* can travel with the figure at zero cost to the page
budget -- which matters when the article is 23 pages against a 46-page draft.

The block is written plain-sentence-first, then compact JSON. That order is
deliberate: a screen reader reaching a wall of raw JSON before any statement of
what the figure shows is worse for a human than no description at all. The
sentence is for the reader; the JSON is for anyone -- person or program --
who wants the values without digging out the source data.

The JSON sidecar is the same payload as a standalone file, so a CI check can
regenerate a figure and diff its data against what the manuscript ships. A
figure that silently stops matching `results/` is the failure mode this exists
to prevent; `tests/test_figure_data.py` is the check.

Conventions enforced here rather than left to each script:

* Agg backend, always -- the host is headless and a GUI backend fails at
  import time, halfway through a campaign.
* One consistent style, so eight figures look like one paper.
* Vector PDF, no rasterized text.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless host: never a GUI backend
import matplotlib.pyplot as plt  # noqa: E402

# The manuscript is a SEPARATE git repo, gitignored from this one, and it exists
# only in the main checkout -- never in a worktree. Resolving it relative to
# this file therefore silently creates a stray Figures/ inside whatever worktree
# a script happens to run from, and the manuscript never sees the figure. So the
# location is explicit: PRESLEY_PAPER_DIR, or the --figures-dir a script passes.
PAPER = pathlib.Path(
    os.environ.get("PRESLEY_PAPER_DIR",
                   pathlib.Path(__file__).resolve().parents[1] / "68e8b6bb11d0dd9e62a67aef")
)
FIGURES = PAPER / "Figures"

# One column of the ACM two-column layout, in inches, so figures are placed at
# their natural size and text is never scaled by \includegraphics.
COLUMN_WIDTH = 3.33
FULL_WIDTH = 7.0

_STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,      # embed TrueType: no rasterized text
}

# Colour-blind safe, and distinguishable in greyscale print.
COLORS = {
    "win": "#1b7837",
    "loss": "#c2545b",
    "neutral": "#4a4a4a",
    "accent": "#3b6ea5",
    "muted": "#9a9a9a",
}


def style():
    """Apply the shared style. Call once at the top of a plot script."""
    plt.rcParams.update(_STYLE)


def _describe_block(sentence: str, data: Dict[str, Any]) -> str:
    r"""The `\Description{}` body: plain sentence, then compact JSON.

    LaTeX-special characters in the JSON are escaped. `%` in particular would
    comment out the rest of the line and silently truncate the description --
    and since the block never renders on the page, nobody would see it happen.
    """
    compact = json.dumps(data, separators=(",", ":"), sort_keys=True)
    for char, escaped in (("\\", "\\textbackslash "), ("%", "\\%"), ("&", "\\&"),
                          ("#", "\\#"), ("_", "\\_"), ("$", "\\$"),
                          ("^", "\\^{}"), ("~", "\\~{}")):
        compact = compact.replace(char, escaped)
    return f"\\Description{{{sentence} Data: {compact}}}"


def emit(name: str, fig, sentence: str, data: Dict[str, Any],
         figures_dir: Optional[pathlib.Path] = None) -> Dict[str, pathlib.Path]:
    """Write the figure, its data, and its description. Returns the paths.

    `sentence` states what the figure shows, in words, for a reader who cannot
    see it. `data` is the values behind it -- semantic keys, flat arrays, no
    nesting deeper than it needs.
    """
    out = pathlib.Path(figures_dir) if figures_dir else FIGURES
    out.mkdir(parents=True, exist_ok=True)

    pdf = out / f"{name}.pdf"
    fig.savefig(pdf)
    plt.close(fig)

    payload = {"figure": name, "sentence": sentence, "data": data}
    js = out / f"{name}.json"
    js.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    desc = out / f"{name}.desc.tex"
    desc.write_text(_describe_block(sentence, data) + "\n")

    return {"pdf": pdf, "json": js, "desc": desc}


def load_data(name: str, figures_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """The committed sidecar for a figure, for the CI freshness check."""
    out = pathlib.Path(figures_dir) if figures_dir else FIGURES
    return json.loads((out / f"{name}.json").read_text())
