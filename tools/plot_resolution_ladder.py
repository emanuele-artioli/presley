"""Does the saving survive at higher resolution? — `fig:resolution-ladder`.

The stored JSON for this figure held only prose; the per-clip BD-rates live in
the figure's own accessibility description, which is where this script reads
them from and then writes them back into the JSON so the data has one home.

`tools/analyze_resolution_ladder.py`, named in CLAIM(fig:resolution-ladder) as
the reproduction path, is NOT in the repository, and two of its four run-config
files are missing. So this cannot re-derive the numbers from `results/` the way
the restorability figure does; it redraws exactly what was published and checks
the per-resolution medians, win counts and sign-test p-values against the CLAIM
anchor, refusing to emit if any of them disagree.

Authored at full textwidth, the width it is included at.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figure_style import FULL, despine, use_paper_style, width_in  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1] / "68e8b6bb11d0dd9e62a67aef" / "Figures"

# Values as landed in CLAIM(fig:resolution-ladder); the figure must match them.
PUBLISHED = {
    "640x360":   {"wins": 3, "n": 8, "median_pct": 2.8,  "sign_p": 0.7266},
    "1280x720":  {"wins": 7, "n": 7, "median_pct": -5.3, "sign_p": 0.0156},
    "1920x1080": {"wins": 6, "n": 7, "median_pct": -4.3, "sign_p": 0.125},
}
ORDER = ["640x360", "1280x720", "1920x1080"]


def data_from_description():
    txt = (HERE / "resolution_ladder.desc.tex").read_text()
    start = txt.index("Data: {") + len("Data: ")
    depth, end = 0, start
    for i in range(start, len(txt)):        # brace-match: the alt text wraps it
        if txt[i] == "{":
            depth += 1
        elif txt[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    blob = txt[start:end].replace("\\_", "_").replace("\\%", "%")
    return json.loads(blob)


def main() -> int:
    d = data_from_description()
    by = d["by_resolution"]

    drift = []
    for res, pub in PUBLISHED.items():
        got = by[res]
        if got["wins"] != pub["wins"] or got["n"] != pub["n"]:
            drift.append(f"{res}: {got['wins']}/{got['n']}, published "
                         f"{pub['wins']}/{pub['n']}")
        if abs(got["median_pct"] - pub["median_pct"]) > 0.05:
            drift.append(f"{res}: median {got['median_pct']}, published "
                         f"{pub['median_pct']}")
        if abs(got["sign_p"] - pub["sign_p"]) > 5e-4:
            drift.append(f"{res}: p {got['sign_p']}, published {pub['sign_p']}")
    if drift:
        print("ABORT — figure disagrees with CLAIM(fig:resolution-ladder):")
        print("\n".join("  " + x for x in drift))
        return 1
    print("reproduced every published median, win count and p-value")

    # give the data one home rather than leaving it only in the alt text
    jf = HERE / "resolution_ladder.json"
    j = json.loads(jf.read_text())
    j.setdefault("data", {}).update({k: d[k] for k in
                                     ("by_resolution", "metric", "sign_convention",
                                      "arm", "baseline", "caveats",
                                      "block_grid_held_constant")})
    jf.write_text(json.dumps(j, indent=1, sort_keys=True) + "\n")

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(width_in(FULL), 1.56), sharex=True)
    for ax, res in zip(axes, ORDER):
        g = by[res]
        pairs = sorted(zip(g["bd_rate_pct"], g["videos"]))
        vals = [p[0] for p in pairs]
        labs = [p[1] for p in pairs]
        y = range(len(vals))
        ax.barh(list(y), vals, height=0.7, edgecolor="black", linewidth=0.3,
                color=["#2e7d5b" if v < 0 else "#b5443a" for v in vals])
        ax.axvline(0, color="black", lw=0.7)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labs, fontsize=6.8)
        ax.invert_yaxis()
        star = "*" if g["sign_p"] < 0.05 else ""
        ax.set_title(f"{res}\n{g['wins']}/{g['n']} save, median "
                     f"{g['median_pct']:+.1f}%{star}",
                     fontsize=7.0, pad=3, linespacing=1.05)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.tick_params(axis="y", length=0)
        despine(ax)
    axes[1].set_xlabel("BD-rate on foreground LPIPS (%), negative = fewer bits",
                       fontsize=7.0, labelpad=1)
    fig.tight_layout(pad=0.3, w_pad=1.4)
    fig.savefig(HERE / "resolution_ladder.pdf", format="pdf")
    print("wrote", HERE / "resolution_ladder.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
