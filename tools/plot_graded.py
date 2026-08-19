"""Graded multi-level against binary strength, per sequence — `fig:graded`.

The retirement of graded strength rests on a sign count, and a sign count
without its distribution hides both how large the losses are and that two
sequences go the other way. Both belong in the figure: the retirement is of
the default, not of the mechanism.

Values are re-derived by tools/analyze_w2b_graded.py; this script refuses to
draw if the count or median disagrees with what the article states.
"""
from __future__ import annotations

import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figure_style import HALF, despine, use_paper_style, width_in  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1] / "68e8b6bb11d0dd9e62a67aef" / "Figures"

# BD-rate of graded against binary on background LPIPS; positive = graded costs
# more bits for the same quality. From analyze_w2b_graded.py, b=64 at 1920x1080.
DATA = [("bear", 53.5), ("bike-packing", 8.2), ("bmx-bumps", 1.9),
        ("breakdance", 2.9), ("color-run", 0.7), ("dancing", 8.4),
        ("dog", 7.2), ("dogs-jump", 46.0), ("drift-turn", 1.4),
        ("india", 1.8), ("lindy-hop", 2.6), ("motorbike", -4.2),
        ("pigs", 54.0), ("scooter-black", 47.0), ("tennis", -11.1),
        ("train", 68.3)]
PUBLISHED_WORSE, PUBLISHED_N, PUBLISHED_MEDIAN = 14, 16, 5.1


def main() -> int:
    worse = sum(1 for _, v in DATA if v > 0)
    med = statistics.median(v for _, v in DATA)
    if (worse, len(DATA)) != (PUBLISHED_WORSE, PUBLISHED_N) or abs(med - PUBLISHED_MEDIAN) > 0.05:
        print(f"ABORT — {worse}/{len(DATA)} median {med:+.1f}, "
              f"published {PUBLISHED_WORSE}/{PUBLISHED_N} median {PUBLISHED_MEDIAN:+.1f}")
        return 1
    print(f"reproduced: graded worse on {worse}/{len(DATA)}, median {med:+.1f}%")

    use_paper_style()
    rows = sorted(DATA, key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(width_in(HALF), 0.076 * len(rows) + 0.46))
    ax.barh(range(len(rows)), [v for _, v in rows],
            color=["#2e7d5b" if v < 0 else "#b5443a" for _, v in rows],
            edgecolor="black", linewidth=0.3, height=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.axvline(med, color="#B00020", lw=0.9, ls="--")
    ax.annotate(f"median {PUBLISHED_MEDIAN:+.1f}%", xy=(med, -0.6), xytext=(3, 0),
                textcoords="offset points", fontsize=6.2, color="#B00020", va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([v for v, _ in rows], fontsize=6.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("BD-rate, graded vs binary (%)", fontsize=7.0, labelpad=1)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.tick_params(axis="y", length=0)
    despine(ax)
    fig.tight_layout(pad=0.25)
    fig.savefig(HERE / "graded.pdf", format="pdf")
    print("wrote", HERE / "graded.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
