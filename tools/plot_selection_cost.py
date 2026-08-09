#!/usr/bin/env python3
"""The selection score against a true bit oracle -- `fig:selection-cost`.

The positive half of the article's selection story, and the half that is
currently buried inside a paragraph about what is missing. On the axis it
actually models -- how many bits a block costs -- the score is close to
optimal, and the figure's job is to make the remaining headroom legible as an
area rather than as a sentence.

Per video: what fraction of the bits an exact leave-one-superblock-out oracle
would free is freed by taking the top quarter by score, against what random
selection of the same number already frees. That null is the whole point. A
capture ratio without it reads as "83% of the way to the oracle", which is not
what it means: random already scores 0.402, so the score's real contribution is
the gap above the null, not the height of the bar.

The figure also has to carry the breach honestly. drift-straight captures 0.510
against its own null of 0.436 -- barely above chance -- and its rank correlation
is +0.082 with a bootstrap interval containing zero. It is drawn in the loss
colour rather than dropped or averaged away, because the mean cannot be quoted
without it.

Usage:
    python tools/plot_selection_cost.py --data-root .
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import figkit  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Below this margin over its own null the proxy is not distinguishable from
# picking superblocks at random on that clip.
NEAR_CHANCE = 0.10


def parse_means(text: str):
    """The tool's OWN summary means, not a re-average of its printed rows.

    Averaging the per-video values as printed gives 0.834 where the tool
    reports 0.833 -- it rounds each row to three places before printing. A
    thousandth is nothing numerically and everything for provenance: the
    article cites 0.833, and a figure quietly showing 0.834 is the first step
    of a chain that no longer traces.
    """
    means = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("mean capture (SC)"):
            means["capture"] = float(s.split()[3])
        elif s.startswith("mean random null"):
            means["null"] = float(s.split()[3])
        elif s.startswith("mean oracle share"):
            means["oracle_share"] = float(s.split()[3])
    return means


def parse(text: str):
    rows, started = [], False
    for line in text.splitlines():
        if line.startswith("video "):
            started = True
            continue
        if not started:
            continue
        parts = line.split()
        if len(parts) != 9:
            if rows:
                break
            continue
        try:
            rows.append({"video": parts[0], "rho": float(parts[1]),
                         "capture": float(parts[3]), "random": float(parts[5]),
                         "oracle_share": float(parts[7])})
        except ValueError:
            if rows:
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--figures-dir", default=None)
    args = ap.parse_args()

    tool = REPO_ROOT / "tools" / "analyze_f1_oracle.py"
    proc = subprocess.run([sys.executable, str(tool), "--results-dir",
                           str(pathlib.Path(args.data_root) / "results")],
                          capture_output=True, text=True, cwd=args.data_root)
    rows = parse(proc.stdout)
    means = parse_means(proc.stdout)
    if len(rows) != 8 or len(means) != 3:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"expected 8 probe videos, parsed {len(rows)}")

    rows.sort(key=lambda r: r["capture"] - r["random"])
    names = [r["video"] for r in rows]
    cap = np.array([r["capture"] for r in rows])
    null = np.array([r["random"] for r in rows])
    gain = cap - null

    figkit.style()
    fig, ax = plt.subplots(figsize=(figkit.FULL_WIDTH, 2.4))
    y = np.arange(len(rows))

    # The null as the bar's base, the score's contribution stacked on it, and
    # the remainder to 1.0 left open as the headroom any better cost model
    # could compete for.
    ax.barh(y, null, color=figkit.COLORS["muted"], height=0.62,
            label="random selection (the null)")
    colors = [figkit.COLORS["loss"] if g < NEAR_CHANCE else figkit.COLORS["win"]
              for g in gain]
    ax.barh(y, gain, left=null, color=colors, height=0.62,
            label="added by the complexity score")
    ax.barh(y, 1.0 - cap, left=cap, color="white", edgecolor=figkit.COLORS["neutral"],
            height=0.62, linewidth=0.5, hatch="///",
            label="headroom left for a better cost model")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of the oracle's recoverable bits captured at the top 25%")
    ax.set_title("On the axis it models, the score is close to a bit oracle "
                 f"(mean {means['capture']:.3f} vs a {means['null']:.3f} null)")
    ax.legend(fontsize=6, loc="lower right", framealpha=0.95)
    fig.tight_layout()

    near_chance = [r["video"] for r, g in zip(rows, gain) if g < NEAR_CHANCE]

    sentence = (
        "Horizontal stacked bars, one per probe video, showing what fraction of "
        "an exact bit oracle's recoverable bits the complexity score captures when "
        "the top quarter of superblocks is degraded. Each bar starts with the grey "
        "fraction that random selection already captures, then adds the score's "
        f"contribution, then leaves the hatched remainder open. The mean capture is "
        f"{means['capture']:.3f} against a random null of {means['null']:.3f}, so the score "
        "adds about four tenths over chance and the cost axis has little headroom "
        f"left. On {', '.join(near_chance)} the score is near chance and is drawn in "
        "the contrasting colour rather than averaged away.")

    data = {
        "quantity": "capture ratio: fraction of an exact leave-one-superblock-out "
                    "oracle's freed bits obtained by degrading the top 25% by score",
        "coding": "inter, SVT-AV1 fixed QP 43, 64x64 superblocks, n=8 probe videos",
        "videos": names,
        "capture": [round(float(x), 3) for x in cap],
        "random_null": [round(float(x), 3) for x in null],
        "gain_over_null": [round(float(x), 3) for x in gain],
        "mean_capture": means["capture"],
        "mean_random_null": means["null"],
        "near_chance_videos": near_chance,
        "why_the_null_matters": "a capture ratio without its null reads as "
                                "'83% of the way to the oracle', which it is not: "
                                "random selection already captures 0.402",
        "oracle_top25_share_of_total_bits": means["oracle_share"],
        "headroom_reading": "the oracle's top quarter is 30.1% of total bitrate, so "
                            "the entire remaining cost-side headroom is worth about "
                            "5% of the bitrate",
    }

    paths = figkit.emit("selection_cost", fig, sentence, data, figures_dir=args.figures_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
