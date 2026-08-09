#!/usr/bin/env python3
"""Bitrate saving against resolution -- `fig:resolution-ladder`.

Replaces a table nobody would read: 6 videos x 3 resolutions x 2 metrics is 36
BD-rate numbers, and the question it answers is a shape question -- does the
saving survive as resolution rises, and is it the same story on every clip?

Drawn as one panel per resolution, one bar per video, sorted, with the zero
line marked. Bars below zero are wins.

Two things the figure has to carry honestly, both in the caption and in the
machine-readable description:

* the 360p rung splits 3/6 and supports no claim -- and the two clips that
  reverse are dog and pigs, the same two the article's AV1 breadth result
  already identifies, so the split is a reproduction rather than noise;
* camel is absent at 720p and 1080p because 5 of its runs carry a non-empty
  `invariant_failures`. That is why those rungs are n=5, and n=5 cannot reach
  p=0.05 on an exact two-tailed sign test however unanimous it is.

Usage:
    python tools/plot_resolution_ladder.py --data-root .
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from math import comb

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import figkit  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from bd_rate import BDError, bd_rate  # noqa: E402

LADDER = ((640, 360), (1280, 720), (1920, 1080))


def sign_p(k: int, n: int) -> float:
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) * 0.5 ** n for i in range(lo + 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--figures-dir", default=None)
    # Same contract as analyze_resolution_ladder.py: extensions are passed in,
    # never appended to the published run-files, so the clips behind the figure
    # are visible in the command that drew it.
    ap.add_argument("--baselines", nargs="+",
                    default=["config/w1f_ladder_baselines.yaml"])
    ap.add_argument("--presley", nargs="+",
                    default=["config/w1f_ladder_presley.yaml"])
    args = ap.parse_args()
    data_root = pathlib.Path(args.data_root).resolve()

    import yaml

    import presley  # noqa: F401
    from presley import db as _db
    from presley.runner import compute_experiment_hash

    def arm(paths):
        out = {}
        entries = []
        for path in paths:
            q = pathlib.Path(path)
            entries += yaml.safe_load(open(q if q.is_absolute() else data_root / q))["experiments"]
        for cfg in entries:
            d = _db.load_run(str(data_root / "results"), compute_experiment_hash(cfg))
            if d is None or d.get("invariant_failures"):
                continue          # uncitable runs never reach a figure
            out.setdefault((cfg["video"], cfg["width"], cfg["height"]), []).append(d)
        return out

    base = arm(args.baselines)
    pres = arm(args.presley)

    per_res, videos = {}, sorted({k[0] for k in pres})
    for w, h in LADDER:
        cells = {}
        for v in videos:
            b, p = base.get((v, w, h), []), pres.get((v, w, h), [])
            if len(b) < 4 or len(p) < 4:
                continue
            try:
                cells[v] = bd_rate([x["actual_bitrate_bps"] for x in b],
                                   [x["metrics"]["foreground"]["lpips_mean"] for x in b],
                                   [x["actual_bitrate_bps"] for x in p],
                                   [x["metrics"]["foreground"]["lpips_mean"] for x in p],
                                   lower_is_better=True)
            except (BDError, KeyError, TypeError):
                continue
        per_res[(w, h)] = cells

    figkit.style()
    fig, axes = plt.subplots(1, len(LADDER), figsize=(figkit.FULL_WIDTH, 2.1),
                             sharey=True)

    # ONE order for all three panels. Sorting each panel independently while
    # sharing the y axis silently relabels the bars -- panel 2's third bar would
    # carry panel 1's third label. A fixed order also lets a reader track one
    # clip across resolutions, which is the question the figure is asked, and
    # leaves a visible gap where camel is excluded rather than closing it up.
    order = sorted(videos, key=lambda v: per_res[LADDER[0]].get(v, 0.0))
    slot = {v: i for i, v in enumerate(order)}

    summary = {}
    for ax, (w, h) in zip(axes, LADDER):
        cells = per_res[(w, h)]
        present = [v for v in order if v in cells]
        vals = [cells[v] for v in present]
        colors = [figkit.COLORS["win"] if x < 0 else figkit.COLORS["loss"] for x in vals]

        ax.barh([slot[v] for v in present], vals, color=colors, height=0.62)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_ylim(len(order) - 0.5, -0.5)

        for v in order:
            if v not in cells:
                ax.text(0, slot[v], "  not citable", va="center", ha="left",
                        fontsize=6, color=figkit.COLORS["muted"], style="italic")

        n = len(vals)
        wins = sum(1 for x in vals if x < 0)
        p = sign_p(wins, n)
        ax.set_title(f"{w}$\\times${h}\n{wins}/{n} below zero, $p={p:.3f}$")
        ax.set_xlabel("BD-rate on FG-LPIPS (%)")

        summary[f"{w}x{h}"] = {
            "videos": present,
            "bd_rate_pct": [round(x, 1) for x in vals],
            "excluded": [v for v in order if v not in cells],
            "n": n, "wins": wins, "sign_p": round(p, 4),
            "median_pct": round(float(np.median(vals)), 1),
        }

    axes[0].set_ylabel("")
    fig.tight_layout()

    sentence = (
        "Grouped horizontal bar chart with three panels, one per resolution "
        "(640x360, 1280x720, 1920x1080). Each bar is one video's BD-rate on "
        "foreground LPIPS for PRESLEY against a pristine SVT-AV1 baseline; bars "
        "left of zero mean PRESLEY needs fewer bits for equal foreground quality. "
        "At 640x360 the outcome splits three to three, with dog and pigs the two "
        "clips that reverse. At 1280x720 and 1920x1080 every measured video is "
        "below zero, but each rung has only five videos because camel's runs there "
        "are not citable, and five videos cannot reach p=0.05 on an exact "
        "two-tailed sign test.")

    data = {
        "metric": "BD-rate on FG-LPIPS, percent",
        "sign_convention": "negative means PRESLEY needs fewer bits at equal quality",
        "baseline": "pristine SVT-AV1, fixed QP 43/50/55/60",
        "arm": "presley_ai downsample + Real-ESRGAN, shrink_amount 0.25, fg_protect",
        "block_grid_held_constant": "80x45 at every resolution (block size 8/16/24)",
        "by_resolution": summary,
        "caveats": [
            "camel excluded at 720p and 1080p: non-empty invariant_failures",
            "at fixed QP the higher rungs run at roughly half the bits per pixel, "
            "i.e. more starved, which is the regime that favours the method; the "
            "apparent strengthening with resolution is confounded with that",
            "the 360p split reproduces the known dog/pigs non-replication",
        ],
    }

    paths = figkit.emit("resolution_ladder", fig, sentence, data,
                        figures_dir=args.figures_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
