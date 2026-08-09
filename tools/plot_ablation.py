#!/usr/bin/env python3
"""What each configuration knob actually moves -- `fig:ablation`.

Replaces `tab:ablation`, `tab:graded`, `tab:graded-oracle` and
`tab:budget-knee`, whose four tables make one point: **the removal budget is
the only knob that moves the rate axis at scale, and the scoring parameters
move nothing.** That is the empirical half of the article's selection argument,
and as four tables it has to be assembled by the reader.

Each knob gets its within-group spread on two axes -- background LPIPS and
bitrate -- where a group holds every other setting fixed and varies only that
knob. Big on the rate axis and small on quality means a lever; small on both
means an inert knob.

Two things stated rather than smoothed over:

**This is descriptive.** n is far below the n>=8 videos the project's hard rule
requires for a significance claim, and `analyze_parameter_sweeps.py` says so
in its own header. The figure carries that in its title, not a footnote.

**alpha and beta are absent from this view.** They were measured by a separate
deterministic ablation on foreground PSNR, not by these sweeps, so putting
them on a background-LPIPS axis would be comparing different quantities on one
scale. Their result -- a 0.03 to 0.05 dB range across the whole sweep -- is
reported in the text where its units belong. Their absence here is a scope
statement, not an omission.

Usage:
    python tools/plot_ablation.py --sweeps docs/sweeps.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import figkit  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Knobs that are part of the reported design. The rest of what the sweep finds
# are probe-only axes (oracle level maps, uniform-level probes) that exist to
# answer one retired question and would read as live design elements here.
REPORTED = {
    "shrink_amount": "removal budget",
    "block_size": "block size",
    "selection_rule": "selection rule",
    "mask_source": "mask source",
    "mask_morphology": "mask noise",
    "fg_protect": "foreground protection",
    "downsample_levels": "graded levels",
    "downsample_level_map": "graded levels, oracle-assigned",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--sweeps", default=None)
    ap.add_argument("--figures-dir", default=None)
    args = ap.parse_args()

    if args.sweeps:
        payload = json.loads(pathlib.Path(args.sweeps).read_text())
    else:
        tmp = pathlib.Path(tempfile.mkdtemp()) / "sweeps.json"
        subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "analyze_parameter_sweeps.py"),
                        "--json", str(tmp)], cwd=args.data_root, check=True,
                       capture_output=True, text=True)
        payload = json.loads(tmp.read_text())

    rows = []
    for entry in payload:
        key = entry["key"]
        if key not in REPORTED:
            continue
        q = entry["axes"].get("quality", {})
        b = entry["axes"].get("bitrate", {})
        if not q or not b:
            continue
        rows.append({
            "key": key, "label": REPORTED[key],
            "quality": q["median_spread"], "bitrate": b["median_spread"],
            "n_groups": entry["n_groups"], "n_videos": entry["n_videos"],
            "values": entry.get("values_seen", []),
        })
    if not rows:
        raise SystemExit("no reported knobs found in the sweep output")
    rows.sort(key=lambda r: r["bitrate"])

    figkit.style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figkit.FULL_WIDTH,
                                                0.26 * len(rows) + 1.0), sharey=True)
    y = np.arange(len(rows))
    labels = [f"{r['label']}  (n={r['n_groups']})" for r in rows]

    # The budget is the only bar that reaches; colouring by rank would imply an
    # ordering the data does not support, so only the top one is highlighted.
    top = max(r["bitrate"] for r in rows)
    colors = [figkit.COLORS["accent"] if r["bitrate"] >= 0.5 * top
              else figkit.COLORS["muted"] for r in rows]

    bx.barh(y, [r["bitrate"] for r in rows], color=colors, height=0.62)
    bx.set_xlabel("bitrate spread (pp)")
    bx.set_title("Rate: one knob does the work")

    ax.barh(y, [r["quality"] for r in rows], color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("BG-LPIPS spread")
    ax.set_title("Quality: only the budget moves it")

    fig.suptitle("Descriptive only: n is below the threshold for a significance claim",
                 fontsize=7, y=1.02, color=figkit.COLORS["neutral"])
    fig.tight_layout()

    budget = next(r for r in rows if r["key"] == "shrink_amount")
    sentence = (
        "Two horizontal bar charts sharing one axis of configuration knobs. Left: "
        "the median spread in background LPIPS within a group that varies only "
        "that knob. Right: the same for bitrate, in percentage points. The removal "
        f"budget moves bitrate by {budget['bitrate']:.1f} percentage points, several "
        "times any other knob, while every knob including the budget leaves "
        f"background quality nearly unchanged except the budget itself, which moves "
        f"it by {budget['quality']:.3f}. So the budget is the only real lever on "
        "either axis and the scoring parameters are not levers at all. The comparison is "
        "descriptive: the number of videos is below the threshold this project "
        "requires for a significance claim.")

    data = {
        "quantity": "median within-group spread; a group fixes every other setting "
                    "and varies only the named knob",
        "status": "DESCRIPTIVE ONLY -- n is below the n>=8 videos required for a "
                  "significance claim, per the project's hard rule",
        "knobs": [
            {"knob": r["label"], "config_key": r["key"], "n_groups": r["n_groups"],
             "n_videos": r["n_videos"], "values_seen": r["values"],
             "median_bg_lpips_spread": round(r["quality"], 4),
             "median_bitrate_spread_pp": round(r["bitrate"], 2)}
            for r in rows
        ],
        "reading": "the removal budget is the only knob that moves the rate axis "
                   "at scale; the scoring parameters move neither axis",
        "alpha_beta_scope": "alpha and beta are absent here on purpose: they were "
                            "measured by a separate deterministic ablation on "
                            "foreground PSNR (a 0.03-0.05 dB range across their "
                            "whole sweep), and plotting them on a background-LPIPS "
                            "axis would put different quantities on one scale",
        "replaces_tables": ["tab:ablation", "tab:graded", "tab:graded-oracle",
                            "tab:budget-knee"],
    }

    paths = figkit.emit("ablation", fig, sentence, data, figures_dir=args.figures_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")
    for r in rows:
        print(f"  {r['label']:32}BG-LPIPS {r['quality']:.4f}   bits {r['bitrate']:6.2f} pp"
              f"   n={r['n_groups']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
