#!/usr/bin/env python3
"""Does it generalize? One bar per held-out clip -- `fig:breadth`.

Replaces `tab:breadth`, `tab:breadth-ext`, `tab:breadth-ext-presley` and
`tab:breadth-ratematched`, which between them answer one referee question --
does this work outside the clips it was developed on -- by reporting win rates
a reader has to trust rather than outcomes a reader can see.

Per clip: the bitrate PRESLEY needs against the pristine baseline at the same
fixed QP, paired within (video, QP) and taken as the median over the rungs.
Negative is a saving. Sorted, coloured by sign, split by dataset family so the
DAVIS clips the system was built on sit apart from the held-out MOSEv2 and
YouTube-VOS ones.

What this deliberately does NOT do is collapse to a win rate. "7/18" invites
the reading that the method works 39% of the time, when the honest content is
a distribution with a fat negative tail on some families and a positive one on
others -- and the article's whole position is that the outcome is
content-dependent. A bar per clip is that distribution.

The foreground gate is reported alongside rather than folded in: a bitrate
saving bought by damaging the foreground is not a saving, so clips whose FG
quality moved against the baseline are marked.

Usage:
    python tools/plot_breadth.py --data-root .
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import figkit  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# The breadth recipe's rungs.
QPS = (32, 37)
MIN_RUNGS = 2

# The two arms the breadth tables actually report. tab:breadth and
# tab:breadth-ext are the ELVIS bridge (blackout) on 33 clips; the presley_ai
# downsample arm is a smaller set. Drawing only one of them would silently
# retire tables it does not cover -- the first draft of this figure did exactly
# that, showing 11 clips and claiming to replace all four tables.
ARMS = (("elvis", "blackout", "bridge (ELVIS blackout)"),
        ("presley_ai", "downsample", "PRESLEY (downsample)"))


def family(video: str) -> str:
    if video.startswith("mosev2"):
        return "MOSEv2 (held out)"
    if video.startswith("youtube_vos"):
        return "YouTube-VOS (held out)"
    return "DAVIS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--figures-dir", default=None)
    args = ap.parse_args()
    data_root = pathlib.Path(args.data_root).resolve()

    import presley  # noqa: F401
    from presley import db as _db

    conn = _db.connect(str(data_root / "results"))
    runs = collections.defaultdict(dict)
    for row in conn.execute(
            "SELECT hash, component, video, qp FROM v_citable WHERE qp IS NOT NULL"):
        if row["qp"] not in QPS:
            continue
        doc = _db.get_run(conn, row["hash"])
        cfg = doc.get("config") or {}
        mode = cfg.get("degradation") or cfg.get("removal_mode")
        if row["component"] == "baselines":
            arm = "baselines"
        else:
            match = [a for a in ARMS if a[0] == row["component"] and a[1] == mode]
            if not match:
                continue
            arm = f"{match[0][0]}/{match[0][1]}"
        fg = ((doc.get("metrics") or {}).get("foreground") or {}).get("lpips_mean")
        runs[(row["video"], row["qp"])][arm] = (doc.get("actual_bitrate_bps"), fg)

    def rows_for(component, mode):
        per_clip, fg_delta = collections.defaultdict(list), collections.defaultdict(list)
        key = f"{component}/{mode}"
        for (video, _qp), arms in runs.items():
            if key not in arms or "baselines" not in arms:
                continue
            (pb, pf), (bb, bf) = arms[key], arms["baselines"]
            if not pb or not bb:
                continue
            per_clip[video].append(100.0 * (pb - bb) / bb)
            if pf is not None and bf is not None:
                fg_delta[video].append(pf - bf)
        out = [(v, float(np.median(d)),
                float(np.median(fg_delta[v])) if fg_delta.get(v) else None, len(d))
               for v, d in per_clip.items() if len(d) >= MIN_RUNGS]
        out.sort(key=lambda r: (family(r[0]), r[1]))
        return out

    panels = [(label, rows_for(comp, mode)) for comp, mode, label in ARMS]
    panels = [(lab, r) for lab, r in panels if r]
    if not panels:
        raise SystemExit("no matched arm/baseline pairs at the breadth rungs")
    rows = panels[0][1]

    figkit.style()
    tallest = max(len(r) for _, r in panels)
    fig, axes = plt.subplots(1, len(panels), sharex=True,
                             figsize=(figkit.FULL_WIDTH, 0.14 * tallest + 1.5))
    axes = np.atleast_1d(axes)

    summary = {}
    for ax, (label, prows) in zip(axes, panels):
        vals = [r[1] for r in prows]
        colors = [figkit.COLORS["win"] if v < 0 else figkit.COLORS["loss"] for v in vals]
        y = np.arange(len(prows))
        ax.barh(y, vals, color=colors, height=0.7)

        # A saving bought by damaging the foreground is not a saving. Mark it
        # rather than silently counting it as a win.
        marked = []
        for i, (vid, delta, fgd, _n) in enumerate(prows):
            if fgd is not None and fgd > 0.01 and delta < 0:
                ax.plot(delta - 2.0, i, marker="x", markersize=3.0,
                        color=figkit.COLORS["neutral"])
                marked.append(vid)

        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0].split("/")[-1][:13] for r in prows], fontsize=5)
        ax.set_ylim(tallest - 0.5, -0.5)
        ax.set_xlabel("bitrate vs pristine baseline (%)")

        fams = [family(r[0]) for r in prows]
        for i in range(1, len(fams)):
            if fams[i] != fams[i - 1]:
                ax.axhline(i - 0.5, color=figkit.COLORS["neutral"],
                           linewidth=0.7, linestyle=":")

        wins = sum(1 for v in vals if v < 0)
        ax.set_title(f"{label}\n{wins}/{len(vals)} save bits", fontsize=7)

        by_family = collections.defaultdict(list)
        for vid, delta, _f, _n in prows:
            by_family[family(vid)].append(delta)
        summary[label] = {
            "n_clips": len(prows), "clips_saving_bits": wins,
            "clips_costing_bits": len(prows) - wins,
            "median_pct": round(float(np.median(vals)), 1),
            "median_by_family": {k: round(float(np.median(v)), 1)
                                 for k, v in by_family.items()},
            "saving_with_foreground_cost": marked,
            "per_clip": [{"video": v, "bitrate_delta_pct": round(d, 1),
                          "fg_lpips_delta": (round(f, 4) if f is not None else None)}
                         for v, d, f, _n in prows],
        }
    fig.tight_layout()

    total = sum(s["n_clips"] for s in summary.values())
    total_win = sum(s["clips_saving_bits"] for s in summary.values())
    sentence = (
        f"Two panels of horizontal bars, one per arm, {total} clip-level outcomes in "
        "total, each grouped by dataset family into DAVIS and the held-out MOSEv2 "
        "and YouTube-VOS sets. Each bar is the bitrate that arm needs against a "
        "pristine baseline at the same fixed QP, so left of zero is a saving. "
        f"Across both arms {total_win} of {total} clips save bits. The split runs "
        "within families rather than between them, which is the article's "
        "content-dependence result shown as a distribution rather than as a win "
        "rate. Bars marked with a cross saved bits while their foreground quality "
        "moved against the baseline, so that saving is not free.")

    data = {
        "quantity": "median over QP rungs of (arm bitrate - baseline bitrate) / "
                    "baseline bitrate, percent, paired within (video, QP)",
        "rungs": list(QPS),
        "arms": summary,
        "totals": {"clips": total, "saving_bits": total_win,
                   "costing_bits": total - total_win},
        "why_not_a_win_rate": "a win rate invites 'the method works N% of the time'; "
                              "the honest content is a distribution whose sign is "
                              "content-dependent, which is the article's position",
        "replaces_tables": ["tab:breadth", "tab:breadth-ext",
                            "tab:breadth-ext-presley", "tab:breadth-ratematched"],
    }

    paths = figkit.emit("breadth", fig, sentence, data, figures_dir=args.figures_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")
    for label, s in summary.items():
        print(f"  {label:26}{s['clips_saving_bits']}/{s['n_clips']} save  "
              f"median {s['median_pct']:+.1f}%   by family {s['median_by_family']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
