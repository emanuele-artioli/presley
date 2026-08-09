#!/usr/bin/env python3
"""Every restoration backbone against its own control -- `fig:restorers`.

Replaces seven tables (`tab:inpainters`, `tab:conditioned`,
`tab:conditioned-twins`, `tab:conditioned-stream-diffvsr`, `tab:instantir-kill`,
`tab:goal2`, `tab:goal2-breadth`) whose finding is singular and does not need
seven tables to state: **no backbone separates from Real-ESRGAN, and what
separates is the transport it is paired with, not the model.**

Each point is one restorer run minus its matched `none` control -- the same
video, transport, block size, budget, codec and QP, differing only in whether a
restorer ran. That pairing is what makes the comparison a measurement of the
model rather than of the operating point, and runs without an exact control are
dropped rather than compared against an approximate one.

Positive = the restorer improved BG-LPIPS over the degraded video it was
handed. Deliberately reported as a raw delta with the per-pair points visible,
not as a multiple of a threshold: the article is retiring its uncited JND
constants, and a dot plot shows the spread a threshold multiple hides.

Note what this does NOT show: restored-vs-original against a pristine baseline,
which is the Goal-2 headline and a different (harder) question. This is the
restoration *gain* -- the mechanism -- and it is labelled as such.

Usage:
    python tools/plot_restorer_catalogue.py --data-root .
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import figkit  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Keys that identify the restorer rather than the operating point. Two runs
# that agree on everything else and differ here are a matched pair.
RESTORER_KEYS = {"restorer", "restorer_params", "inpainter", "inpainter_params"}

# Which family each transport belongs to. The split is the finding: in-painters
# discard the transmitted pixels inside the hole by construction, conditioned
# restorers consume them.
FAMILY = {
    "downsample": "conditioned (keeps the prior)",
    "blur": "conditioned (keeps the prior)",
    "ac_truncate": "conditioned (keeps the prior)",
    "blackout": "in-painting (discards the prior)",
    "freeze": "in-painting (discards the prior)",
    "mean_fill": "in-painting (discards the prior)",
}
MIN_PAIRS = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--figures-dir", default=None)
    args = ap.parse_args()
    data_root = pathlib.Path(args.data_root).resolve()

    import presley  # noqa: F401
    from presley import db as _db

    conn = _db.connect(str(data_root / "results"))
    by_context = collections.defaultdict(dict)
    for row in conn.execute("SELECT hash FROM v_citable"):
        doc = _db.get_run(conn, row["hash"])
        cfg = doc.get("config") or {}
        restorer = cfg.get("restorer") or cfg.get("inpainter")
        if not restorer:
            continue
        bg = ((doc.get("metrics") or {}).get("background") or {}).get("lpips_mean")
        if bg is None:
            continue
        transport = cfg.get("degradation") or cfg.get("removal_mode")
        if transport not in FAMILY:
            continue
        context = json.dumps({k: v for k, v in sorted(cfg.items())
                              if k not in RESTORER_KEYS}, sort_keys=True, default=str)
        by_context[context][restorer] = (bg, transport)

    # Gain against the matched `none` control AND the absolute quality reached.
    #
    # Both panels are mandatory, and the reason is a trap this figure walked
    # into on its first draft. Blackout in-painting shows the LARGEST gains
    # (+0.087) against downsample + Real-ESRGAN's +0.034 -- but blackout
    # destroys far more to begin with, so a bigger gain is recovery from a
    # deeper hole, not a better picture. Gain alone ranks the transports that
    # damage most as the ones that restore best, which is backwards. The
    # absolute panel is what the viewer actually receives.
    gains = collections.defaultdict(list)
    absolute = collections.defaultdict(list)
    for arms in by_context.values():
        if "none" not in arms:
            continue
        base_bg, transport = arms["none"]
        for restorer, (bg, _) in arms.items():
            if restorer == "none":
                continue
            key = (FAMILY[transport], transport, restorer)
            gains[key].append(base_bg - bg)
            absolute[key].append(bg)

    rows = [(fam, tr, r, v) for (fam, tr, r), v in gains.items() if len(v) >= MIN_PAIRS]
    if not rows:
        raise SystemExit("no matched restorer/none pairs found")
    # Sorted by the ABSOLUTE result, best first, because that is the ordering a
    # reader should take away.
    rows.sort(key=lambda t: (t[0], float(np.median(absolute[(t[0], t[1], t[2])]))))

    figkit.style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figkit.FULL_WIDTH,
                                                0.34 * len(rows) + 1.3), sharey=True)

    labels, ypos = [], []
    for i, (fam, transport, restorer, vals) in enumerate(rows):
        med = float(np.median(vals))
        color = figkit.COLORS["win"] if med > 0 else figkit.COLORS["loss"]
        ax.scatter(vals, [i] * len(vals), s=9, alpha=0.45, color=color,
                   edgecolors="none", zorder=2)
        ax.plot([med, med], [i - 0.30, i + 0.30], color=color, linewidth=2.0, zorder=3)

        av = absolute[(fam, transport, restorer)]
        bx.scatter(av, [i] * len(av), s=9, alpha=0.45,
                   color=figkit.COLORS["accent"], edgecolors="none", zorder=2)
        bx.plot([np.median(av)] * 2, [i - 0.30, i + 0.30],
                color=figkit.COLORS["accent"], linewidth=2.0, zorder=3)

        labels.append(f"{restorer} on {transport}  (n={len(vals)})")
        ypos.append(i)

    ax.axvline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("gain over the unrestored control\n(higher = the restorer helped)")
    ax.set_title("Recovery from the hole it was given")
    bx.set_xlabel("BG-LPIPS of the restored output\n(lower = closer to the original)")
    bx.set_title("What the viewer actually receives")

    fams = [r[0] for r in rows]
    for i in range(1, len(fams)):
        if fams[i] != fams[i - 1]:
            for axis in (ax, bx):
                axis.axhline(i - 0.5, color=figkit.COLORS["neutral"],
                             linewidth=0.6, linestyle=":")
    fig.tight_layout()

    data = {
        "quantity": "BG-LPIPS of the matched unrestored control minus BG-LPIPS of "
                    "the restored run; positive means the restorer helped",
        "pairing": "exact: same video, transport, block size, budget, codec and QP; "
                   "runs without an exact `none` control are dropped, never "
                   "compared against an approximate one",
        "note": "the gain panel is the MECHANISM and the absolute panel is what "
                "the viewer receives; neither is restored-vs-original against a "
                "pristine baseline at matched rate, which is the Goal-2 headline "
                "and a harder question",
        "trap": "ranking by gain alone ranks the transports that damage most as "
                "the ones that restore best: blackout in-painting shows the "
                "largest gains while reaching worse absolute quality than "
                "downsample with Real-ESRGAN",
        "arms": [
            {"family": fam, "transport": tr, "restorer": r, "n_pairs": len(v),
             "median_gain_over_control": round(float(np.median(v)), 4),
             "median_absolute_bg_lpips": round(float(np.median(absolute[(fam, tr, r)])), 4),
             "min_gain": round(float(np.min(v)), 4),
             "max_gain": round(float(np.max(v)), 4)}
            for fam, tr, r, v in rows
        ],
        "replaces_tables": ["tab:inpainters", "tab:conditioned",
                            "tab:conditioned-twins", "tab:conditioned-stream-diffvsr",
                            "tab:instantir-kill", "tab:goal2", "tab:goal2-breadth"],
    }

    best_gain = max(rows, key=lambda t: float(np.median(t[3])))
    best_abs = min(rows, key=lambda t: float(np.median(absolute[(t[0], t[1], t[2])])))
    sentence = (
        "Two dot plots sharing one axis of restoration backbone and transport "
        "pairings, grouped into conditioned restorers that consume the transmitted "
        "pixels and in-painters that discard them. Left: each dot is one run minus "
        "its exactly matched unrestored control on background LPIPS, so right of "
        "zero means the restorer improved the background it was handed; the largest "
        f"median gain is {best_gain[2]} on {best_gain[1]} at "
        f"{float(np.median(best_gain[3])):.3f}. Right: the absolute background LPIPS "
        "of the restored output, where lower is closer to the original and "
        f"{best_abs[2]} on {best_abs[1]} is best at "
        f"{float(np.median(absolute[(best_abs[0], best_abs[1], best_abs[2])])):.3f}. "
        "The two panels disagree, and that is the point: the transports that "
        "destroy most show the largest recovery while still delivering a worse "
        "picture, so gain over a control cannot be read as restoration quality.")

    paths = figkit.emit("restorer_catalogue", fig, sentence, data,
                        figures_dir=args.figures_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")
    for fam, tr, r, v in rows:
        print(f"  {fam:32}{r:16}{tr:12}n={len(v):3d}  median {np.median(v):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
