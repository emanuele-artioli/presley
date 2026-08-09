"""The regime reversal, drawn as rate-distortion ladders — `fig:regime-reversal`.

`tab:av1` and `tab:av1-breadth` each collapse a four-rung QP ladder into a
single Bjontegaard scalar. That scalar is exactly what hides this article's most
consequential finding: `bear`/`camel` free bits when starved, `dog`/`pigs` cost
bits when starved AND free them at the mildest rung, so the flip is *inverted*
rather than merely absent. A BD-rate integral averages that inversion away.

This script draws the four ladders side by side so the reversal is visible
directly: the bridge curve sitting LEFT of the baseline at equal height means
fewer bits for the same foreground quality. On the two incumbents it leans left,
on the two added videos it leans right.

Provenance is shared with the table on purpose -- the pairing and the BD-rate
come from the same `scripts/bd_rate.py` the CLAIM lines cite, and the script
re-derives the four published BD-rates and refuses to emit a figure if any of
them disagrees. A figure that silently contradicts a landed table would be worse
than no figure.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless host: never a GUI backend
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from bd_rate import bd_rate  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT = (pathlib.Path(__file__).resolve().parents[1]
       / "68e8b6bb11d0dd9e62a67aef" / "Figures" / "regime_reversal.pdf")

# Rungs are recalibrated per video so the baseline PSNR ranges bracket each
# other; they are NOT the same QPs. Taken from CLAIM(tab:av1)/CLAIM(tab:av1-breadth).
LADDERS = {
    "bear":  (43, 51, 58, 61),
    "camel": (42, 50, 58, 62),
    "dog":   (50, 55, 60, 63),
    "pigs":  (50, 55, 60, 63),
}
# Incumbents first, then the two that reverse.
ORDER = ("bear", "camel", "dog", "pigs")
INCUMBENTS = {"bear", "camel"}
MODE = "blackout"  # the stronger arm, and the one both tables lead with

# Published BD-rates the figure must reproduce (negative = fewer bits).
PUBLISHED = {"bear": -28.9, "camel": -25.6, "dog": +27.3, "pigs": +18.3}
TOLERANCE = 0.15  # percentage points; guards against transcription drift only


def load():
    """Pair pristine baselines with the bs16 bridge arm, per (video, qp)."""
    base, bridge = {}, {}
    for p in RESULTS.glob("*/result.json"):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        c = d.get("config") or {}
        v = c.get("video")
        if v not in LADDERS or c.get("codec") != "svtav1" or c.get("height") != 360:
            continue
        qp = (c.get("codec_params") or {}).get("qp")
        if qp not in LADDERS[v]:
            continue
        if d.get("invariant_failures"):
            continue  # a run with failures is never citable, so never plotted
        if d.get("rate_control") != "crf":
            continue  # fixed-QP only; a VBR point is not evidence
        if c.get("component") == "baselines":
            base[(v, qp)] = d
        elif (c.get("component") == "elvis" and c.get("block_size") == 16
              and c.get("removal_mode") == MODE):
            bridge[(v, qp)] = d
    return base, bridge


def curve(store, video):
    """(rate kbps, FG-PSNR dB) up the ladder, or None if a rung is missing."""
    pts = []
    for qp in LADDERS[video]:
        d = store.get((video, qp))
        if d is None:
            return None
        rate = d["actual_bitrate_bps"] / 1000.0  # never target_bitrate
        psnr = d["metrics"]["foreground"]["psnr_mean"]
        pts.append((rate, psnr))
    return sorted(pts)


def main() -> int:
    base, bridge = load()
    curves, failures = {}, []
    for v in ORDER:
        b, g = curve(base, v), curve(bridge, v)
        if b is None or g is None:
            failures.append(f"{v}: missing rungs (baseline={b is not None}, bridge={g is not None})")
            continue
        curves[v] = (b, g)

    if failures:
        print("ABORT — incomplete ladders:")
        print("\n".join("  " + f for f in failures))
        return 1

    # Reconciliation against the landed tables, before anything is drawn.
    print("BD-rate reconciliation (bridge vs pristine, negative = fewer bits):")
    drift = []
    for v in ORDER:
        b, g = curves[v]
        got = bd_rate([r for r, _ in b], [q for _, q in b],
                      [r for r, _ in g], [q for _, q in g])
        want = PUBLISHED[v]
        ok = abs(got - want) <= TOLERANCE
        print(f"  {v:6s} computed {got:+7.2f}%   published {want:+6.1f}%   "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            drift.append(f"{v}: computed {got:+.2f}% vs published {want:+.1f}%")
    if drift:
        print("\nABORT — figure disagrees with the landed table:")
        print("\n".join("  " + d for d in drift))
        return 1

    # --- draw ------------------------------------------------------------
    # Print conventions: colourblind-safe, and separable in greyscale through
    # marker shape and line style, not colour alone.
    C_BASE, C_BRIDGE = "#4d4d4d", "#0072B2"
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.35))
    for ax, v in zip(axes.ravel(), ORDER):
        b, g = curves[v]
        ax.plot([r for r, _ in b], [q for _, q in b], color=C_BASE,
                marker="o", ms=4.5, lw=1.3, ls="-", label="pristine SVT-AV1")
        ax.plot([r for r, _ in g], [q for _, q in g], color=C_BRIDGE,
                marker="s", ms=4.5, lw=1.6, ls="--", label="bridge (blackout)")
        verdict = "frees bits" if PUBLISHED[v] < 0 else "costs bits"
        ax.set_title(f"\\texttt{{{v}}}  ({PUBLISHED[v]:+.1f}\\% BD-rate, {verdict})"
                     if False else f"{v}\n{PUBLISHED[v]:+.1f}% BD-rate, {verdict}",
                     fontsize=7, pad=3,
                     fontweight="bold" if v not in INCUMBENTS else "normal")
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for ax in axes.ravel():
        ax.set_xlabel("bitrate (kbit/s)", fontsize=9)
    axes.ravel()[0].set_ylabel("foreground PSNR (dB)", fontsize=9)

    # Say what to look for, once, and show it as the horizontal gap that BD-rate
    # actually integrates -- not as the vertical gap, which is a different claim.
    def rate_at(pts, q):
        """Bitrate where a ladder reaches quality q, linearly in log-rate."""
        import math
        for (r0, q0), (r1, q1) in zip(pts, pts[1:]):
            if q0 <= q <= q1:
                t = (q - q0) / (q1 - q0)
                return math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0)))
        return None

    panel_by_video = dict(zip(ORDER, axes.ravel()))
    for panel, v in ((panel_by_video["bear"], "bear"), (panel_by_video["dog"], "dog")):
        b, g = curves[v]
        # a quality level both ladders actually reach, so nothing is extrapolated
        q = max(b[0][1], g[0][1]) + 0.55 * (min(b[-1][1], g[-1][1]) - max(b[0][1], g[0][1]))
        rb, rg = rate_at(b, q), rate_at(g, q)
        if rb is None or rg is None:
            continue
        panel.annotate("", xy=(rg, q), xytext=(rb, q),
                       arrowprops=dict(arrowstyle="<->", lw=0.9, color="#B00020"))
        panel.plot([rb, rg], [q, q], ls=":", lw=0.6, color="#B00020", zorder=1)
        saved = (rb - rg) / rb * 100.0
        word = "fewer" if saved > 0 else "more"
        panel.annotate(f"at equal FG quality:\n{abs(saved):.0f}% {word} bits",
                       xy=((rb + rg) / 2, q), xytext=(0.30, 0.08),
                       textcoords="axes fraction", fontsize=5.5, ha="left",
                       color="#B00020",
                       arrowprops=dict(arrowstyle="-", lw=0.6, color="#B00020"))

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight")  # vector, for print
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
