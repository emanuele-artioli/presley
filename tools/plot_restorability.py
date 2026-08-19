"""Post-restoration damage against transmit-time complexity — `fig:restorability`.

The stored JSON for this figure kept only summary statistics, so the plot is
re-derived from `results/block_damage_s1b.npz` through the very functions the
analysis uses. Importing `analyze_m1_restorability` rather than reimplementing
its loop is the point: a figure that drifts from the analysis it illustrates is
worse than no figure, so this script also refuses to emit unless it reproduces
the published n, median rho and positive count.

Authored at half textwidth, to pair with the selection-cost panel.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figure_style import HALF, despine, use_paper_style, width_in  # noqa: E402
import analyze_m1_restorability as M1  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1] / "68e8b6bb11d0dd9e62a67aef" / "Figures"
PUB = json.loads((HERE / "restorability.json").read_text())["data"]


def zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0.0


def collect():
    z = np.load(M1.REPO_ROOT / "results" / "block_damage_s1b.npz", allow_pickle=True)
    run, vid, res = z["run"], z["video"], z["restorer"]
    sy, sx, frame, bs = z["sy"], z["sx"], z["frame"], z["block_size"]
    dmg = z["delta_psnr"]
    keep = np.isin(res, M1.RESTORERS) & np.isfinite(dmg)
    idx_by_run = collections.defaultdict(list)
    for i in np.nonzero(keep)[0]:
        idx_by_run[str(run[i])].append(i)

    rhos, pooled_sc, pooled_d = [], [], []
    for r, idxs in sorted(idx_by_run.items()):
        idxs = np.array(idxs)
        v = str(vid[idxs[0]]); b = int(bs[idxs[0]])
        geom = M1.run_geometry(r)
        if geom is None:
            continue
        width, height = geom
        ev = M1.evca_superblocks(v, width, height, b, int(frame[idxs].max()) + 1)
        if ev is None:
            continue
        d = dmg[idxs]
        fr, yy, xx = frame[idxs], sy[idxs], sx[idxs]
        ok = ((fr < ev["sc"].shape[0]) & (yy < ev["sc"].shape[1])
              & (xx < ev["sc"].shape[2]))
        if ok.sum() < 30:
            continue
        fr, yy, xx, d = fr[ok], yy[ok], xx[ok], d[ok]
        degraded = z["strength_frac"][idxs][ok] > 0
        if degraded.sum() < 30:
            continue
        fr, yy, xx, d = fr[degraded], yy[degraded], xx[degraded], d[degraded]
        sc = ev["sc"][fr, yy, xx]
        rhos.append(M1.spearman(sc, d))
        pooled_sc.append(zscore(sc))
        pooled_d.append(zscore(d))
    return np.array(rhos), np.concatenate(pooled_sc), np.concatenate(pooled_d)


def main() -> int:
    rhos, sc_z, d_z = collect()
    n, pos, med = len(rhos), int((rhos > 0).sum()), float(np.median(rhos))
    drift = []
    if n != PUB["n_runs"]:
        drift.append(f"n={n}, published {PUB['n_runs']}")
    if pos != PUB["runs_positive"]:
        drift.append(f"positive={pos}, published {PUB['runs_positive']}")
    if abs(med - PUB["rho_median"]) > 5e-3:
        drift.append(f"median rho={med:+.4f}, published {PUB['rho_median']:+.4f}")
    if drift:
        print("ABORT — figure disagrees with the landed claim:")
        print("\n".join("  " + d for d in drift))
        return 1
    print(f"reproduced: n={n}, {pos}/{n} positive, median rho {med:+.4f}")

    use_paper_style()
    fig, (axs, axh) = plt.subplots(
        1, 2, figsize=(width_in(HALF), 1.64),
        gridspec_kw={"width_ratios": [1.15, 1.0]})

    step = max(1, len(sc_z) // 6000)          # keep the PDF light; pattern is dense
    axs.scatter(sc_z[::step], d_z[::step], s=0.7, alpha=0.16,
                color="#4878a8", linewidths=0, rasterized=True)
    m, c = np.polyfit(sc_z, d_z, 1)
    xs = np.linspace(sc_z.min(), sc_z.max(), 50)
    axs.plot(xs, m * xs + c, color="#B00020", lw=1.1)
    axs.set_xlabel("spatial complexity ($z$)", labelpad=1)
    axs.set_ylabel("damage after restoration ($z$)", labelpad=1)
    axs.set_title("more complex → more damage", fontsize=6.8, pad=3)
    axs.set_xlim(-2.6, 3.4); axs.set_ylim(-3.0, 3.4)

    axh.hist(rhos, bins=16, color="#4878a8", edgecolor="white", linewidth=0.4)
    axh.axvline(med, color="#B00020", lw=1.0, ls="--")
    axh.annotate(f"median {med:+.3f}", xy=(med, axh.get_ylim()[1] * 0.96),
                 xytext=(-3, 0), textcoords="offset points", fontsize=6.2,
                 color="#B00020", va="top", ha="right")
    axh.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    axh.set_xlabel(r"within-run Spearman $\rho$", labelpad=1)
    axh.set_ylabel("runs", labelpad=1)
    axh.set_title(f"{pos}/{n} runs positive", fontsize=6.8, pad=3)

    for ax in (axs, axh):
        despine(ax)
        ax.tick_params(labelsize=6.2)
    fig.tight_layout(pad=0.3, w_pad=1.3)
    fig.savefig(HERE / "restorability.pdf", format="pdf", dpi=400)
    print("wrote", HERE / "restorability.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
