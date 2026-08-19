"""What each configuration transmits and what it recovers — `fig:patches`.

One crop, one column per configuration, two rows: what arrives at the client
before restoration, and what the restorer makes of it. Showing both rows is the
point --- the configurations differ as much in how much they destroy as in how
well they rebuild, and a restored-only figure hides the first half.

The crop is a background region, because the foreground is protected in every
configuration and would show no difference at all.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figure_style import use_paper_style, width_in  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = ROOT / "68e8b6bb11d0dd9e62a67aef" / "Figures"
VIDEO, QP, FRAME = "youtube_vos/30fe0ed0ce", 47, 14
CROP = (300, 40, 96)                            # (x, y, size), a background region

WANT = [("baselines", None, None, "pristine baseline"),
        ("elvis", "inpaint", None, "ELVIS\n+ ProPainter"),
        ("presley_ai", "downsample", "realesrgan", "PRESLEY downsample\n+ Real-ESRGAN")]


def find_runs():
    out = {}
    for d in (ROOT / "results").iterdir():
        f = d / "result.json"
        if not f.is_file():
            continue
        try:
            j = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if j.get("invariant_failures"):
            continue
        c = j.get("config") or {}
        if (c.get("video") != VIDEO or c.get("width") != 640
                or (c.get("codec_params") or {}).get("qp") != QP):
            continue
        if c.get("component") == "elvis":
            # `inpainter: none` runs are the unrestored control, not ELVIS
            if c.get("inpainter") in (None, "none"):
                continue
            key = ("elvis", "inpaint", None)
        else:
            key = (c.get("component"), c.get("degradation"), c.get("restorer"))
        v, tx = j.get("output_video"), j.get("transmitted_video")
        if v and pathlib.Path(v).exists():
            bg = (j.get("metrics") or {}).get("background") or {}
            out.setdefault(key, {"path": v, "bg_lpips": bg.get("lpips_mean"),
                                 "tx": tx if tx and pathlib.Path(tx).exists() else None})
    return out


def frame_at(path, n):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {n} of {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    runs = find_runs()
    cols = [(lab, runs[(a, b, c)]) for a, b, c, lab in WANT if (a, b, c) in runs]
    missing = [lab for a, b, c, lab in WANT if (a, b, c) not in runs]
    if missing:
        print("missing configurations, not drawn:", missing)
    refp = ROOT / "cache" / f"{VIDEO}_640x360" / "reference_frames" / f"{FRAME:05d}.png"
    refp = ROOT / "cache" / f"{VIDEO}_640x360" / "reference_frames" / f"{FRAME:05d}.png"
    ref = cv2.cvtColor(cv2.imread(str(refp)), cv2.COLOR_BGR2RGB)
    x, y, sz = CROP

    use_paper_style()
    # Columns that neither degrade nor restore repeat across both rows, so the
    # grid stays uniform and the row labels mean the same thing in every column.
    cells = [("reference", None, ref, ref)]
    for lab, r in cols:
        tx = frame_at(r["tx"], FRAME) if r["tx"] else frame_at(r["path"], FRAME)
        cells.append((lab, r["bg_lpips"], tx, frame_at(r["path"], FRAME)))
    ncol = len(cells)
    fig, axes = plt.subplots(2, ncol,
                             figsize=(width_in(0.70), width_in(0.70) / ncol * 2 + 0.52))

    def show(ax, img, title=None):
        ax.imshow(img[y:y + sz, x:x + sz])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        if title:
            ax.set_title(title, fontsize=6.2, pad=2.5, linespacing=1.05)

    for c, (lab, lp, tx, out) in enumerate(cells):
        show(axes[0][c], tx, lab if lp is None else f"{lab}\nBG-LPIPS {lp:.3f}")
        show(axes[1][c], out)
    axes[0][0].set_ylabel("transmitted", fontsize=6.3, labelpad=2)
    axes[1][0].set_ylabel("restored", fontsize=6.3, labelpad=2)
    fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.3)
    fig.savefig(HERE / "patch_comparison.pdf", format="pdf", dpi=300)
    print("wrote", HERE / "patch_comparison.pdf", f"({ncol} columns x transmitted/restored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
