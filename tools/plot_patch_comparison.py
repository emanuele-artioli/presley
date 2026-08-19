"""Side-by-side crops of what each configuration actually delivers — `fig:patches`.

The metric tables say the conditioned transport delivers the best background.
This is the same claim at the pixel level: identical frame, identical crop, one
column per configuration, so a reader can check the ordering by eye instead of
taking the LPIPS column on trust.

Crops are chosen from background-only regions using the run's own foreground
mask, because the foreground is protected in every configuration and would show
no difference at all.
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
CROPS = [(300, 40, 96), (150, 210, 96)]        # (x, y, size), background regions

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
        v = j.get("output_video")
        if v and pathlib.Path(v).exists():
            bg = (j.get("metrics") or {}).get("background") or {}
            out.setdefault(key, {"path": v, "bg_lpips": bg.get("lpips_mean")})
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
    ref = cv2.cvtColor(cv2.imread(str(refp)), cv2.COLOR_BGR2RGB)
    panels = [("reference", None, ref)] + [(lab, r["bg_lpips"], frame_at(r["path"], FRAME))
                                           for lab, r in cols]

    use_paper_style()
    n = len(panels)
    fig, axes = plt.subplots(len(CROPS), n,
                             figsize=(width_in(0.74), width_in(0.74) / n * len(CROPS) + 0.40))
    for r, (x, y, s) in enumerate(CROPS):
        for c, (lab, lp, img) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(img[y:y + s, x:x + s])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if r == 0:
                t = lab if lp is None else f"{lab}\nBG-LPIPS {lp:.3f}"
                ax.set_title(t, fontsize=6.2, pad=2.5, linespacing=1.05)
    fig.tight_layout(pad=0.2, w_pad=0.35, h_pad=0.35)
    fig.savefig(HERE / "patch_comparison.pdf", format="pdf", dpi=300)
    print("wrote", HERE / "patch_comparison.pdf", f"({n} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
