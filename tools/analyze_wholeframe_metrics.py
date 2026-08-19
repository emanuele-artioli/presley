#!/usr/bin/env python3
"""Whole-frame metrics for every measure the article names — `tab:wholeframe`.

Section 2.3 surveys PSNR, SSIM, VMAF, LPIPS, DISTS, FID and FVMD, and the
foreground claims then rest on only two of them. A metric named in a paper and
never reported is a loose end, so this reports all seven at whole-frame scope,
where each is defined without a region restriction and the bounding-box problem
that bars the others from foreground duty does not arise.

The article's own rules apply: fixed-QP/CRF only, runs carrying a non-empty
`invariant_failures` are skipped, and every arm is paired against the pristine
baseline within a (video, codec, QP, resolution) group, so nothing is averaged
across operating points that were never comparable. Per-metric pair counts are
printed because they differ -- FID and FVMD were only ever computed on the
hole-filling arms.
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
from math import comb

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

ARMS = {
    ("presley_ai", "downsample", "realesrgan"): "PRESLEY downsample + Real-ESRGAN",
    ("presley_ai", "mean_fill", "propainter"): "PRESLEY blackout + ProPainter",
    ("presley_ai", "freeze", "propainter"): "PRESLEY freeze + ProPainter",
    ("elvis", None, None): "ELVIS (block removal)",
}
METRICS = ["psnr_mean", "ssim_mean", "vmaf_mean", "lpips_mean", "dists_mean",
           "fid", "fvmd"]
FIXED_QP = ("crf", "cqp", "fixed_qp", "qp", "")


def collect():
    base, arms = {}, collections.defaultdict(dict)
    for d in RESULTS.iterdir():
        f = d / "result.json"
        if not f.is_file():
            continue
        try:
            j = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if j.get("invariant_failures"):
            continue
        if (j.get("rate_control") or "").lower() not in FIXED_QP:
            continue
        c = j.get("config") or {}
        ov = (j.get("metrics") or {}).get("overall") or {}
        if not ov:
            continue
        key = (c.get("video"), c.get("codec"),
               (c.get("codec_params") or {}).get("qp"),
               c.get("width"), c.get("height"))
        if c.get("component") == "baselines":
            base[key] = ov
        else:
            name = ARMS.get((c.get("component"), c.get("degradation"),
                             c.get("restorer")))
            if name:
                arms[name][key] = ov
    return base, arms


def sign_p(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) * 0.5 ** n for i in range(lo + 1)))


def paired_between_arms(arms):
    """Conditioned arm against each other arm, same operating point.

    The per-arm rows above are each paired against the baseline, but their
    group sets differ, so reading an ordering off them would compare medians
    taken over different operating points. This pairs the arms directly.
    """
    cond = "PRESLEY downsample + Real-ESRGAN"
    print("\npaired between arms (same video, codec, QP, resolution):")
    for other in ARMS.values():
        if other == cond or other not in arms:
            continue
        common = [k for k in arms[cond] if k in arms[other]]
        for m in ("psnr_mean", "vmaf_mean", "lpips_mean", "dists_mean"):
            ds = [arms[cond][k][m] - arms[other][k][m] for k in common
                  if isinstance(arms[cond][k].get(m), (int, float))
                  and isinstance(arms[other][k].get(m), (int, float))]
            if len(ds) < 3:
                continue
            lower_better = m in ("lpips_mean", "dists_mean")
            wins = sum(1 for x in ds if (x < 0) == lower_better and x != 0)
            print(f"  vs {other:34s} {m:11s} n={len(ds):4d} "
                  f"median {statistics.median(ds):+8.3f} "
                  f"conditioned better {wins}/{len(ds)} p={sign_p(wins, len(ds)):.4f}")


def main() -> int:
    base, arms = collect()
    head = "".join(f"{m.replace('_mean', ''):>10s}" for m in METRICS)
    print(f"{'arm':34s}{head}{'pairs':>7s}")
    for name in ARMS.values():
        rec = arms.get(name)
        if not rec:
            continue
        common = [k for k in rec if k in base]
        cells, counts = [], []
        for m in METRICS:
            deltas = [rec[k][m] - base[k][m] for k in common
                      if isinstance(rec[k].get(m), (int, float))
                      and isinstance(base[k].get(m), (int, float))]
            counts.append(len(deltas))
            cells.append(f"{statistics.median(deltas):+10.3f}"
                         if len(deltas) >= 3 else f"{'--':>10s}")
        print(f"{name:34s}{''.join(cells)}{len(common):7d}")
        print(f"{'':34s}" + "".join(f"{'n=' + str(c):>10s}" for c in counts))
    paired_between_arms(arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
