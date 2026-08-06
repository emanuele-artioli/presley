"""Re-express the codec-ROI link on perceptual metrics, at the video as unit — A1.

`tab:roi` currently reports foreground/background PSNR over 17 *operating
points*. Two things are wrong with that as it stands, and both are free to fix:

1. **The unit is the operating point, not the video.** This project's own rule
   is that the unit of analysis is the video — `bear` contributes five rungs and
   would otherwise carry five times the weight of `camel`. Collapsing to a
   per-video mean gives 8 videos, and the sign counts become 8/8, which is
   exact-sign-test p=0.0078 rather than an uncorrected count over
   non-independent points.
2. **It is PSNR-only**, while every other perceptual verdict in the paper gates
   on LPIPS with DISTS corroborating. All 31 fixed-QP ROI runs already carry
   `foreground.lpips_mean`, `background.lpips_mean`, `dists_fg` and `dists_bg` —
   verified 31/31 — so nothing needs re-running.

What the link shows is bit *relocation at constant rate*: Δbits is centred on
zero (median −0.1%), while the foreground gains and the background pays. That is
the cleanest single-mechanism demonstration in the paper, and it currently reads
as a retraction because the (genuinely refuted) bitrate-saving claim was removed
from around it.

Scope limit that must travel with any text derived from this: kvazaar is the
only encoder in this study whose native ROI moved bits measurably. SVT-AV1's is
within JND at 64x64 superblock granularity — a *granularity limitation*, not a
refutation of ROI — and the x265 comparison is VBR-vs-VBR, which hard rule 1
excludes from degradation claims entirely. Do not write "x265 and AV1 ROI
failed".
"""
from __future__ import annotations

import collections
import json
import pathlib
from math import comb

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

# Metrics and their direction. LPIPS/DISTS are lower-is-better; PSNR higher.
LOWER_BETTER = {"lpips_mean", "dists_fg", "dists_bg"}

# JND thresholds this project gates every quality verdict on. A delta below the
# threshold is "no perceptible difference" and is NEVER reported as a trend --
# no matter how consistent its sign or how small its p-value.
JND = {"psnr_mean": 0.5, "lpips_mean": 0.05, "dists_fg": 0.05, "dists_bg": 0.05}


def sign_p(k, n):
    """Exact two-tailed sign test — never one-tailed (5/5 is 0.0625, not 0.031)."""
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) * 0.5 ** n for i in range(lo + 1)))


def load():
    """Pair each fixed-QP kvazaar ROI run with the baseline at its own target."""
    roi, base = {}, {}
    for p in RESULTS.glob("*/result.json"):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        c = d.get("config") or {}
        if d.get("invariant_failures"):
            continue
        if d.get("rate_control") != "cqp":        # fixed-QP only, never VBR
            continue
        key = (c.get("video"), c.get("target_bitrate"))
        if c.get("component") == "roi" and c.get("roi_method") == "kvazaar":
            # Canonical W1a set only: default alpha/beta/block_size. The extra
            # bear rows are alpha/beta/block-size ablations and are not the link.
            if (c.get("alpha"), c.get("beta"), c.get("block_size")) != (0.5, 0.5, 8):
                continue
            roi[key] = d
        elif c.get("component") == "baselines" and c.get("codec") == "kvazaar":
            base[key] = d
    return roi, base


def rate(d):
    return d["actual_bitrate_bps"] / 1000.0


def m(d, region, key):
    return (d.get("metrics", {}).get(region, {}) or {}).get(key)


def main() -> int:
    roi, base = load()
    pairs = sorted(set(roi) & set(base))
    if not pairs:
        print("no matched (video, target) pairs found")
        return 1

    metrics = [("foreground", "psnr_mean"), ("background", "psnr_mean"),
               ("foreground", "lpips_mean"), ("background", "lpips_mean"),
               ("foreground", "dists_fg"), ("background", "dists_bg")]

    per_video = collections.defaultdict(lambda: collections.defaultdict(list))
    bits = collections.defaultdict(list)
    for v, t in pairs:
        r, b = roi[(v, t)], base[(v, t)]
        bits[v].append((rate(r) - rate(b)) / rate(b) * 100.0)
        for reg, k in metrics:
            a, c = m(r, reg, k), m(b, reg, k)
            if a is None or c is None:
                continue
            per_video[(reg, k)][v].append(a - c)

    print(f"Codec ROI (kvazaar) vs its own matched fixed-QP baseline.")
    print(f"{len(pairs)} operating points collapsed to the VIDEO as unit.\n")

    vids = sorted({v for v, _ in pairs})
    print(f"videos (n={len(vids)}): {', '.join(vids)}\n")

    mb = {v: sum(x) / len(x) for v, x in bits.items()}
    print("Rate is held constant -- this is relocation, not a saving:")
    print(f"  per-video mean d-bits: median {sorted(mb.values())[len(mb)//2]:+.1f}%, "
          f"range {min(mb.values()):+.1f}..{max(mb.values()):+.1f}%\n")

    print(f"{'metric':28s}{'n':>4}{'favouring':>11}{'median delta':>14}"
          f"{'sign p':>10}{'JND':>8}{'perceptible?':>14}")
    for reg, k in metrics:
        d = per_video[(reg, k)]
        if not d:
            print(f"{reg+'.'+k:28s}{'--':>4}{'no data':>11}")
            continue
        means = {v: sum(x) / len(x) for v, x in d.items()}
        n = len(means)
        vals = sorted(means.values())
        med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        # "favouring ROI" = FG improves, BG degrades. For lower-is-better metrics
        # improvement is negative; for PSNR it is positive.
        lower = k in LOWER_BETTER
        want_negative = lower if reg == "foreground" else not lower
        good = sum(1 for x in means.values()
                   if (x < 0) == want_negative and x != 0)
        thr = JND[k]
        perceptible = abs(med) >= thr
        print(f"{reg+'.'+k:28s}{n:>4}{f'{good}/{n}':>11}{med:>+14.4f}"
              f"{sign_p(good, n):>10.4f}{thr:>8.2f}"
              f"{('YES' if perceptible else 'sub-JND'):>14}")

    print("\nReading: ROI moves quality from background to foreground at unchanged")
    print("rate. 'Favouring' counts videos where FG improved AND BG degraded in")
    print("the direction that metric defines as better/worse.")
    print("\nUnit is the video. bear contributes 5 operating points and still")
    print("counts once -- that is the point of the re-unitization.")
    print("\nVERDICT GATE: a sub-JND delta is 'no perceptible difference', never a")
    print("trend, regardless of how consistent its sign or how small its p-value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
