"""Bjontegaard-Delta rate/quality between two rate-quality curves.

The repo had no BD-rate implementation, but CLAUDE.md names BD-rate as the
bar for paper-grade claims ("compare at similar actual bitrates for
preliminary conclusions, and use BD-rate curves for paper-grade claims"), so
this is that tool.

Standard Bjontegaard (VCEG-M33 / Bjontegaard 2001): fit a cubic polynomial to
quality vs log10(rate), integrate the difference over the overlapping rate
range, and report the average. Two outputs:

  bd_rate    -- average % bitrate change of B vs A at equal quality.
                NEGATIVE = B needs fewer bits for the same quality = B wins.
  bd_quality -- average quality change of B vs A at equal bitrate.

Two deviations from the video-coding default, both deliberate:

1. **Perceptual metrics, not PSNR.** PRESLEY's Goal 2 is explicitly
   BG-LPIPS/BG-DISTS, and BG-PSNR is barred from being the verdict (a flat
   fill scores highest while looking worst). Pass ``lower_is_better=True`` for
   LPIPS/DISTS. The math is identical; only the sign convention flips, which
   is handled internally so a negative ``bd_rate`` always means "B is better".

2. **Rate is transmitted bytes, not video bytes.** PRESLEY ships a side
   channel (the per-block strength map) that must be billed to the method, so
   callers pass ``transmitted_size_bytes``.

Requires >= 4 points per curve (cubic fit) and a non-empty overlap in rate.
Both are checked rather than assumed -- a BD number computed off two points,
or off curves that barely overlap, is the kind of thing that reads as
authoritative and is meaningless.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np


class BDError(ValueError):
    """Raised when the inputs cannot support a meaningful BD computation."""


def _prep(rate: Sequence[float], qual: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    r = np.asarray(rate, dtype=float)
    q = np.asarray(qual, dtype=float)
    if r.size != q.size:
        raise BDError(f"rate/quality length mismatch: {r.size} vs {q.size}")
    if r.size < 4:
        raise BDError(f"need >= 4 points for a cubic BD fit, got {r.size}")
    if np.any(r <= 0):
        raise BDError("rates must be positive")
    order = np.argsort(r)
    return r[order], q[order]


def bd_rate(rate_a, qual_a, rate_b, qual_b, lower_is_better: bool = True) -> float:
    """Average % bitrate change of curve B vs curve A at equal quality.

    Negative => B achieves the same quality for fewer bits => B is better.
    """
    ra, qa = _prep(rate_a, qual_a)
    rb, qb = _prep(rate_b, qual_b)
    # Work in "higher is better" space so the integration bounds are ordered
    # the same way regardless of metric direction.
    if lower_is_better:
        qa, qb = -qa, -qb

    la, lb = np.log10(ra), np.log10(rb)
    # Integrate over quality, so the polynomials are rate = f(quality).
    pa = np.polyfit(qa, la, 3)
    pb = np.polyfit(qb, lb, 3)

    lo = max(qa.min(), qb.min())
    hi = min(qa.max(), qb.max())
    if hi <= lo:
        raise BDError(
            "curves do not overlap in quality; BD-rate is undefined "
            f"(A spans [{qa.min():.4f},{qa.max():.4f}], B spans [{qb.min():.4f},{qb.max():.4f}])")

    ia = np.polyval(np.polyint(pa), hi) - np.polyval(np.polyint(pa), lo)
    ib = np.polyval(np.polyint(pb), hi) - np.polyval(np.polyint(pb), lo)
    return float((10 ** ((ib - ia) / (hi - lo)) - 1) * 100)


def bd_quality(rate_a, qual_a, rate_b, qual_b, lower_is_better: bool = True) -> float:
    """Average quality change of B vs A at equal bitrate (in metric units).

    Sign is returned in the metric's own convention: for a lower-is-better
    metric a negative result means B is better.
    """
    ra, qa = _prep(rate_a, qual_a)
    rb, qb = _prep(rate_b, qual_b)
    la, lb = np.log10(ra), np.log10(rb)
    pa = np.polyfit(la, qa, 3)
    pb = np.polyfit(lb, qb, 3)

    lo = max(la.min(), lb.min())
    hi = min(la.max(), lb.max())
    if hi <= lo:
        raise BDError("curves do not overlap in rate; BD-quality is undefined")

    ia = np.polyval(np.polyint(pa), hi) - np.polyval(np.polyint(pa), lo)
    ib = np.polyval(np.polyint(pb), hi) - np.polyval(np.polyint(pb), lo)
    return float((ib - ia) / (hi - lo))


def overlap_fraction(rate_a, rate_b) -> float:
    """Fraction of the union rate range that the two curves share.

    Report this next to any BD number: a BD computed over a sliver of shared
    range is arithmetically valid and practically meaningless.
    """
    la = np.log10(np.asarray(rate_a, dtype=float))
    lb = np.log10(np.asarray(rate_b, dtype=float))
    lo, hi = max(la.min(), lb.min()), min(la.max(), lb.max())
    union_lo, union_hi = min(la.min(), lb.min()), max(la.max(), lb.max())
    if union_hi <= union_lo:
        return 0.0
    return float(max(0.0, hi - lo) / (union_hi - union_lo))
