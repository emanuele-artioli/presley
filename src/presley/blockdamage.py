"""Superblock geometry for the damage-after-restoration denominator.

PRESLEY's selection score is a ratio, dBits / dDamage-after-restoration, and
only the numerator was ever modelled. The denominator is measured per 64x64
superblock -- AV1's superblock, kvazaar's CTU and LPIPS's patch all agree on
that size, so it is the unit at which "this block came back badly" is both
codec-meaningful and perceptually meaningful.

Two consumers share this module so they cannot drift apart on geometry:
`tools/mine_block_damage.py`, which joins runs already on disk, and
`presley.components.probe_block_damage`, which measures one run through the
runner so the number carries a `results/<hash>`. A disagreement between them
about where superblock boundaries fall would be invisible in both outputs.

MSE is what gets pooled, never PSNR -- PSNR is logarithmic, so averaging it
across blocks is meaningless. Partial edge superblocks are kept with their true
pixel area rather than dropped, mirroring how AV1 pads to a superblock and crops
on output.
"""

from __future__ import annotations

import numpy as np

SB = 64          # superblock side in pixels
MAX_I = 255.0    # 8-bit peak signal, matching the evaluation code's PSNR convention


def psnr_from_mse(mse: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10((MAX_I ** 2) / np.maximum(mse, 1e-12))


def pool_weights(n_blocks: int, block_size: int, extent: int) -> np.ndarray:
    """Area-overlap matrix mapping a block axis onto the 64-pixel SB axis.

    Returns (n_sb, n_blocks) where entry [s, b] is the number of pixels block b
    contributes to superblock s. Exact for any block_size, including ones that
    do not divide 64 (bs=24 appears in real runs), and for a trailing partial
    superblock.
    """
    n_sb = (extent + SB - 1) // SB
    w = np.zeros((n_sb, n_blocks), dtype=np.float64)
    for b in range(n_blocks):
        b0, b1 = b * block_size, min((b + 1) * block_size, extent)
        if b1 <= b0:
            continue
        for s in range(max(0, b0 // SB), min(n_sb, (b1 - 1) // SB + 1)):
            s0, s1 = s * SB, min((s + 1) * SB, extent)
            overlap = min(b1, s1) - max(b0, s0)
            if overlap > 0:
                w[s, b] = overlap
    return w


def pool_to_superblocks(arr: np.ndarray, block_size: int, height: int, width: int) -> np.ndarray:
    """Area-weighted mean of a (F, BY, BX) block array onto a (F, SY, SX) SB grid."""
    _, by, bx = arr.shape
    wy = pool_weights(by, block_size, height)
    wx = pool_weights(bx, block_size, width)
    # (SY,BY) @ (F,BY,BX) @ (BX,SX), normalized by the pooled pixel area.
    num = np.einsum("yb,fbc,xc->fyx", wy, arr.astype(np.float64), wx)
    den = np.einsum("yb,xc->yx", wy, wx)[None, :, :]
    return num / np.maximum(den, 1e-12)


def superblock_mse(refs, decs) -> np.ndarray:
    """Per-superblock MSE between two equal-length frame lists -> (F, SY, SX).

    Computed directly on the SB grid rather than pooled up from a finer one:
    when both sides are available as pixels there is no reason to inherit a
    run's own block_size and then undo it. Edge superblocks average over their
    real pixels only, which is the same convention `pool_to_superblocks` reaches
    by weighting.
    """
    if not refs or not decs:
        return np.zeros((0, 0, 0), dtype=np.float64)
    n = min(len(refs), len(decs))
    height, width = refs[0].shape[:2]
    n_sy, n_sx = (height + SB - 1) // SB, (width + SB - 1) // SB

    out = np.zeros((n, n_sy, n_sx), dtype=np.float64)
    for f in range(n):
        diff = refs[f].astype(np.float64) - decs[f].astype(np.float64)
        sq = diff * diff
        for sy in range(n_sy):
            y0, y1 = sy * SB, min((sy + 1) * SB, height)
            for sx in range(n_sx):
                x0, x1 = sx * SB, min((sx + 1) * SB, width)
                out[f, sy, sx] = float(sq[y0:y1, x0:x1].mean())
    return out
