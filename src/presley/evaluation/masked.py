"""Region-restricted PSNR/MSE/SSIM and the foreground bounding boxes.

The mask is what makes a number a foreground number. An all-false mask
falls back to the whole frame by design, so callers must not present an
FG metric without knowing a mask was present.

Both bbox helpers return (y1, y2, x1, x2) — row bounds first."""

import numpy as np
import cv2
from typing import Dict, Any, List
from presley.preprocessing import get_reference_frames, get_ufo_masks
_REF_CACHE: Dict[Any, Any] = {}
_MASK_CACHE: Dict[Any, Any] = {}
_DISTS_CACHE: Dict[str, Any] = {}
from skimage.metrics import structural_similarity as ssim


def _masked_psnr(ref: np.ndarray, dec: np.ndarray, mask: np.ndarray = None) -> float:
    if ref is None or dec is None: return 0.0
    ref_f, dec_f = ref.astype(np.float32), dec.astype(np.float32)
    diff = ref_f[mask] - dec_f[mask] if mask is not None and np.any(mask) else ref_f - dec_f
    mse = float(np.mean(diff ** 2)) if diff.size else 0.0
    if mse < 1e-10: return 100.0
    return float(min(20 * np.log10(255.0 / np.sqrt(mse)), 100.0))
def _masked_mse(ref: np.ndarray, dec: np.ndarray, mask: np.ndarray = None) -> float:
    if ref is None or dec is None: return 0.0
    ref_f, dec_f = ref.astype(np.float32), dec.astype(np.float32)
    diff = ref_f[mask] - dec_f[mask] if mask is not None and np.any(mask) else ref_f - dec_f
    return float(np.mean(diff ** 2)) if diff.size else 0.0
def _masked_ssim(ref: np.ndarray, dec: np.ndarray, mask: np.ndarray = None) -> float:
    if ref is None or dec is None: return 0.0
    ref_y = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    dec_y = cv2.cvtColor(dec, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    
    if mask is not None:
        if not np.any(mask): return 1.0
        ys, xs = np.where(mask)
        y1, y2, x1, x2 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        ref_y = ref_y[y1:y2, x1:x2].copy()
        dec_y = dec_y[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2]
        ref_y[~mask_crop] = 0
        dec_y[~mask_crop] = 0
        
    h, w = ref_y.shape[:2]
    smallest_dim = min(h, w)
    if smallest_dim < 3: return 1.0
    win_size = min(7, smallest_dim if smallest_dim % 2 == 1 else max(3, smallest_dim - 1))
    return float(ssim(ref_y, dec_y, data_range=255, gaussian_weights=True, win_size=win_size))
def _fg_tight_bbox(mask: np.ndarray, w: int, h: int, pad: int = 8):
    """Per-frame tight FG bounding box. Returns (y1, y2, x1, x2), or None if empty.

    Unlike `_fg_union_bbox` this does NOT union across frames. The union box is
    background-dominated on every video (100% of the frame on india, 58.6% on tennis
    against a 4.0% true FG), and any metric built on it is not a foreground metric --
    see TECHNICAL_REPORT_PIPELINE_INFRA.md 2026-07-16. A per-frame box is 1.3-3.8x
    tighter (tennis 58.6% -> 15.2%, dog 64.8% -> 23.6%) but is STILL
    background-dominated (~74% BG on tennis), which is exactly why its only caller
    writes the key `fid_fg_bbox` and never `fid_fg`.

    No even-alignment here: that constraint exists in `_fg_union_bbox` only because its
    crop is re-encoded as yuv420. This crop is fed straight to Inception. `pad` matches
    `_fg_union_bbox` so the two boxes stay comparable.
    """
    yy, xx = np.where(mask)
    if not len(yy):
        return None
    y1, y2 = max(0, yy.min() - pad), min(h, yy.max() + 1 + pad)
    x1, x2 = max(0, xx.min() - pad), min(w, xx.max() + 1 + pad)
    return y1, y2, x1, x2
def _fg_union_bbox(masks: List[np.ndarray], w: int, h: int, pad: int = 8):
    """Union FG bounding box across frames, padded and even-aligned for yuv420.

    WARNING: this box is not a foreground region -- it is 100% of the frame on india
    and 58.6% on tennis (true FG 4.0%). Do not build a new "FG" metric on it; see
    `_fg_tight_bbox` and TECHNICAL_REPORT_PIPELINE_INFRA.md 2026-07-16. Retained for
    the VMAF backfill, whose FG numbers are already excluded from the paper's FG claim.
    """
    ys, xs = [], []
    for m in masks:
        yy, xx = np.where(m)
        if len(yy):
            ys += [yy.min(), yy.max()]; xs += [xx.min(), xx.max()]
    if not ys:
        return None
    y1, y2 = max(0, min(ys) - pad), min(h, max(ys) + 1 + pad)
    x1, x2 = max(0, min(xs) - pad), min(w, max(xs) + 1 + pad)
    # even-align (yuv420 requires even dimensions)
    y1, x1 = y1 - (y1 % 2), x1 - (x1 % 2)
    if (y2 - y1) % 2: y2 = min(h, y2 + 1) if y2 < h else y2 - 1
    if (x2 - x1) % 2: x2 = min(w, x2 + 1) if x2 < w else x2 - 1
    return y1, y2, x1, x2
