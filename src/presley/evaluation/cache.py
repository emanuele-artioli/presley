"""Memoized reference frames and masks, plus atomic array writes.

Loading references and masks off NFS dominates evaluation time — far more
than computing the metrics — so `evaluate_all` loads each video's frames
once and reuses them across every experiment on that video."""

import os
import numpy as np
from typing import Dict, Any, List
from presley.preprocessing import get_reference_frames, get_ufo_masks
_REF_CACHE: Dict[Any, Any] = {}
_MASK_CACHE: Dict[Any, Any] = {}
_DISTS_CACHE: Dict[str, Any] = {}


def _get_refs_cached(video_name, width, height, dataset_dir, cache_dir):
    key = (video_name, width, height)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    return _REF_CACHE[key]
def _get_masks_cached(video_name, width, height, block_size, ref_frames_dir, cache_dir, temporal_pool=False):
    key = (video_name, width, height, block_size, temporal_pool)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = get_ufo_masks(video_name, width, height, block_size, ref_frames_dir, cache_dir, temporal_pool=temporal_pool)
    return _MASK_CACHE[key]
def _fvmd_feats_dir(cache_dir: str) -> str:
    d = os.path.join(cache_dir, "fvmd_feats")
    os.makedirs(d, exist_ok=True)
    return d
def _savez_atomic(path: str, **arrays) -> None:
    """Write an .npz via tmp+rename. These are long GPU jobs on a shared box; an
    interrupted in-place savez leaves a truncated file that a later run would
    find via os.path.exists and fail to load."""
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
