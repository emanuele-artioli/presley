"""Transmitted side-channel helpers shared by elvis and presley_ai.

Two pieces of the "bridge" recipe live here so both components use one
implementation:

- ``save_binary_masks`` / ``load_binary_masks``: bit-pack the per-block binary
  removal/degradation map (np.packbits + DEFLATE) before it counts toward the
  transmitted budget. The map is 1 bit per block; storing it packed shaves the
  fixed side-channel cost that dominates the *starved* regime (~37 kbps of
  savez'd int8 masks -> a few kbps).
- ``composite_passthrough``: emit already-transmitted pixels everywhere and the
  generated pixels only inside the holes. This decouples foreground quality from
  the generator (restorer/in-painter) -- transmitted FG pixels are reproduced
  bit-exact instead of being re-encoded through the generator's output, which is
  what turns the projected +0.30 dB FG win into a measured one.
"""

import os
import numpy as np


def save_binary_masks(masks_list, path: str) -> int:
    """Persist per-block binary masks bit-packed. Returns on-disk byte size."""
    arr = np.asarray(masks_list).astype(np.uint8)
    packed = np.packbits(arr)
    np.savez_compressed(path, packed=packed, shape=np.array(arr.shape, dtype=np.int64))
    return os.path.getsize(path)


def load_binary_masks(path: str) -> np.ndarray:
    """Inverse of save_binary_masks -> int8 array of the original shape."""
    d = np.load(path)
    shape = tuple(int(x) for x in d['shape'])
    n = int(np.prod(shape))
    return np.unpackbits(d['packed'])[:n].reshape(shape).astype(np.int8)


def composite_passthrough(transmitted_frames, generated_frames, pix_masks):
    """Passthrough compositing: transmitted pixels outside holes, generated inside.

    ``pix_masks[i]`` is a full-resolution boolean array (True = hole to fill from
    the generated frame). Transmitted pixels outside the hole are emitted exactly
    as decoded, so foreground quality is independent of the generator.
    """
    out = []
    for t, g, m in zip(transmitted_frames, generated_frames, pix_masks):
        m3 = m[..., None] if m.ndim == 2 else m
        out.append(np.where(m3, g, t))
    return out
