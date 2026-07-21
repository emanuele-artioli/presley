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


def save_level_masks(maps_list, path: str) -> int:
    """Persist per-block INTEGER strength maps, bit-plane packed. Returns bytes.

    Generalizes ``save_binary_masks`` to the multi-level maps that the
    downsample/blur/noise degradations emit. Those were previously written with
    a bare ``np.savez_compressed`` of int32, which is ~2.3x larger than needed:
    measured on bear bs8, 19793 B raw vs ~8500 B packed (-57%, ~-27 kbps) --
    and the side channel counts toward ``transmitted_size_bytes``, so it is a
    real part of the rate comparison rather than bookkeeping.

    Each bit-plane is packed separately and then DEFLATE'd. A binary map needs
    only one plane, and that case **delegates to ``save_binary_masks`` so the
    file is byte-identical to what it wrote before** -- same npz keys, same
    bytes, same ``os.path.getsize``. That is deliberate and load-bearing:
    ``transmitted_size_bytes`` feeds every rate comparison, so routing the hole
    degradations through this function must not move a single existing number.
    (Writing distinct keys for the 1-plane case would have changed the npz
    member names and therefore the file size.)

    Float maps (noise) are rounded to int; noise is a retired dead end and its
    exact side-channel size is not load-bearing for any claim.
    """
    arr = np.asarray(maps_list)
    arr = np.rint(arr).astype(np.int64)
    if arr.min() < 0:
        raise ValueError("save_level_masks expects non-negative strength maps")
    peak = int(arr.max())
    if peak <= 1:
        return save_binary_masks(arr, path)
    n_planes = int(peak).bit_length()
    planes = np.stack([np.packbits(((arr >> b) & 1).astype(np.uint8)) for b in range(n_planes)])
    np.savez_compressed(path, planes=planes,
                        n_planes=np.int64(n_planes),
                        shape=np.array(arr.shape, dtype=np.int64))
    return os.path.getsize(path)


def load_binary_masks(path: str) -> np.ndarray:
    """Inverse of save_binary_masks -> int8 array of the original shape.

    Also reads the ``save_level_masks`` format and the pre-2026-07-20 raw
    ``strength_maps=`` format (both collapsed to nonzero/zero), so callers that
    only need "was this block degraded" work on any file in results/.
    """
    return (load_level_masks(path) > 0).astype(np.int8)


def load_level_masks(path: str) -> np.ndarray:
    """Inverse of save_level_masks -> int32 array of the original shape.

    Reads every on-disk form present in results/, which is four:
    bit-plane packed (current), single-plane ``packed`` (== save_binary_masks),
    the pre-2026-07-20 raw ``strength_maps`` int32 savez that downsample/blur/
    noise wrote, and an older raw ``masks`` int8 savez from early elvis runs.
    Nothing writes the last two any more, but 44 files in results/ use them and
    a loader that cannot read the archive is a loader that quietly excludes
    experiments from any re-analysis.
    """
    d = np.load(path)
    for legacy_key in ('strength_maps', 'masks'):  # legacy raw savez forms
        if legacy_key in d:
            return d[legacy_key].astype(np.int32)
    shape = tuple(int(x) for x in d['shape'])
    n = int(np.prod(shape))
    if 'packed' in d:  # legacy / single-plane binary file
        return np.unpackbits(d['packed'])[:n].reshape(shape).astype(np.int32)
    n_planes = int(d['n_planes'])
    planes = d['planes']
    if n_planes == 1:
        planes = planes[None, :]
    out = np.zeros(n, dtype=np.int32)
    for b in range(n_planes):
        out |= np.unpackbits(planes[b])[:n].astype(np.int32) << b
    return out.reshape(shape)


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
