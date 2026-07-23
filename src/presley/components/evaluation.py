"""Backwards-compatible shim for the old single-module evaluation.

The implementation moved to the `presley.evaluation` package on 2026-07-22.
This module stays because the installed console script is pinned to
`presley.components.evaluation:main` in an already-created conda env — leaving
it here means the split does not require reinstalling mid-revision, and no
existing `from presley.components.evaluation import ...` breaks.

Prefer importing from `presley.evaluation` (or one of its submodules) in new
code. This file can go once `pyproject.toml`'s entry point is repointed and
the env reinstalled, which is safest after the TOMM revision ships.
"""

from presley.evaluation import *  # noqa: F401,F403
from presley.evaluation import (  # noqa: F401
    _collect_fvmd_rows,
    _dists_layer_weights,
    _fd_with_terms,
    _fg_tight_bbox,
    _fg_union_bbox,
    _fvmd_feats_dir,
    _fvmd_hist_rows,
    _fvmd_on_frames,
    _fvmd_rows_cached,
    _get_dists_model,
    _get_masks_cached,
    _get_refs_cached,
    _inception_feats,
    _masked_mse,
    _masked_psnr,
    _masked_ssim,
    _savez_atomic,
    _vmaf_on_frames,
    _write_yuv420,
    main,
)

if __name__ == "__main__":
    main()
