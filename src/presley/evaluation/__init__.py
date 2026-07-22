"""Evaluation of encoded/restored results against their references.

Split out of the single 1739-line `presley.components.evaluation` module. The
grouping is by what a reader is looking for:

    masked      region-restricted PSNR/MSE/SSIM and the foreground bounding boxes
    perceptual  LPIPS, DISTS, FID — the metrics the paper's quality claim rests on
    vmaf        VMAF and the yuv420 writing it needs
    fvmd        Frechet Video Motion Distance, reported but never gating
    cache       memoized reference frames and masks (the real cost of a pass)
    run         the evaluation pass over one experiment or a whole tree
    backfill    in-place metric backfills over existing results
    reports     validity diagnostics for FVMD and FID
    cli         the `presley-evaluate` entry point

Names are re-exported here, and `presley.components.evaluation` remains as a
shim, so existing imports and the installed console script keep working.
"""

from presley.evaluation.backfill import (
    backfill_dists,
    backfill_dists_all,
    backfill_fid,
    backfill_fid_all,
    backfill_fvmd,
    backfill_fvmd_all,
    backfill_lpips,
    backfill_lpips_all,
    backfill_transmitted_perceptual,
    backfill_transmitted_perceptual_all,
    backfill_vmaf,
    backfill_vmaf_all,
    drop_unionbbox_keys,
    drop_unionbbox_keys_all,
)
from presley.evaluation.cache import (
    _fvmd_feats_dir,
    _get_masks_cached,
    _get_refs_cached,
    _savez_atomic,
)
from presley.evaluation.cli import main
from presley.evaluation.fvmd import (
    _collect_fvmd_rows,
    _fd_with_terms,
    _fvmd_hist_rows,
    _fvmd_on_frames,
    _fvmd_rows_cached,
    fvmd_set_level,
)
from presley.evaluation.masked import (
    _fg_tight_bbox,
    _fg_union_bbox,
    _masked_mse,
    _masked_psnr,
    _masked_ssim,
)
from presley.evaluation.perceptual import (
    _dists_layer_weights,
    _get_dists_model,
    _inception_feats,
    calculate_dists,
    calculate_dists_masked,
    calculate_fid,
    calculate_fid_bbox,
    calculate_lpips,
    calculate_lpips_masked,
)
from presley.evaluation.reports import (
    fid_validity_report,
    fvmd_setlevel_report,
    fvmd_validity_report,
)
from presley.evaluation.run import evaluate_all, run_evaluation
from presley.evaluation.vmaf import _vmaf_on_frames, _write_yuv420, calculate_vmaf

__all__ = [
    "backfill_dists", "backfill_dists_all", "backfill_fid", "backfill_fid_all",
    "backfill_fvmd", "backfill_fvmd_all", "backfill_lpips", "backfill_lpips_all",
    "backfill_transmitted_perceptual", "backfill_transmitted_perceptual_all",
    "backfill_vmaf", "backfill_vmaf_all", "drop_unionbbox_keys",
    "drop_unionbbox_keys_all",
    "calculate_dists", "calculate_dists_masked", "calculate_fid",
    "calculate_fid_bbox", "calculate_lpips", "calculate_lpips_masked",
    "calculate_vmaf",
    "evaluate_all", "run_evaluation", "main",
    "fvmd_set_level", "fid_validity_report", "fvmd_setlevel_report",
    "fvmd_validity_report",
]
