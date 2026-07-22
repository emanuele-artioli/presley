"""In-place metric backfills over existing results.

Each of these re-reads the on-disk videos and appends metrics to an
existing result.json. No re-encoding and no re-running of experiments,
so they are safe to apply to the whole tree; all are re-entrant and
skip results that already carry the metric unless forced."""

import os
import json
import numpy as np
import torch
from typing import Dict, Any, List
from presley.preprocessing import get_reference_frames, get_ufo_masks
from presley.encode_utils import load_frames_from_video
_REF_CACHE: Dict[Any, Any] = {}
_MASK_CACHE: Dict[Any, Any] = {}
_DISTS_CACHE: Dict[str, Any] = {}
from presley.evaluation.cache import _get_masks_cached, _get_refs_cached
from presley.evaluation.masked import _fg_union_bbox
from presley.evaluation.fvmd import _fvmd_on_frames
from presley.evaluation.vmaf import _vmaf_on_frames
from presley.evaluation.perceptual import calculate_dists_masked, calculate_fid, calculate_fid_bbox, calculate_lpips_masked


def backfill_lpips(experiment_hash: str, results_dir: str, cache_dir: str, dataset_dir: str,
                   force: bool = False) -> str:
    """Append FG/BG/overall masked LPIPS to an existing result.json in place.

    Metric-only pass: reads the on-disk output video and the (memoized) refs/masks,
    computes region-restricted LPIPS, and writes it into metrics[<region>]["lpips_mean"].
    Does NOT re-encode or recompute any other metric, and works on fast_only results
    too. Skips experiments that already carry a foreground LPIPS unless force=True.
    Returns a one-line status string.
    """
    exp_results_dir = os.path.join(results_dir, experiment_hash)
    result_path = os.path.join(exp_results_dir, "result.json")
    if not os.path.exists(result_path):
        return f"{experiment_hash}: no result.json"
    with open(result_path, 'r') as f:
        data = json.load(f)
    if "metrics" not in data:
        return f"{experiment_hash}: no metrics yet (run eval first)"
    if not force and "lpips_mean" in data["metrics"].get("foreground", {}):
        return f"{experiment_hash}: FG-LPIPS already present"

    cfg = data['config']
    video_name, width, height = cfg['video'], cfg['width'], cfg['height']
    block_size = cfg.get('block_size', 8)
    output_video = data.get('output_video')
    if not output_video or not os.path.exists(output_video):
        return f"{experiment_hash}: output video missing"

    _, refs, _ = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    # Always the true per-frame FG mask -- see the matching note in run_evaluation.
    ufo_masks = _get_masks_cached(video_name, width, height, block_size, ref_frames_dir, cache_dir)
    decs = load_frames_from_video(output_video)

    n = min(len(refs), len(decs), len(ufo_masks))
    masks = [ufo_masks[i] > 127 for i in range(n)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lp = calculate_lpips_masked(refs[:n], decs[:n], masks, device)
    for region in ("foreground", "background", "overall"):
        data["metrics"].setdefault(region, {})
        vals = lp[region]
        # nanmean, and NaN -> None: empty-mask frames are excluded rather than counted
        # as a perfect 0.0, and a bare NaN literal would be invalid JSON.
        if np.any(~np.isnan(vals)):
            data["metrics"][region]["lpips_mean"] = float(np.nanmean(vals))
            data["metrics"][region]["lpips_std"] = float(np.nanstd(vals))
        else:
            data["metrics"][region]["lpips_mean"] = None
            data["metrics"][region]["lpips_std"] = None
        data["metrics"][region]["lpips_n_valid"] = int(np.count_nonzero(~np.isnan(vals)))

    tmp = result_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, result_path)

    def _f(region):
        v = data["metrics"][region]["lpips_mean"]
        return "n/a" if v is None else f"{v:.4f}"
    return (f"{experiment_hash}: FG-LPIPS={_f('foreground')} BG={_f('background')} "
            f"OV={_f('overall')}")
def backfill_vmaf(experiment_hash: str, results_dir: str, cache_dir: str, dataset_dir: str,
                  force: bool = False) -> str:
    """Append overall + FG-crop VMAF (default and NEG models) to an existing
    result.json in place.

    Metric-only pass over on-disk artifacts, like backfill_lpips: no re-encode.
    The FG value is written as `vmaf_fg_bbox` (not `vmaf_mean`) because it is
    computed on the per-video union FG bounding-box crop (VMAF needs
    constant-resolution natural frames; a mask cannot be applied directly), so
    it includes substantial BG context within the box and is NOT a foreground
    metric — see TECHNICAL_REPORT_PIPELINE_INFRA.md 2026-07-16 and
    `compare.py`'s BANNED_FG_KEYS. Comparisons are within-video at matched
    bitrate, where the box is identical across methods, but the value must
    never be cited for the FG chain claim. The NEG model discounts
    enhancement/sharpening gains; reporting both makes the
    sharpening-vs-fidelity split explicit.
    """
    exp_results_dir = os.path.join(results_dir, experiment_hash)
    result_path = os.path.join(exp_results_dir, "result.json")
    if not os.path.exists(result_path):
        return f"{experiment_hash}: no result.json"
    with open(result_path, 'r') as f:
        data = json.load(f)
    if "metrics" not in data:
        return f"{experiment_hash}: no metrics yet (run eval first)"
    if not force and "vmaf_fg_bbox" in data["metrics"].get("foreground", {}):
        return f"{experiment_hash}: FG-VMAF already present"

    cfg = data['config']
    video_name, width, height = cfg['video'], cfg['width'], cfg['height']
    block_size = cfg.get('block_size', 8)
    output_video = data.get('output_video')
    if not output_video or not os.path.exists(output_video):
        return f"{experiment_hash}: output video missing"

    _, refs, _ = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    ufo_masks = _get_masks_cached(video_name, width, height, block_size, ref_frames_dir, cache_dir)
    decs = load_frames_from_video(output_video)

    n = min(len(refs), len(decs), len(ufo_masks))
    if n == 0:
        return f"{experiment_hash}: no decodable frames"
    masks = [ufo_masks[i] > 127 for i in range(n)]

    ov = _vmaf_on_frames(refs[:n], decs[:n])
    ov_neg = _vmaf_on_frames(refs[:n], decs[:n], neg=True)
    bb = _fg_union_bbox(masks, width, height)
    fg = fg_neg = {"mean": 0.0, "std": 0.0}
    if bb:
        y1, y2, x1, x2 = bb
        ref_c = [refs[i][y1:y2, x1:x2] for i in range(n)]
        dec_c = [decs[i][y1:y2, x1:x2] for i in range(n)]
        fg = _vmaf_on_frames(ref_c, dec_c)
        fg_neg = _vmaf_on_frames(ref_c, dec_c, neg=True)

    m = data["metrics"]
    m.setdefault("overall", {})["vmaf_mean"] = ov["mean"]
    m["overall"]["vmaf_std"] = ov["std"]
    m["overall"]["vmaf_neg_mean"] = ov_neg["mean"]
    # Named _fg_bbox (not vmaf_mean/vmaf_neg_mean): this is a union-bbox crop,
    # not a mask metric, and must never be read as a foreground quality signal
    # (see docstring above and compare.py's BANNED_FG_KEYS).
    m.setdefault("foreground", {})["vmaf_fg_bbox"] = fg["mean"]
    m["foreground"]["vmaf_std_fg_bbox"] = fg["std"]
    m["foreground"]["vmaf_neg_fg_bbox"] = fg_neg["mean"]

    tmp = result_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, result_path)
    return (f"{experiment_hash}: OV-VMAF={ov['mean']:.2f} (neg {ov_neg['mean']:.2f}) "
            f"FG-VMAF={fg['mean']:.2f} (neg {fg_neg['mean']:.2f})")
def backfill_vmaf_all(results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> None:
    for entry in sorted(os.listdir(results_dir)):
        if os.path.isdir(os.path.join(results_dir, entry)):
            print(backfill_vmaf(entry, results_dir, cache_dir, dataset_dir, force=force))
def backfill_dists(experiment_hash: str, results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> str:
    """Append overall + true mask-weighted FG/BG DISTS to an existing result.json.

    Writes `foreground.dists_fg` / `background.dists_bg` (true masked, via
    `calculate_dists_masked`) and `overall.dists_mean` (whole-frame, unchanged in
    meaning -- the masked pass reproduces it exactly under uniform weights).

    The old union-bbox `foreground.dists_mean` is NOT written any more; it is removed
    separately by `drop_unionbbox_keys`. The sentinel below changed with it, so the
    corpus recomputes once naturally -- no --force needed.
    """
    exp_results_dir = os.path.join(results_dir, experiment_hash)
    result_path = os.path.join(exp_results_dir, "result.json")
    if not os.path.exists(result_path): return f"{experiment_hash}: no result.json"
    with open(result_path, 'r') as f: data = json.load(f)
    if "metrics" not in data: return f"{experiment_hash}: no metrics yet"
    if not force and "dists_fg" in data["metrics"].get("foreground", {}): return f"{experiment_hash}: masked FG-DISTS already present"

    cfg = data['config']
    video_name, width, height = cfg['video'], cfg['width'], cfg['height']
    block_size = cfg.get('block_size', 8)
    output_video = data.get('output_video')
    if not output_video or not os.path.exists(output_video): return f"{experiment_hash}: output video missing"

    _, refs, _ = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    ufo_masks = _get_masks_cached(video_name, width, height, block_size, ref_frames_dir, cache_dir)
    decs = load_frames_from_video(output_video)

    n = min(len(refs), len(decs), len(ufo_masks))
    if n == 0: return f"{experiment_hash}: no decodable frames"
    masks = [ufo_masks[i] > 127 for i in range(n)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # One masked pass yields all three regions -- this replaces the previous two passes
    # (full frame + union-bbox crop), so it is cheaper as well as correct.
    ds = calculate_dists_masked(refs[:n], decs[:n], masks, device)

    def _agg(vals):
        """(mean, std) as JSON-safe values: NaN -> None, which json.dump writes as
        `null`. A bare float('nan') would be dumped as the literal `NaN`, which is
        invalid JSON and rejected by strict parsers. Mirrors backfill_fvmd."""
        if not np.any(~np.isnan(vals)):
            return None, None
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    m = data["metrics"]
    ov_mean, ov_std = _agg(ds["overall"])
    m.setdefault("overall", {})["dists_mean"] = ov_mean
    m["overall"]["dists_std"] = ov_std
    fg = m.setdefault("foreground", {})
    fg["dists_fg"], fg["dists_fg_std"] = _agg(ds["foreground"])
    # frames contributing to the FG number; < n means some frames had an empty mask
    fg["dists_fg_n_valid"] = int(np.count_nonzero(~np.isnan(ds["foreground"])))
    bgm = m.setdefault("background", {})
    bgm["dists_bg"], bgm["dists_bg_std"] = _agg(ds["background"])

    tmp = result_path + ".tmp"
    with open(tmp, 'w') as f: json.dump(data, f, indent=2)
    os.replace(tmp, result_path)
    def _f(x):
        return "n/a" if x is None else f"{x:.4f}"
    return (f"{experiment_hash}: OV-DISTS={_f(m['overall']['dists_mean'])} "
            f"FG(masked)={_f(fg['dists_fg'])} BG(masked)={_f(bgm['dists_bg'])} "
            f"n_valid={fg['dists_fg_n_valid']}/{n}")
def backfill_dists_all(results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> None:
    for entry in sorted(os.listdir(results_dir)):
        if os.path.isdir(os.path.join(results_dir, entry)) and not entry.startswith('_'):
            print(backfill_dists(entry, results_dir, cache_dir, dataset_dir, force=force))
def backfill_transmitted_perceptual(experiment_hash: str, results_dir: str, cache_dir: str,
                                    dataset_dir: str, force: bool = False) -> str:
    """Append masked LPIPS + masked DISTS for the TRANSMITTED (decoded degraded)
    video to metrics.transmitted.<region>, vs the original references.

    This is what makes the restoration *perceptual* gain measurable
    (metrics.<region>.lpips_mean - metrics.transmitted.<region>.lpips_mean):
    until now only PSNR/SSIM/MSE existed under metrics.transmitted, and a
    PSNR-only gain systematically understates hallucinated detail (the
    mean_fill trap, in reverse). Metric-only pass like backfill_lpips: no
    re-encode. Skips experiments without a native-resolution transmitted
    video (elvis shrink packs blocks into a smaller rectangle; the FG/BG mask
    is not pixel-comparable to that geometry -- same skip as run_evaluation).
    """
    exp_results_dir = os.path.join(results_dir, experiment_hash)
    result_path = os.path.join(exp_results_dir, "result.json")
    if not os.path.exists(result_path): return f"{experiment_hash}: no result.json"
    with open(result_path, 'r') as f: data = json.load(f)
    if "metrics" not in data: return f"{experiment_hash}: no metrics yet"
    transmitted_video = data.get('transmitted_video')
    if not transmitted_video or not os.path.exists(transmitted_video):
        return f"{experiment_hash}: no transmitted video"
    trans = data["metrics"].setdefault("transmitted", {})
    if not force and "lpips_mean" in trans.get("foreground", {}):
        return f"{experiment_hash}: transmitted perceptual already present"

    cfg = data['config']
    video_name, width, height = cfg['video'], cfg['width'], cfg['height']
    block_size = cfg.get('block_size', 8)

    decs = load_frames_from_video(transmitted_video)
    if not decs: return f"{experiment_hash}: transmitted video not decodable"
    if decs[0].shape[:2] != (height, width):
        th, tw = decs[0].shape[:2]
        return (f"{experiment_hash}: transmitted is {tw}x{th}, native {width}x{height} "
                f"(packed removal geometry) -- skipped")

    _, refs, _ = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    # Always the true per-frame FG mask -- see the matching note in run_evaluation.
    ufo_masks = _get_masks_cached(video_name, width, height, block_size, ref_frames_dir, cache_dir)
    n = min(len(refs), len(decs), len(ufo_masks))
    if n == 0: return f"{experiment_hash}: no comparable frames"
    masks = [ufo_masks[i] > 127 for i in range(n)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lp = calculate_lpips_masked(refs[:n], decs[:n], masks, device)
    ds = calculate_dists_masked(refs[:n], decs[:n], masks, device)

    def _agg(vals):
        # NaN -> None (JSON null); empty-mask frames excluded, not counted as 0.0.
        if not np.any(~np.isnan(vals)):
            return None, None
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    for region in ("foreground", "background", "overall"):
        r = trans.setdefault(region, {})
        r["lpips_mean"], r["lpips_std"] = _agg(lp[region])
        r["lpips_n_valid"] = int(np.count_nonzero(~np.isnan(lp[region])))
    fg = trans["foreground"]
    fg["dists_fg"], fg["dists_fg_std"] = _agg(ds["foreground"])
    fg["dists_fg_n_valid"] = int(np.count_nonzero(~np.isnan(ds["foreground"])))
    trans["background"]["dists_bg"], trans["background"]["dists_bg_std"] = _agg(ds["background"])
    trans["overall"]["dists_mean"], trans["overall"]["dists_std"] = _agg(ds["overall"])

    tmp = result_path + ".tmp"
    with open(tmp, 'w') as f: json.dump(data, f, indent=2)
    os.replace(tmp, result_path)
    def _f(x):
        return "n/a" if x is None else f"{x:.4f}"
    return (f"{experiment_hash}: transmitted BG-LPIPS={_f(trans['background']['lpips_mean'])} "
            f"BG-DISTS={_f(trans['background']['dists_bg'])} "
            f"FG-LPIPS={_f(fg['lpips_mean'])} n={n}")
def backfill_transmitted_perceptual_all(results_dir: str, cache_dir: str, dataset_dir: str,
                                        force: bool = False) -> None:
    for entry in sorted(os.listdir(results_dir)):
        if os.path.isdir(os.path.join(results_dir, entry)) and not entry.startswith('_'):
            print(backfill_transmitted_perceptual(entry, results_dir, cache_dir, dataset_dir, force=force))
def drop_unionbbox_keys(experiment_hash: str, results_dir: str) -> str:
    """Delete the union-bbox "FG" DISTS/FID values from metrics.foreground.

    `metrics.foreground.dists_mean` and `metrics.foreground.fid` were computed on
    `_fg_union_bbox`, which is not a foreground region (100% of the frame on india --
    where FG-DISTS was measured bit-identical to overall-DISTS on 16/16 experiments --
    and 58.6% on tennis against a 4.0% true FG). They are superseded by `dists_fg` and
    `fid_fg_bbox`, which are written under new key names, so these would otherwise sit
    stale under a node called `foreground` where a future script or session would read
    them as foreground numbers. That is exactly the failure this deletion prevents.

    Their values are preserved in scratch/metric_audit_pre.tsv purely so the delta
    report can quantify how far the corrected numbers moved -- they must never be cited.

    `metrics.overall.dists_mean`/`dists_std`/`fid` are legitimate whole-frame metrics and
    are NOT touched. Idempotent; a no-op once the keys are gone. No recomputation.
    """
    result_path = os.path.join(results_dir, experiment_hash, "result.json")
    if not os.path.exists(result_path):
        return f"{experiment_hash}: no result.json"
    with open(result_path, 'r') as f:
        data = json.load(f)
    fg = data.get("metrics", {}).get("foreground", {})
    dropped = [k for k in ("dists_mean", "fid") if k in fg]
    if not dropped:
        return f"{experiment_hash}: no union-bbox FG keys (already dropped)"
    for k in dropped:
        del fg[k]
    tmp = result_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, result_path)
    return f"{experiment_hash}: dropped foreground.{', foreground.'.join(dropped)}"
def drop_unionbbox_keys_all(results_dir: str) -> None:
    for entry in sorted(os.listdir(results_dir)):
        if os.path.isdir(os.path.join(results_dir, entry)) and not entry.startswith('_'):
            print(drop_unionbbox_keys(entry, results_dir))
def backfill_fid(experiment_hash: str, results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> str:
    exp_results_dir = os.path.join(results_dir, experiment_hash)
    result_path = os.path.join(exp_results_dir, "result.json")
    if not os.path.exists(result_path): return f"{experiment_hash}: no result.json"
    with open(result_path, 'r') as f: data = json.load(f)
    if "metrics" not in data: return f"{experiment_hash}: no metrics yet"
    if not force and "fid_fg_bbox" in data["metrics"].get("foreground", {}): return f"{experiment_hash}: fid_fg_bbox already present"

    cfg = data['config']
    video_name, width, height = cfg['video'], cfg['width'], cfg['height']
    block_size = cfg.get('block_size', 8)
    output_video = data.get('output_video')
    if not output_video or not os.path.exists(output_video): return f"{experiment_hash}: output video missing"

    _, refs, _ = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    ufo_masks = _get_masks_cached(video_name, width, height, block_size, ref_frames_dir, cache_dir)
    decs = load_frames_from_video(output_video)

    n = min(len(refs), len(decs), len(ufo_masks))
    if n == 0: return f"{experiment_hash}: no decodable frames"
    masks = [ufo_masks[i] > 127 for i in range(n)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ov_fid = calculate_fid(refs[:n], decs[:n], device)
    bx = calculate_fid_bbox(refs[:n], decs[:n], masks, device)

    m = data["metrics"]
    m.setdefault("overall", {})["fid"] = float(ov_fid)
    fg = m.setdefault("foreground", {})
    # Named fid_fg_bbox, never fid_fg: this is a per-frame bbox crop, still largely
    # background (see bg_frac below), and is not a foreground metric.
    # NaN -> None so json.dump writes `null`; a bare NaN literal is invalid JSON.
    def _j(x):
        return None if x is None or np.isnan(x) else float(x)
    fg["fid_fg_bbox"] = _j(bx["fid"])
    # Scale/coverage diagnostics -- per-frame boxes vary in size, which is variance that
    # whole-frame FID does not carry, and bg_frac is what justifies the key's name.
    fg["fid_fg_bbox_area_frac_mean"] = _j(bx["area_frac_mean"])
    fg["fid_fg_bbox_area_frac_std"] = _j(bx["area_frac_std"])
    fg["fid_fg_bbox_scale_cv"] = _j(bx["scale_cv"])
    fg["fid_fg_bbox_bg_frac_mean"] = _j(bx["bg_frac_mean"])
    fg["fid_fg_bbox_n_used"] = int(bx["n_used"])
    fg["fid_fg_bbox_n_skipped_empty"] = int(bx["n_skipped_empty"])

    tmp = result_path + ".tmp"
    with open(tmp, 'w') as f: json.dump(data, f, indent=2)
    os.replace(tmp, result_path)
    if not bx["n_used"]:
        return f"{experiment_hash}: OV-FID={ov_fid:.2f} fid_fg_bbox=n/a (no non-empty FG masks)"
    return (f"{experiment_hash}: OV-FID={ov_fid:.2f} fid_fg_bbox={bx['fid']:.2f} "
            f"(box {bx['area_frac_mean']*100:.1f}% of frame, {bx['bg_frac_mean']*100:.0f}% BG inside, "
            f"scale_cv={bx['scale_cv']:.3f}, n={bx['n_used']}/{n})")
def backfill_fid_all(results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> None:
    for entry in sorted(os.listdir(results_dir)):
        if os.path.isdir(os.path.join(results_dir, entry)) and not entry.startswith('_'):
            print(backfill_fid(entry, results_dir, cache_dir, dataset_dir, force=force))
def backfill_fvmd(experiment_hash: str, results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> str:
    exp_results_dir = os.path.join(results_dir, experiment_hash)
    result_path = os.path.join(exp_results_dir, "result.json")
    if not os.path.exists(result_path): return f"{experiment_hash}: no result.json"
    with open(result_path, 'r') as f: data = json.load(f)
    if "metrics" not in data: return f"{experiment_hash}: no metrics yet"
    if not force and "fvmd" in data["metrics"].get("foreground", {}): return f"{experiment_hash}: FG-FVMD already present"
    
    cfg = data['config']
    video_name, width, height = cfg['video'], cfg['width'], cfg['height']
    block_size = cfg.get('block_size', 8)
    output_video = data.get('output_video')
    if not output_video or not os.path.exists(output_video): return f"{experiment_hash}: output video missing"

    _, refs, _ = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    ufo_masks = _get_masks_cached(video_name, width, height, block_size, ref_frames_dir, cache_dir)
    decs = load_frames_from_video(output_video)

    n = min(len(refs), len(decs), len(ufo_masks))
    if n == 0: return f"{experiment_hash}: no decodable frames"
    masks = [ufo_masks[i] > 127 for i in range(n)]
    
    ov_fvmd = _fvmd_on_frames(refs[:n], decs[:n])
    bb = _fg_union_bbox(masks, width, height)
    fg_fvmd = float('nan')
    if bb:
        y1, y2, x1, x2 = bb
        ref_c = [refs[i][y1:y2, x1:x2] for i in range(n)]
        dec_c = [decs[i][y1:y2, x1:x2] for i in range(n)]
        fg_fvmd = _fvmd_on_frames(ref_c, dec_c)

    # Store None (JSON null) rather than 0.0/NaN on failure or absent FG, so a
    # genuine failure is distinguishable from a real value downstream.
    def _clean(x):
        return None if (x is None or np.isnan(x)) else float(x)
    m = data["metrics"]
    m.setdefault("overall", {})["fvmd"] = _clean(ov_fvmd)
    m.setdefault("foreground", {})["fvmd"] = _clean(fg_fvmd)

    tmp = result_path + ".tmp"
    with open(tmp, 'w') as f: json.dump(data, f, indent=2)
    os.replace(tmp, result_path)
    return f"{experiment_hash}: OV-FVMD={ov_fvmd} FG-FVMD={fg_fvmd}"
def backfill_fvmd_all(results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False, shard: str = None) -> None:
    import concurrent.futures
    entries = [entry for entry in sorted(os.listdir(results_dir)) 
               if os.path.isdir(os.path.join(results_dir, entry)) and not entry.startswith('_')]
    if shard:
        idx, total = map(int, shard.split('/'))
        entries = [e for i, e in enumerate(entries) if i % total == idx]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(backfill_fvmd, entry, results_dir, cache_dir, dataset_dir, force): entry for entry in entries}
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res:
                    print(res)
            except Exception as e:
                print(f"Error on {futures[future]}: {e}")
def backfill_lpips_all(results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> None:
    for entry in sorted(os.listdir(results_dir)):
        if os.path.isdir(os.path.join(results_dir, entry)):
            print(backfill_lpips(entry, results_dir, cache_dir, dataset_dir, force=force))
