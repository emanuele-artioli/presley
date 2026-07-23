"""The evaluation pass over one experiment or a whole results tree."""

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
from presley.evaluation.masked import _masked_mse, _masked_psnr, _masked_ssim
from presley.evaluation.perceptual import calculate_dists, calculate_lpips
from presley.evaluation.vmaf import calculate_vmaf


def run_evaluation(experiment_hash: str, results_dir: str, cache_dir: str, dataset_dir: str, fast: bool = False) -> None:
    """Compute metrics for one experiment.

    fast=True computes only the cheap frame-level metrics (foreground/background/
    overall PSNR/SSIM/MSE) and skips the slow ones (LPIPS, DISTS, VMAF, FVMD and
    the per-block loop). Fast-only results are tagged with metrics["fast_only"]
    so a later full evaluation upgrades them in place.
    """
    exp_results_dir = os.path.join(results_dir, experiment_hash)
    result_path = os.path.join(exp_results_dir, "result.json")

    if not os.path.exists(result_path):
        print(f"Result JSON not found for {experiment_hash}")
        return

    with open(result_path, 'r') as f:
        data = json.load(f)

    if "metrics" in data:
        if fast or not data["metrics"].get("fast_only"):
            print(f"Metrics already computed for {experiment_hash}")
            return
        print(f"Upgrading fast-only metrics to full for {experiment_hash}")
        
    video_name = data['config']['video']
    width = data['config']['width']
    height = data['config']['height']
    block_size = data['config'].get('block_size', 8)
    
    raw_yuv_path, refs, framerate = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    # Evaluation always measures against the TRUE per-frame FG mask, regardless
    # of temporal_pool_masks (an encoding-time selection knob). Coupling them
    # inflated bmx-trees's measured FG region from ~5% to 46% of the frame and
    # fabricated a +2dB "win" that vanished under the true mask (see
    # PIPELINE_INFRA report, 2026-07-11 entry) -- ground-truth FG/BG regions
    # must stay a fixed definition independent of any method's own mask usage.
    ufo_masks = _get_masks_cached(video_name, width, height, block_size, ref_frames_dir, cache_dir)

    output_video = data.get('output_video')
    if not output_video or not os.path.exists(output_video):
        print(f"Output video missing for {experiment_hash}")
        return
        
    decs = load_frames_from_video(output_video)
    
    num_frames = min(len(refs), len(decs), len(ufo_masks))
    
    fg_psnr, fg_ssim, fg_mse = [], [], []
    bg_psnr, bg_ssim, bg_mse = [], [], []
    ov_psnr, ov_ssim, ov_mse = [], [], []
    
    transmitted_video = data.get('transmitted_video')
    has_transmitted = transmitted_video and os.path.exists(transmitted_video)
    if has_transmitted:
        trans_decs = load_frames_from_video(transmitted_video)
        num_frames = min(num_frames, len(trans_decs))
        # elvis's shrink removal_mode packs surviving blocks into a smaller
        # rectangle, so its transmitted_video is sub-native resolution (unlike
        # blackout/freeze/presley_ai/baselines, whose transmitted video is
        # always native). The native-resolution FG/BG mask isn't pixel-
        # comparable to that packed geometry -- skip the transmitted-quality
        # metric rather than crash the whole evaluation on the boolean-index
        # shape mismatch.
        if trans_decs and trans_decs[0].shape[:2] != (height, width):
            th, tw = trans_decs[0].shape[:2]
            print(f"Transmitted video for {experiment_hash} is {tw}x{th}, native is "
                  f"{width}x{height} (packed removal geometry) -- skipping transmitted-quality metric")
            has_transmitted = False
        else:
            t_fg_psnr, t_fg_ssim, t_fg_mse = [], [], []
            t_bg_psnr, t_bg_ssim, t_bg_mse = [], [], []
            t_ov_psnr, t_ov_ssim, t_ov_mse = [], [], []
    
    # Block level metrics (slow: full per-block loop; skipped in fast mode)
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size
    if not fast:
        block_psnr = np.zeros((num_frames, num_blocks_y, num_blocks_x), dtype=np.float32)
        block_ssim = np.zeros((num_frames, num_blocks_y, num_blocks_x), dtype=np.float32)
        block_mse  = np.zeros((num_frames, num_blocks_y, num_blocks_x), dtype=np.float32)
        from presley.degradation import split_image_into_blocks

    for i in range(num_frames):
        r = refs[i]
        d = decs[i]
        m = ufo_masks[i] > 127

        # Frame level. Fast mode computes PSNR+MSE only (both ~free from one diff);
        # SSIM (skimage, the slowest frame metric) is deferred to the full pass.
        fg_psnr.append(_masked_psnr(r, d, m))
        fg_mse.append(_masked_mse(r, d, m))
        bg_psnr.append(_masked_psnr(r, d, ~m))
        bg_mse.append(_masked_mse(r, d, ~m))
        ov_psnr.append(_masked_psnr(r, d))
        ov_mse.append(_masked_mse(r, d))

        if fast:
            if has_transmitted:
                td = trans_decs[i]
                t_fg_psnr.append(_masked_psnr(r, td, m))
                t_fg_mse.append(_masked_mse(r, td, m))
                t_bg_psnr.append(_masked_psnr(r, td, ~m))
                t_bg_mse.append(_masked_mse(r, td, ~m))
                t_ov_psnr.append(_masked_psnr(r, td))
                t_ov_mse.append(_masked_mse(r, td))
            continue

        fg_ssim.append(_masked_ssim(r, d, m))
        bg_ssim.append(_masked_ssim(r, d, ~m))
        ov_ssim.append(_masked_ssim(r, d))

        if has_transmitted:
            td = trans_decs[i]
            t_fg_psnr.append(_masked_psnr(r, td, m))
            t_fg_mse.append(_masked_mse(r, td, m))
            t_bg_psnr.append(_masked_psnr(r, td, ~m))
            t_bg_mse.append(_masked_mse(r, td, ~m))
            t_ov_psnr.append(_masked_psnr(r, td))
            t_ov_mse.append(_masked_mse(r, td))
            
            t_fg_ssim.append(_masked_ssim(r, td, m))
            t_bg_ssim.append(_masked_ssim(r, td, ~m))
            t_ov_ssim.append(_masked_ssim(r, td))

        # Block level. Crop to a whole number of blocks first: resolutions whose
        # H/W aren't a multiple of block_size (e.g. 540 % 8 != 0) otherwise make
        # split_image_into_blocks' reshape raise and abort the whole eval pass.
        hb, wb = num_blocks_y * block_size, num_blocks_x * block_size
        r_b = split_image_into_blocks(r[:hb, :wb], block_size)
        d_b = split_image_into_blocks(d[:hb, :wb], block_size)

        for by in range(num_blocks_y):
            for bx in range(num_blocks_x):
                block_psnr[i, by, bx] = _masked_psnr(r_b[by, bx], d_b[by, bx])
                block_ssim[i, by, bx] = _masked_ssim(r_b[by, bx], d_b[by, bx])
                block_mse[i, by, bx]  = _masked_mse(r_b[by, bx], d_b[by, bx])

    if not fast:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lpips_vals = calculate_lpips(refs[:num_frames], decs[:num_frames], device)
        dists_vals = calculate_dists(refs[:num_frames], decs[:num_frames], device)
        vmaf_data = calculate_vmaf(raw_yuv_path, output_video, width, height, framerate)

        np.savez_compressed(os.path.join(exp_results_dir, "block_psnr.npz"), block_psnr)
        np.savez_compressed(os.path.join(exp_results_dir, "block_ssim.npz"), block_ssim)
        np.savez_compressed(os.path.join(exp_results_dir, "block_mse.npz"), block_mse)

    def _block(psnr, ssim, mse):
        d = {
            "psnr_mean": float(np.mean(psnr)), "psnr_std": float(np.std(psnr)),
            "mse_mean": float(np.mean(mse)), "mse_std": float(np.std(mse)),
        }
        if not fast:  # SSIM only computed in the full pass
            d["ssim_mean"] = float(np.mean(ssim))
            d["ssim_std"] = float(np.std(ssim))
        return d

    metrics = {
        "foreground": _block(fg_psnr, fg_ssim, fg_mse),
        "background": _block(bg_psnr, bg_ssim, bg_mse),
        "overall": _block(ov_psnr, ov_ssim, ov_mse),
    }
    
    if has_transmitted:
        metrics["transmitted"] = {
            "foreground": _block(t_fg_psnr, t_fg_ssim, t_fg_mse),
            "background": _block(t_bg_psnr, t_bg_ssim, t_bg_mse),
            "overall": _block(t_ov_psnr, t_ov_ssim, t_ov_mse),
        }

    if fast:
        metrics["fast_only"] = True
    else:
        metrics["overall"].update({
            "lpips_mean": float(np.mean(lpips_vals)), "lpips_std": float(np.std(lpips_vals)),
            "dists_mean": float(np.mean(dists_vals)), "dists_std": float(np.std(dists_vals)),
            "vmaf_mean": vmaf_data["mean"], "vmaf_std": vmaf_data["std"],
        })
        metrics["block_level"] = {
            "psnr": {"shape": list(block_psnr.shape), "path": "block_psnr.npz"},
            "ssim": {"shape": list(block_ssim.shape), "path": "block_ssim.npz"},
            "mse":  {"shape": list(block_mse.shape), "path": "block_mse.npz"}
        }

    data["metrics"] = metrics
    # Atomic write: a crash mid-rewrite would otherwise truncate result.json and
    # force a full re-run of a potentially hours-long experiment.
    tmp_path = result_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, result_path)

    print(f"Evaluated metrics for {experiment_hash}")
def evaluate_all(results_dir: str, cache_dir: str, dataset_dir: str, fast: bool = False) -> None:
    for entry in os.listdir(results_dir):
        # Skip bookkeeping dirs (e.g. _superseded) — not experiment hashes.
        if entry.startswith('_'):
            continue
        exp_dir = os.path.join(results_dir, entry)
        if os.path.isdir(exp_dir):
            # Isolate failures: one un-evaluatable result must not abort the pass
            # for every other experiment (they share one expensive NFS load).
            try:
                run_evaluation(entry, results_dir, cache_dir, dataset_dir, fast=fast)
            except Exception as e:
                print(f"WARNING: evaluation failed for {entry}: {type(e).__name__}: {e}")
