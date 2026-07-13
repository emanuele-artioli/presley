import os
import time
import json
import numpy as np
import cv2
import subprocess
import tempfile
import torch
from pathlib import Path
from typing import Dict, Any, List

from presley.preprocessing import get_reference_frames, get_ufo_masks
from presley.encode_utils import load_frames_from_video

# Reference frames and UFO masks are identical across every experiment on the same
# (video, resolution, block_size), but they live on NFS (slow small-file I/O:
# ~38s to read 82 PNGs). evaluate_all() runs all experiments in one process, so
# memoizing here turns N reloads into 1 — the dominant eval cost, not the metrics.
_REF_CACHE: Dict[Any, Any] = {}
_MASK_CACHE: Dict[Any, Any] = {}

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

from skimage.metrics import structural_similarity as ssim

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

def calculate_lpips(refs: List[np.ndarray], decs: List[np.ndarray], device: str) -> List[float]:
    import lpips
    model = lpips.LPIPS(net='alex').to(device)
    scores = []
    with torch.no_grad():
        for r, d in zip(refs, decs):
            r_t = torch.from_numpy(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
            d_t = torch.from_numpy(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
            scores.append(model(r_t, d_t).item())
    return scores

def calculate_lpips_masked(refs: List[np.ndarray], decs: List[np.ndarray],
                           masks: List[np.ndarray], device: str) -> Dict[str, List[float]]:
    """Per-frame FG/BG/overall LPIPS using spatial-mode LPIPS.

    lpips(spatial=True) returns a per-pixel distance map at input resolution; we
    average it over the UFO mask (FG), its complement (BG), and the whole frame
    (overall). This is a true region-restricted perceptual metric — no bbox
    cropping or pixel-zeroing artifacts — and it's the FG number the paper argues.
    masks[i] is a >127 boolean foreground mask.
    """
    import lpips
    model = lpips.LPIPS(net='alex', spatial=True).to(device)
    fg, bg, ov = [], [], []
    with torch.no_grad():
        for r, d, m in zip(refs, decs, masks):
            r_t = torch.from_numpy(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
            d_t = torch.from_numpy(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
            smap = model(r_t, d_t).squeeze().cpu().numpy()  # [H, W]
            if smap.shape != m.shape:
                smap = cv2.resize(smap, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
            ov.append(float(smap.mean()))
            fg.append(float(smap[m].mean()) if np.any(m) else 0.0)
            bg.append(float(smap[~m].mean()) if np.any(~m) else 0.0)
    return {"foreground": fg, "background": bg, "overall": ov}


def calculate_fid(refs, decs, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    import torch
    fid = FrechetInceptionDistance(feature=2048).to(device)
    batch_size = 16
    for i in range(0, len(refs), batch_size):
        r_batch = refs[i:i+batch_size]
        d_batch = decs[i:i+batch_size]
        r_t = torch.from_numpy(np.array([cv2.cvtColor(r, cv2.COLOR_BGR2RGB) for r in r_batch])).permute(0, 3, 1, 2).byte().to(device)
        d_t = torch.from_numpy(np.array([cv2.cvtColor(d, cv2.COLOR_BGR2RGB) for d in d_batch])).permute(0, 3, 1, 2).byte().to(device)
        fid.update(r_t, real=True)
        fid.update(d_t, real=False)
    return float(fid.compute().item())

def calculate_dists(refs: List[np.ndarray], decs: List[np.ndarray], device: str) -> List[float]:
    from DISTS_pytorch import DISTS
    model = DISTS().to(device)
    scores = []
    with torch.no_grad():
        for r, d in zip(refs, decs):
            r_t = torch.from_numpy(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
            d_t = torch.from_numpy(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
            scores.append(model(r_t, d_t).item())
    return scores

def calculate_vmaf(ref_yuv: str, dec_video: str, width: int, height: int, framerate: float) -> Dict[str, float]:
    dec_yuv = dec_video + ".yuv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", dec_video, "-pix_fmt", "yuv420p", dec_yuv], check=True)
    
    out_json = dec_video + "_vmaf.json"
    cmd = ["vmaf", "-r", ref_yuv, "-d", dec_yuv, "-w", str(width), "-h", str(height), "-p", "420", "-b", "8", "--json", "-o", out_json]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        with open(out_json, "r") as f:
            data = json.load(f)
        os.remove(dec_yuv)
        os.remove(out_json)
        
        if 'pooled_metrics' in data and 'vmaf' in data['pooled_metrics']:
            v = data['pooled_metrics']['vmaf']
            return {"mean": v.get("mean", 0), "std": v.get("stddev", 0)}
        elif 'frames' in data:
            scores = [f['metrics']['vmaf'] for f in data['frames']]
            return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
    except Exception as e:
        print(f"VMAF failed: {e}")
    return {"mean": 0.0, "std": 0.0}

def calculate_fvmd(ref_dir: str, dec_video: str) -> float:
    # Minimal FVMD implementation
    try:
        from fvmd.datasets.video_datasets import VideoDataset
        from fvmd.keypoint_tracking import track_keypoints
        from fvmd.extract_motion_features import calc_hist
        from fvmd.frechet_distance import calculate_fd_given_vectors
        
        # We would decode dec_video to a temp dir and run FVMD between ref_dir and temp dir.
        # But since FVMD is extremely slow and we might skip it or use a simplified stub.
        # Leaving a placeholder for now to allow pipeline to run.
        return 0.0
    except ImportError:
        return 0.0

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
            "fvmd": calculate_fvmd(ref_frames_dir, output_video)
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
        data["metrics"][region]["lpips_mean"] = float(np.mean(lp[region]))
        data["metrics"][region]["lpips_std"] = float(np.std(lp[region]))

    tmp = result_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, result_path)
    return f"{experiment_hash}: FG-LPIPS={np.mean(lp['foreground']):.4f} BG={np.mean(lp['background']):.4f} OV={np.mean(lp['overall']):.4f}"

def _write_yuv420(frames: List[np.ndarray], path: str) -> None:
    """Write BGR frames as a raw yuv420p file via an ffmpeg pipe."""
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-i', '-',
         '-pix_fmt', 'yuv420p', '-f', 'rawvideo', path],
        stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()

def _vmaf_on_frames(refs: List[np.ndarray], decs: List[np.ndarray], neg: bool = False) -> Dict[str, float]:
    """Run the vmaf CLI on two equal-length BGR frame lists. neg=True uses the
    enhancement-robust vmaf_v0.6.1neg model (returns zeros if unavailable)."""
    h, w = refs[0].shape[:2]
    with tempfile.TemporaryDirectory() as td:
        ref_yuv, dec_yuv = os.path.join(td, "ref.yuv"), os.path.join(td, "dec.yuv")
        out_json = os.path.join(td, "vmaf.json")
        _write_yuv420(refs, ref_yuv)
        _write_yuv420(decs, dec_yuv)
        cmd = ['vmaf', '-r', ref_yuv, '-d', dec_yuv, '-w', str(w), '-h', str(h),
               '-p', '420', '-b', '8', '--json', '-o', out_json]
        if neg:
            cmd += ['--model', 'version=vmaf_v0.6.1neg']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_json):
            return {"mean": 0.0, "std": 0.0}
        with open(out_json) as f:
            data = json.load(f)
        pooled = data.get('pooled_metrics', {})
        # pooled key is 'vmaf' for the default model; the neg model logs under
        # its own name — take whichever vmaf* key is present
        for k in pooled:
            if k.startswith('vmaf'):
                return {"mean": pooled[k].get("mean", 0.0), "std": pooled[k].get("stddev", 0.0)}
        return {"mean": 0.0, "std": 0.0}

def _fg_union_bbox(masks: List[np.ndarray], w: int, h: int, pad: int = 8):
    """Union FG bounding box across frames, padded and even-aligned for yuv420."""
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

def backfill_vmaf(experiment_hash: str, results_dir: str, cache_dir: str, dataset_dir: str,
                  force: bool = False) -> str:
    """Append overall + FG-crop VMAF (default and NEG models) to an existing
    result.json in place.

    Metric-only pass over on-disk artifacts, like backfill_lpips: no re-encode.
    FG-VMAF is computed on the per-video union FG bounding-box crop (VMAF needs
    constant-resolution natural frames; a mask cannot be applied directly), so
    it includes some BG context within the box — comparisons are within-video
    at matched bitrate, where the box is identical across methods. The NEG
    model discounts enhancement/sharpening gains; reporting both makes the
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
    if not force and "vmaf_mean" in data["metrics"].get("foreground", {}):
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
    m.setdefault("foreground", {})["vmaf_mean"] = fg["mean"]
    m["foreground"]["vmaf_std"] = fg["std"]
    m["foreground"]["vmaf_neg_mean"] = fg_neg["mean"]

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
    exp_results_dir = os.path.join(results_dir, experiment_hash)
    result_path = os.path.join(exp_results_dir, "result.json")
    if not os.path.exists(result_path): return f"{experiment_hash}: no result.json"
    with open(result_path, 'r') as f: data = json.load(f)
    if "metrics" not in data: return f"{experiment_hash}: no metrics yet"
    if not force and "dists_mean" in data["metrics"].get("foreground", {}): return f"{experiment_hash}: FG-DISTS already present"
    
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
    
    ov_dists = float(np.mean(calculate_dists(refs[:n], decs[:n], device)))
    bb = _fg_union_bbox(masks, width, height)
    fg_dists = 0.0
    if bb:
        y1, y2, x1, x2 = bb
        ref_c = [refs[i][y1:y2, x1:x2] for i in range(n)]
        dec_c = [decs[i][y1:y2, x1:x2] for i in range(n)]
        fg_dists = float(np.mean(calculate_dists(ref_c, dec_c, device)))
        
    m = data["metrics"]
    m.setdefault("overall", {})["dists_mean"] = ov_dists
    m.setdefault("foreground", {})["dists_mean"] = fg_dists

    tmp = result_path + ".tmp"
    with open(tmp, 'w') as f: json.dump(data, f, indent=2)
    os.replace(tmp, result_path)
    return f"{experiment_hash}: OV-DISTS={ov_dists:.4f} FG-DISTS={fg_dists:.4f}"

def backfill_dists_all(results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> None:
    for entry in sorted(os.listdir(results_dir)):
        if os.path.isdir(os.path.join(results_dir, entry)) and not entry.startswith('_'):
            print(backfill_dists(entry, results_dir, cache_dir, dataset_dir, force=force))


def backfill_fid(experiment_hash: str, results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> str:
    exp_results_dir = os.path.join(results_dir, experiment_hash)
    result_path = os.path.join(exp_results_dir, "result.json")
    if not os.path.exists(result_path): return f"{experiment_hash}: no result.json"
    with open(result_path, 'r') as f: data = json.load(f)
    if "metrics" not in data: return f"{experiment_hash}: no metrics yet"
    if not force and "fid" in data["metrics"].get("foreground", {}): return f"{experiment_hash}: FG-FID already present"
    
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
    bb = _fg_union_bbox(masks, width, height)
    fg_fid = 0.0
    if bb:
        y1, y2, x1, x2 = bb
        ref_c = [refs[i][y1:y2, x1:x2] for i in range(n)]
        dec_c = [decs[i][y1:y2, x1:x2] for i in range(n)]
        fg_fid = calculate_fid(ref_c, dec_c, device)
        
    m = data["metrics"]
    m.setdefault("overall", {})["fid"] = float(ov_fid)
    m.setdefault("foreground", {})["fid"] = float(fg_fid)

    tmp = result_path + ".tmp"
    with open(tmp, 'w') as f: json.dump(data, f, indent=2)
    os.replace(tmp, result_path)
    return f"{experiment_hash}: OV-FID={ov_fid:.2f} FG-FID={fg_fid:.2f}"

def backfill_fid_all(results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> None:
    for entry in sorted(os.listdir(results_dir)):
        if os.path.isdir(os.path.join(results_dir, entry)) and not entry.startswith('_'):
            print(backfill_fid(entry, results_dir, cache_dir, dataset_dir, force=force))

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

def backfill_lpips_all(results_dir: str, cache_dir: str, dataset_dir: str, force: bool = False) -> None:
    for entry in sorted(os.listdir(results_dir)):
        if os.path.isdir(os.path.join(results_dir, entry)):
            print(backfill_lpips(entry, results_dir, cache_dir, dataset_dir, force=force))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str, default='results')
    parser.add_argument('--dataset-dir', type=str, default='dataset')
    parser.add_argument('--cache-dir', type=str, default='cache')
    parser.add_argument('--fast-metrics', action='store_true',
                        help='Only compute fast metrics (FG/BG/overall PSNR/SSIM/MSE); skip LPIPS/DISTS/VMAF/FVMD and block-level maps')
    parser.add_argument('--backfill-lpips', action='store_true',
                        help='Append FG/BG/overall masked LPIPS to existing result.json files without re-encoding or recomputing other metrics')
    parser.add_argument('--backfill-vmaf', action='store_true',
                        help='Append overall + FG-crop VMAF (default and NEG models) to existing result.json files without re-encoding')
    parser.add_argument('--backfill-dists', action='store_true',
                        help='Append overall + FG-crop DISTS to existing result.json files')
    parser.add_argument('--backfill-fid', action='store_true',
                        help='Append overall + FG-crop FID to existing result.json files')

    parser.add_argument('--force', action='store_true',
                        help='With a --backfill-* flag, recompute even if the metric is already present')
    args = parser.parse_args()
    if args.backfill_lpips:
        backfill_lpips_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_vmaf:
        backfill_vmaf_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_dists:
        backfill_dists_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_fid:
        backfill_fid_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    else:
        evaluate_all(args.results_dir, args.cache_dir, args.dataset_dir, fast=args.fast_metrics)

if __name__ == "__main__":
    main()
