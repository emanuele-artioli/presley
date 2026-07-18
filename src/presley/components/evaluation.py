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

# DISTS is a stateless scorer, so one instance can serve every experiment in a
# backfill pass (VGG16 + weights.pt construction is ~1-3s, paid 267x otherwise).
# NOTE: do NOT do this for FID -- FrechetInceptionDistance is a stateful
# accumulator, and a shared instance would silently pool every experiment
# together. See the comment in calculate_fid.
_DISTS_CACHE: Dict[str, Any] = {}

def _get_dists_model(device: str):
    if device not in _DISTS_CACHE:
        from DISTS_pytorch import DISTS
        _DISTS_CACHE[device] = DISTS().to(device).eval()
    return _DISTS_CACHE[device]

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

    Frames with an empty mask yield NaN, not 0.0: 0.0 is a *perfect* LPIPS score, and
    averaging fabricated zeros into foreground.lpips_mean would bias the paper's headline
    FG metric optimistically. Same convention as calculate_dists_masked and
    _fvmd_on_frames (see the fc203a9980dad7d3 fake-0.0 incident). Aggregate with nanmean.
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
            fg.append(float(smap[m].mean()) if np.any(m) else float('nan'))
            bg.append(float(smap[~m].mean()) if np.any(~m) else float('nan'))
    return {"foreground": fg, "background": bg, "overall": ov}


def calculate_fid(refs, decs, device):
    # NOTE: FrechetInceptionDistance is a stateful accumulator -- it must be constructed
    # per call (or .reset()), never cached like _get_dists_model, or every experiment in
    # a backfill pass would be pooled into one distribution.
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


def calculate_fid_bbox(refs: List[np.ndarray], decs: List[np.ndarray],
                       masks: List[np.ndarray], device: str) -> Dict[str, Any]:
    """Best-effort localised FID over per-frame tight FG bbox crops.

    THIS IS NOT A FOREGROUND METRIC, and its key (`fid_fg_bbox`) must always be written
    and cited by that full name. FID pools Inception down to a single 2048-d vector, so
    there is no spatial axis left to mask and no principled FG-FID exists -- unlike
    DISTS (see `calculate_dists_masked`) or LPIPS, whose spatial maps can be
    mask-weighted. The best available improvement is to replace the union bbox (100% of
    the frame on india, 58.6% on tennis vs 4.0% true FG) with a per-frame tight box,
    which is 1.3-3.8x tighter but still ~74% background on tennis. Cite `dists_fg` or
    FG-LPIPS for the foreground claim; cite this only as a corroborating signal.

    Crops vary in size per frame, so they cannot be batched. We therefore feed them one
    at a time at native size and let torchmetrics resize -- we deliberately do NOT resize
    to 299 ourselves. torch_fidelity's Inception extractor already resizes any input to
    299x299 internally (feature_extractor_inceptionv3.py:111, TF-compat bilinear), so an
    explicit resize would resample twice and would put this metric on a different
    preprocessing path than `overall.fid`, which is fed native frames. Do not "optimise"
    a resize back in.

    Because box size varies per frame while Inception's internal resize is anisotropic,
    this metric carries scale variance that whole-frame FID does not. The returned
    diagnostics quantify it. Note the box is derived from the *reference* mask and
    applied identically to reference and decoded frames, so the resize distortion is the
    same on both sides of every frame: it inflates within-distribution variance rather
    than biasing one side.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048).to(device)
    h, w = refs[0].shape[:2]
    n_used = n_skipped = 0
    areas, bg_fracs = [], []
    for r, d, m in zip(refs, decs, masks):
        bb = _fg_tight_bbox(m, w, h)
        if bb is None:
            # Skip on BOTH sides -- never asymmetrically, or the paired structure that
            # the small-sample validity argument rests on is broken.
            n_skipped += 1
            continue
        y1, y2, x1, x2 = bb
        r_c, d_c, m_c = r[y1:y2, x1:x2], d[y1:y2, x1:x2], m[y1:y2, x1:x2]
        box_px = (y2 - y1) * (x2 - x1)
        areas.append(box_px)
        bg_fracs.append(1.0 - float(m_c.sum()) / box_px)
        for arr, real in ((r_c, True), (d_c, False)):
            t = torch.from_numpy(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).byte().to(device)
            fid.update(t, real=real)
        n_used += 1

    if n_used < 2:
        # FID's covariance is undefined at N<2. Return NaN, never 0.0 (a perfect score).
        score = float('nan')
    else:
        score = float(fid.compute().item())
    side = np.sqrt(np.asarray(areas, dtype=np.float64)) if areas else np.array([0.0])
    return {
        "fid": score,
        "n_used": n_used,
        "n_skipped_empty": n_skipped,
        "area_frac_mean": float(np.mean(areas) / (w * h)) if areas else float('nan'),
        "area_frac_std": float(np.std(areas) / (w * h)) if areas else float('nan'),
        # dimensionless scale jitter across frames -- the artifact whole-frame FID lacks
        "scale_cv": float(side.std() / side.mean()) if areas and side.mean() > 0 else float('nan'),
        # how much background is still inside the box: this is the number that justifies
        # the key being named fid_fg_bbox and not fid_fg
        "bg_frac_mean": float(np.mean(bg_fracs)) if bg_fracs else float('nan'),
    }

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


def _dists_layer_weights(mask_t, feats):
    """Area-downsample a [1,1,H,W] weight map onto each DISTS layer's feature grid.

    Area-averaging makes each entry the fraction of that feature location covered by
    the mask, which is the natural weight for the pooled statistics below. feats[0]
    is the input image at full resolution, so its map is the mask itself.

    Sizes are read from the tensors, never assumed: for a 360x640 input the stages
    are 360/180/90/45/22 rows (L2pooling is 3-tap, stride 2: 45 -> 22).
    """
    import torch.nn.functional as F
    out = []
    for f in feats:
        size = tuple(f.shape[-2:])
        w = mask_t if tuple(mask_t.shape[-2:]) == size else F.interpolate(mask_t, size=size, mode='area')
        out.append(w)
    return out


def calculate_dists_masked(refs: List[np.ndarray], decs: List[np.ndarray],
                           masks: List[np.ndarray], device: str) -> Dict[str, List[float]]:
    """Per-frame FG/BG/overall DISTS with MASK-WEIGHTED spatial pooling.

    Stock `DISTS.forward` pools every layer with `.mean([2,3])` -- a global spatial
    mean/var/cov per channel. This replaces that pooling with a mask-weighted one
    (weighted mean sum(wx)/sum(w), var sum(w(x-mu)^2)/sum(w), cov sum(wxy)/sum(w) - mu_x*mu_y),
    keeping the pretrained alpha/beta weights untouched. It is the exact analogue of
    `calculate_lpips_masked` and returns the same shape.

    Uniform weights reproduce stock DISTS to <1e-5 (float32 reduction order only) --
    that equivalence is the correctness gate for this function.

    This supersedes the old `_fg_union_bbox`-cropped "FG-DISTS", which was not a
    foreground metric: the union bbox is 100% of the frame on india (its FG-DISTS was
    bit-identical to overall-DISTS, verified across 16/16 experiments) and 58.6% on
    tennis against a 4.0% true FG. See TECHNICAL_REPORT_PIPELINE_INFRA.md 2026-07-16.

    Caveat, and it must be stated wherever this is reported: this is mask-*weighted*,
    not mask-*isolated*. Background locations get exactly zero weight, but VGG units at
    stages 4-5 have receptive fields spanning tens of pixels, so an in-mask feature
    still integrates some surrounding background. It measures the foreground in
    context. The same is true of the FG-LPIPS we already report, and it is categorically
    different from the union-bbox defect, where background *locations* were pooled in
    directly.

    masks[i] is a >127 boolean foreground mask. Frames with an empty mask yield NaN,
    not 0.0 -- 0.0 is a perfect DISTS score, and this repo has already been burned once
    by a fabricated 0.0 (fc203a9980dad7d3, a swallowed exception). Aggregate with nanmean.
    """
    model = _get_dists_model(device)
    c1 = c2 = 1e-6
    # Normalisation reproduced verbatim from DISTS.forward.
    w_sum = model.alpha.sum() + model.beta.sum()
    alpha = torch.split(model.alpha / w_sum, model.chns, dim=1)
    beta = torch.split(model.beta / w_sum, model.chns, dim=1)

    def _pooled_score(feats0, feats1, weights) -> float:
        dist1 = 0
        dist2 = 0
        for k in range(len(model.chns)):
            x, y, w = feats0[k], feats1[k], weights[k]
            W = w.sum([2, 3], keepdim=True).clamp_min(1e-8)
            x_mean = (w * x).sum([2, 3], keepdim=True) / W
            y_mean = (w * y).sum([2, 3], keepdim=True) / W
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = (w * (x - x_mean) ** 2).sum([2, 3], keepdim=True) / W
            y_var = (w * (y - y_mean) ** 2).sum([2, 3], keepdim=True) / W
            xy_cov = (w * x * y).sum([2, 3], keepdim=True) / W - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)
        return float((1 - (dist1 + dist2).squeeze()).item())

    fg, bg, ov = [], [], []
    with torch.no_grad():
        for r, d, m in zip(refs, decs, masks):
            r_t = torch.from_numpy(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
            d_t = torch.from_numpy(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
            feats0, feats1 = model.forward_once(r_t), model.forward_once(d_t)

            m_t = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            if tuple(m_t.shape[-2:]) != tuple(r_t.shape[-2:]):
                m_t = torch.nn.functional.interpolate(m_t, size=tuple(r_t.shape[-2:]), mode='area')
            fg_w = _dists_layer_weights(m_t, feats0)
            # Area-averaging is linear, so 1 - area_avg(mask) == area_avg(1 - mask) exactly.
            bg_w = [1.0 - w for w in fg_w]
            ov_w = [torch.ones_like(w) for w in fg_w]

            ov.append(_pooled_score(feats0, feats1, ov_w))
            fg.append(_pooled_score(feats0, feats1, fg_w) if m.any() else float('nan'))
            bg.append(_pooled_score(feats0, feats1, bg_w) if (~m).any() else float('nan'))
    return {"foreground": fg, "background": bg, "overall": ov}


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

def _fvmd_hist_rows(refs: List[np.ndarray], decs: List[np.ndarray]):
    """Track a grid of keypoints across `refs` (ground truth) and `decs`
    (generated) and return `(gt_rows, gen_rows)`, each shape `[n_clips, D]` —
    the per-clip velocity+acceleration motion histograms that FVMD's Frechet
    distance is computed over. FVMD splits each video into fixed-length clips
    (the leading dim), so a single video already yields several feature rows.

    `gt_rows` depends only on the reference frames, so it can be reused as the
    reference distribution for any method's output on the same source video.
    Raises on failure; callers decide how to handle it.
    """
    from fvmd.datasets.video_datasets import VideoDatasetNP
    from fvmd.keypoint_tracking import track_keypoints
    from fvmd.extract_motion_features import calc_hist
    import tempfile

    refs_np = np.expand_dims(np.array(refs), axis=0)
    decs_np = np.expand_dims(np.array(decs), axis=0)
    gt_dataset = VideoDatasetNP(refs_np)
    gen_dataset = VideoDatasetNP(decs_np)
    with tempfile.TemporaryDirectory() as td:
        velo_gen, velo_gt, acc_gen, acc_gt = track_keypoints(
            log_dir=td, gen_dataset=gen_dataset, gt_dataset=gt_dataset, v_stride=1)
    B = velo_gen.shape[0]
    gt_rows = np.concatenate((calc_hist(velo_gt).reshape(B, -1),
                              calc_hist(acc_gt).reshape(B, -1)), axis=1)
    gen_rows = np.concatenate((calc_hist(velo_gen).reshape(B, -1),
                               calc_hist(acc_gen).reshape(B, -1)), axis=1)
    return gt_rows, gen_rows


def _fvmd_on_frames(refs: List[np.ndarray], decs: List[np.ndarray]) -> float:
    """Per-video FVMD: Frechet distance over one video's own clip features.
    NOTE: an intra-video adaptation — the leading dim is the clips of a single
    sequence, so scores are not comparable across videos of different motion
    content. Kept as an internal signal; the paper-grade metric is set-level
    (see `fvmd_set_level`)."""
    try:
        from fvmd.frechet_distance import calculate_fd_given_vectors
        gt_rows, gen_rows = _fvmd_hist_rows(refs, decs)
        return float(calculate_fd_given_vectors(gt_rows, gen_rows))
    except Exception as e:
        print(f"FVMD failed: {e}")
        return float('nan')

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


# ---------------------------------------------------------------------------
# Set-level FVMD (paper-grade): one distribution per method/set, pooled over a
# set of videos vs the matched clean references. Unlike per-video backfill_fvmd,
# this is the statistically-intended usage of a Frechet distance and produces
# ONE score per set — so it is written to a standalone table, never into the
# per-experiment result.json.
# ---------------------------------------------------------------------------

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


def _fvmd_rows_cached(experiment_hash, results_dir, cache_dir, dataset_dir):
    """Return `(gt_rows, gen_rows, key)` for one experiment, tracking keypoints
    only on a cache miss.

    Keypoint tracking is the whole cost of FVMD (~a minute of GPU per video);
    every statistic built on the rows afterwards is pure numpy. Persisting the
    768-dim rows to `cache/fvmd_feats/` makes the null control, the jackknife and
    any future analysis variant instant and re-runnable instead of costing a full
    GPU pass each.

    `key` is `(video, width, height, n_frames)`. The frame count is part of the
    key so a truncated decode can never read back another length's rows.

    A decode whose length differs from the reference is REJECTED rather than
    silently trimmed to `min(len(refs), len(decs))`. Two reasons, both
    correctness:
      * trimming makes `gt_rows` length-dependent, so the same source video
        would mint a second, near-duplicate reference block and get
        double-weighted in the pooled ground truth — breaking the
        "each video contributes exactly once" contract in `_collect_fvmd_rows`.
      * a set whose clips come from different-length decodes is not a
        like-for-like distribution comparison in the first place.
    A rejection is announced loudly; it must not pass as a quietly smaller set.

    Returns `(None, None, None)` if the experiment can't contribute.
    """
    result_path = os.path.join(results_dir, experiment_hash, "result.json")
    if not os.path.exists(result_path):
        print(f"  FVMD skip {experiment_hash}: no result.json"); return None, None, None
    data = json.load(open(result_path))
    cfg = data['config']
    video_name, width, height = cfg['video'], cfg['width'], cfg['height']
    output_video = data.get('output_video')
    if not output_video or not os.path.exists(output_video):
        print(f"  FVMD skip {experiment_hash}: output video missing"); return None, None, None

    feats_dir = _fvmd_feats_dir(cache_dir)
    gen_path = os.path.join(feats_dir, f"{experiment_hash}.npz")
    if os.path.exists(gen_path):
        z = np.load(gen_path)
        key = (video_name, width, height, int(z['n_frames']))
        ref_path = os.path.join(feats_dir, f"ref_{video_name}_{width}x{height}_{key[3]}.npz")
        if os.path.exists(ref_path):
            return np.load(ref_path)['rows'], z['rows'], key

    _, refs, _ = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    decs = load_frames_from_video(output_video)
    if not decs:
        print(f"  FVMD skip {experiment_hash}: no decodable frames"); return None, None, None
    if len(decs) != len(refs):
        print(f"  FVMD REJECT {experiment_hash} ({video_name}): decode has {len(decs)} frames "
              f"but reference has {len(refs)} — excluded from the set (see _fvmd_rows_cached)")
        return None, None, None
    n = len(refs)
    try:
        gt_rows, gen_rows = _fvmd_hist_rows(refs, decs)
    except Exception as e:
        print(f"  FVMD skip {experiment_hash}: {e}"); return None, None, None

    key = (video_name, width, height, n)
    _savez_atomic(gen_path, rows=gen_rows, n_frames=n)
    ref_path = os.path.join(feats_dir, f"ref_{video_name}_{width}x{height}_{n}.npz")
    if not os.path.exists(ref_path):
        _savez_atomic(ref_path, rows=gt_rows)
    return gt_rows, gen_rows, key


def fvmd_set_level(hashes, results_dir, cache_dir, dataset_dir, ref_cache=None):
    """Pool per-clip motion-histogram rows across a SET of experiments (which
    together span several source videos) and return one FVMD for the set's
    generated distribution vs the pooled clean-reference distribution.

    Returns `(fvmd_value, used_hashes)`. `ref_cache` (a dict keyed by
    `(video,width,height,n_frames)`) is populated with each source video's
    reference rows so the ground-truth distribution is tracked once and reused
    (and, crucially, each video contributes to the reference set exactly once)
    across every set that shares `ref_cache`.
    """
    gt_all, gen_all, used = _collect_fvmd_rows(hashes, results_dir, cache_dir,
                                               dataset_dir, ref_cache=ref_cache)
    if not gen_all:
        return float('nan'), []
    fd, _ = _fd_with_terms(np.concatenate(gt_all, axis=0), np.concatenate(gen_all, axis=0))
    return fd, [h for h, _ in used]


def _collect_fvmd_rows(hashes, results_dir, cache_dir, dataset_dir, ref_cache=None):
    """Gather rows for a set. Returns `(gt_blocks, gen_blocks, used)` where
    `gt_blocks` holds one reference row-block per *unique* source video in the
    set (deduped — a video that appears twice must not double-weight the ground
    truth) and `used` is a list of `(hash, key)`."""
    if ref_cache is None:
        ref_cache = {}
    gen_all, used = [], []
    for h in hashes:
        gt_rows, gen_rows, key = _fvmd_rows_cached(h, results_dir, cache_dir, dataset_dir)
        if gen_rows is None:
            continue
        ref_cache.setdefault(key, gt_rows)  # reference rows depend only on the clean video
        gen_all.append(gen_rows)
        used.append((h, key))
    if not gen_all:
        return [], [], []
    ref_keys = {k for _, k in used}
    return [ref_cache[k] for k in ref_keys], gen_all, used


def _fd_with_terms(feat1, feat2):
    """Frechet distance plus its decomposition into the mean and covariance
    terms: `FD = ||mu1-mu2||^2 + (tr(s1) + tr(s2) - 2*tr(covmean))`.

    At our sample size the rows are 768-dim but a 6-video set pools only ~402 of
    them, so the covariance estimate is rank-deficient and its term carries most
    of the estimator's bias. Splitting the terms out shows directly whether a
    score is driven by the mean difference (trustworthy at this N) or by the
    covariance term (not).

    Returns `(fd, {"mean_term":…, "cov_term":…})`. Raises on a numerically
    invalid sqrtm — callers must not turn that into a score.
    """
    from scipy import linalg
    from fvmd.frechet_distance import calculate_activation_statistics
    mu1, s1 = calculate_activation_statistics(feat1)
    mu2, s2 = calculate_activation_statistics(feat2)
    diff = mu1 - mu2
    mean_term = float(diff.dot(diff))
    eps = 1e-5  # parity with fvmd.frechet_distance.calculate_frechet_distance's default
    offset = np.eye(s1.shape[0]) * eps
    covmean, _ = linalg.sqrtm((s1 + offset).dot(s2 + offset), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        # Mirror the reference implementation EXACTLY: a materially complex
        # sqrtm means the result is numerically meaningless, and taking .real
        # unconditionally would launder that failure into a plausible finite
        # score. That matters most here -- a near-singular product is expected
        # when N < D, which is the regime this whole report exists to measure.
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f'Imaginary component {np.max(np.abs(covmean.imag))}')
        covmean = covmean.real
    cov_term = float(np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
    return mean_term + cov_term, {"mean_term": mean_term, "cov_term": cov_term}


def fvmd_setlevel_report(groups_path: str, results_dir: str, cache_dir: str,
                         dataset_dir: str, out_path: str) -> None:
    """Compute one set-level FVMD per group and write a standalone table.

    `groups_path` is a JSON file mapping set-name -> list of experiment hashes,
    e.g. {"baseline": ["ab..","cd..",...], "presley_ai": [...], ...}. All groups
    share a `ref_cache`, so the clean-reference distribution for each source
    video is computed once across the whole report.
    """
    with open(groups_path) as f:
        groups = json.load(f)
    ref_cache = {}
    rows = []
    for name, hashes in groups.items():
        print(f"[set-level FVMD] {name}: {len(hashes)} experiments")
        fd, used = fvmd_set_level(hashes, results_dir, cache_dir, dataset_dir, ref_cache=ref_cache)
        print(f"  -> FVMD={fd}  (used {len(used)}/{len(hashes)})")
        rows.append((name, fd, len(used), len(hashes)))
    with open(out_path, 'w') as f:
        f.write("set\tfvmd\tn_used\tn_total\n")
        for name, fd, nu, nt in rows:
            f.write(f"{name}\t{fd}\t{nu}\t{nt}\n")
    print(f"[set-level FVMD] wrote {out_path}")


def fvmd_validity_report(groups_path: str, results_dir: str, cache_dir: str,
                         dataset_dir: str, out_path: str, n_null: int = 10,
                         seed: int = 0) -> None:
    """Decide whether set-level FVMD can discriminate our methods AT ALL at this
    sample size, before any score is cited.

    FVMD rows here are 1024-dim while a 6-video set pools only ~372 of them, so
    the covariance is rank-deficient (N < D) and the Frechet distance is
    eps-regularised. That sounds fatal and is NOT, for one specific reason:

      **our comparison is PAIRED.** The decoded/restored video contains the same
      clips as its reference, so `gt_rows` and `gen_rows` describe the same
      underlying motion. The covariance-estimation error is then common to both
      sides and largely cancels inside `sqrtm(s1·s2)`. Measured: an *unpaired*
      split of identical data at N=186/side scores ~5.9e3 (real rows) and ~6.5e4
      (synthetic), while a *paired* comparison at N=372 with 1%/5%/20% added
      noise scores 4 / 103 / 1645 — small, and cleanly monotone in the
      perturbation. Rank-deficiency does not dominate a paired score.

    Consequences, learned the hard way (an earlier version of this report got
    both wrong and produced a "null floor" larger than the scores it was meant to
    bound -- an impossibility that revealed the error):

      * A split-half null of the reference is **UNPAIRED** and therefore does not
        bound our paired scores. It is reported below strictly as a diagnostic of
        the estimator's unpaired behaviour, and must never be read as a floor.
      * Subsampling `gen` to match a reference half likewise **breaks the
        pairing** and produces meaningless (huge) numbers. Not done.

    The real uncertainty is that we have only **6 source videos**, so the live
    question is video-sampling, not covariance rank. Hence:

      * chain scores -- one paired FVMD per group, with the mean/cov term split.
      * identity check -- FD(ref, ref) must be exactly 0 (instrument gate).
      * jackknife -- leave-one-video-out spread per group. **This is the gate:**
        if the between-group gaps are not large compared to the jackknife spread,
        the ordering is driven by which videos we happened to pick, and must not
        be cited.
    """
    with open(groups_path) as f:
        groups = json.load(f)
    ref_cache: Dict[Any, Any] = {}
    rows = []

    per_group = {}
    for name, hashes in groups.items():
        print(f"[FVMD validity] collecting {name}: {len(hashes)} experiments")
        gt_blocks, gen_blocks, used = _collect_fvmd_rows(hashes, results_dir, cache_dir,
                                                         dataset_dir, ref_cache=ref_cache)
        if not gen_blocks:
            print(f"  {name}: no usable experiments"); continue
        per_group[name] = (gt_blocks, gen_blocks, used)

    if not per_group:
        print("[FVMD validity] nothing to report"); return

    # --- chain scores + term decomposition -------------------------------
    for name, (gt_blocks, gen_blocks, used) in per_group.items():
        gt = np.concatenate(gt_blocks, axis=0)
        gen = np.concatenate(gen_blocks, axis=0)
        fd, terms = _fd_with_terms(gt, gen)
        print(f"[FVMD validity] {name}: FVMD={fd:.2f} "
              f"(mean_term={terms['mean_term']:.2f} cov_term={terms['cov_term']:.2f}) "
              f"N_gen={gen.shape[0]} N_gt={gt.shape[0]} D={gen.shape[1]}")
        rows.append((name, "score", fd, terms['mean_term'], terms['cov_term'],
                     gen.shape[0], gen.shape[1], len(used)))

    # Groups must span the same source videos, or their scores are not
    # comparable to each other (different ground-truth mixtures).
    vid_sets = {name: frozenset(k[0] for _, k in used) for name, (_, _, used) in per_group.items()}
    if len(set(vid_sets.values())) != 1:
        print("[FVMD validity] WARNING: groups do NOT span the same source videos — "
              "their scores are not directly comparable to each other.")
        for name, vs in vid_sets.items():
            print(f"    {name}: {sorted(vs)}")

    used_keys = sorted({k for _, (_, _, used) in per_group.items() for _, k in used})
    all_ref = np.concatenate([ref_cache[k] for k in used_keys], axis=0)

    # --- instrument gate: a paired comparison of identical rows must be 0 ---
    ident, _ = _fd_with_terms(all_ref, all_ref)
    print(f"[FVMD validity] identity check FD(ref,ref) = {ident:.6g} (must be ~0)")
    rows.append(("_identity_ref_vs_ref", "identity", ident, 0.0, 0.0,
                 all_ref.shape[0], all_ref.shape[1], 0))

    # --- diagnostic ONLY: unpaired split-half of the reference -------------
    # NOT a floor for the paired group scores above — see the docstring. Kept
    # because it quantifies how badly an UNPAIRED pooling behaves at this N/D,
    # which is the trap to avoid if anyone later compares across
    # non-corresponding clip sets.
    rng = np.random.default_rng(seed)
    half = all_ref.shape[0] // 2
    nulls = []
    for _ in range(n_null):
        idx = rng.permutation(all_ref.shape[0])
        fd, terms = _fd_with_terms(all_ref[idx[:half]], all_ref[idx[half:2 * half]])
        nulls.append(fd)
        rows.append(("_unpaired_ref_split", "diagnostic_unpaired", fd,
                     terms['mean_term'], terms['cov_term'], half, all_ref.shape[1], 0))
    print(f"[FVMD validity] [diagnostic, NOT a floor] unpaired ref split "
          f"(N={half}/side): mean={np.mean(nulls):.2f} std={np.std(nulls):.2f}")


    # --- jackknife: leave one source video out ---------------------------
    for name, (_, _, used) in per_group.items():
        vids = sorted({k[0] for _, k in used})
        jk = []
        for drop in vids:
            keep = [h for h, k in used if k[0] != drop]
            gt_b, gen_b, u2 = _collect_fvmd_rows(keep, results_dir, cache_dir,
                                                 dataset_dir, ref_cache=ref_cache)
            if not gen_b:
                continue
            fd, _ = _fd_with_terms(np.concatenate(gt_b, axis=0), np.concatenate(gen_b, axis=0))
            jk.append(fd)
            rows.append((name, f"jackknife_drop_{drop}", fd, float('nan'), float('nan'),
                         0, 0, len(u2)))
        if jk:
            print(f"[FVMD validity] {name}: jackknife mean={np.mean(jk):.2f} "
                  f"std={np.std(jk):.2f} range=[{np.min(jk):.2f}, {np.max(jk):.2f}]")

    with open(out_path, 'w') as f:
        f.write("group\tkind\tfvmd\tmean_term\tcov_term\tn_gen\td\tn_used\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"[FVMD validity] wrote {out_path}")


def _inception_feats(frames: List[np.ndarray], device: str, fid_model=None) -> np.ndarray:
    """[N, 2048] Inception pool3 features, extracted through the SAME module torchmetrics
    FID scores with (`fid.inception`) -- so this measures the actual features, not a
    lookalike. Frames are fed one at a time at native size; the extractor resizes to
    299x299 itself (see calculate_fid_bbox)."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    model = fid_model if fid_model is not None else FrechetInceptionDistance(feature=2048).to(device)
    out = []
    with torch.no_grad():
        for f in frames:
            t = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).byte().to(device)
            out.append(model.inception(t).squeeze(0).cpu().numpy())
    return np.stack(out)


def fid_validity_report(experiment_hash: str, results_dir: str, cache_dir: str,
                        dataset_dir: str, out_path: str, seed: int = 0) -> None:
    """Decide whether a PER-EXPERIMENT FID is meaningful at our sample size, before any
    score is cited. Run this BEFORE backfilling the corpus -- if it fails, the
    per-experiment design is wrong and the backfill is wasted.

    The regime: one experiment gives N ~= 60-90 frames against D = 2048 Inception
    features, so the covariance is badly rank-deficient (N/D ~= 0.03-0.04).

    The obvious defence is the one that rescued set-level FVMD: **our comparison is
    PAIRED** (decoded frames are the same content as their references), so the
    covariance-estimation error is common to both sides and largely cancels inside
    `sqrtm(s1*s2)`. That argument is NOT inherited here, and must not be assumed: FVMD
    held at N=372, D=1024 (ratio 0.36) -- an order of magnitude better than FID's ratio
    here. So it is tested:

      1. identity     -- FD(ref, ref) must be ~0. Instrument gate; if it fails, stop.
                         NOTE this row is deterministic, not data-dependent: with the
                         eps offset, sqrtm((s+eI)(s+eI)) = s+eI, so the score collapses
                         to 2tr(s) - 2tr(s+eI) = -2*eps*D exactly. D=2048 -> -0.0410.
                         (It retro-explains FVMD's reported -0.02: D=1024 -> -0.0205.)
                         It validates the sqrtm path, not the sample size.
      2. estimator noise -- **THE GATE.** FD(ref, ref+noise) at 1%/5%/20%: a known,
                         purely-additive paired perturbation against a clean baseline.
                         Must rise monotonically from ~0. This isolates the estimator's
                         ability to resolve a small paired difference at this N, which
                         is the actual question, and it mirrors the FVMD precedent
                         (4 / 103 / 1645 from an identity of ~0).
      3. decoded+noise -- SECONDARY, and deliberately not a gate. FD(ref, decoded+noise)
                         conflates two effects and must not be read as an estimator
                         check: the decoded video is already far from the reference, and
                         at low bitrate it is *blurred*, so added noise injects
                         high-frequency energy that can move its Inception texture
                         statistics back TOWARD the detailed reference. Measured on
                         tennis fg_bbox: 386 (+1%) -> 369 (+5%) -> 381 (+20%),
                         non-monotone, with the mean term falling 300 -> 254 -> 247.
                         That is the metric conflating noise with texture -- the same
                         effect that makes FID prefer hallucinated detail to blur -- and
                         it is a property of FID, not evidence about N. Reported because
                         it is a real caveat on citing FID for generative restoration.
      4. unpaired split -- DIAGNOSTIC ONLY, reported as `_diagnostic_unpaired_split`.
                         An unpaired split is NOT a floor for the paired scores above.
                         The FVMD version of this row was misread as a floor once
                         already, producing a "null" larger than the scores it was meant
                         to bound -- an impossibility that is how the error was caught.
                         Expect paired scores to sit legitimately below it.

    Pre-registered fallback, recorded before the numbers are seen: if the identity check
    is non-zero, or the noise ladder is not monotone, or the 1% score is not small
    relative to the between-method gaps we intend to cite, then per-experiment FID does
    not survive at N ~= 60-90 and must be DROPPED in favour of set-level pooling (frames
    pooled across the 6 videos per method, as set-level FVMD does, giving N ~= 450-540
    and restoring the FVMD regime). In that case `fid_fg_bbox` becomes set-level only,
    and the existing per-experiment `overall.fid` carries the same warning.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    result_path = os.path.join(results_dir, experiment_hash, "result.json")
    with open(result_path) as f:
        data = json.load(f)
    cfg = data['config']
    video_name, width, height = cfg['video'], cfg['width'], cfg['height']
    _, refs, _ = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    ufo_masks = _get_masks_cached(video_name, width, height, cfg.get('block_size', 8),
                                  ref_frames_dir, cache_dir)
    decs = load_frames_from_video(data['output_video'])
    n = min(len(refs), len(decs), len(ufo_masks))
    refs, decs = refs[:n], decs[:n]
    masks = [ufo_masks[i] > 127 for i in range(n)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FrechetInceptionDistance(feature=2048).to(device)
    rng = np.random.default_rng(seed)
    rows = []
    D = 2048

    print(f"[FID validity] {experiment_hash} ({video_name}, {cfg.get('component')}): "
          f"N={n} frames/side vs D={D} -> N/D={n/D:.3f}")
    fg_frac = float(np.mean([m.mean() for m in masks]))
    print(f"[FID validity] mean true FG fraction: {fg_frac:.3f}")

    def _crop(frames, use_bbox):
        """Whole-frame, or per-frame tight FG bbox (paired: same box on both sides)."""
        if not use_bbox:
            return frames
        out = []
        for f, m in zip(frames, masks):
            bb = _fg_tight_bbox(m, width, height)
            if bb is not None:
                y1, y2, x1, x2 = bb
                out.append(f[y1:y2, x1:x2])
        return out

    for kind, use_bbox in (("whole_frame", False), ("fg_bbox", True)):
        r_f = _crop(refs, use_bbox)
        d_f = _crop(decs, use_bbox)
        if len(r_f) < 2:
            print(f"[FID validity] {kind}: <2 usable frames, skipping")
            continue
        ref_feats = _inception_feats(r_f, device, model)
        n_side = len(ref_feats)

        # 1. identity gate
        ident, terms = _fd_with_terms(ref_feats, ref_feats)
        rows.append((kind, "identity_FD(ref,ref)", ident, terms['mean_term'], terms['cov_term'], n_side, D))
        print(f"[FID validity] {kind:11s} identity FD(ref,ref) = {ident:.4f} (must be ~0) "
              f"-> {'PASS' if abs(ident) < 1.0 else 'FAIL'}")

        # real decoded-vs-reference score, for scale
        dec_feats = _inception_feats(d_f, device, model)
        fd_real, terms = _fd_with_terms(ref_feats, dec_feats)
        rows.append((kind, "paired_decoded", fd_real, terms['mean_term'], terms['cov_term'], n_side, D))
        print(f"[FID validity] {kind:11s} paired decoded-vs-ref = {fd_real:.2f} "
              f"(mean {terms['mean_term']:.2f} / cov {terms['cov_term']:.2f})")

        def _noisy(frames, pct):
            return [np.clip(f.astype(np.float32) + rng.uniform(-pct*255, pct*255, f.shape),
                            0, 255).astype(np.uint8) for f in frames]

        # 2. THE GATE: estimator sensitivity, FD(ref, ref+noise), from a clean baseline.
        est_ladder = []
        for pct in (0.01, 0.05, 0.20):
            nf = _inception_feats(_noisy(r_f, pct), device, model)
            fd, t = _fd_with_terms(ref_feats, nf)
            est_ladder.append(fd)
            rows.append((kind, f"estimator_noise_{int(pct*100)}pct", fd, t['mean_term'], t['cov_term'], n_side, D))
            print(f"[FID validity] {kind:11s} GATE estimator ref-vs-ref+{int(pct*100):2d}% = {fd:.2f} "
                  f"(mean {t['mean_term']:.2f} / cov {t['cov_term']:.2f})")
        mono = all(est_ladder[i] < est_ladder[i+1] for i in range(len(est_ladder)-1))
        print(f"[FID validity] {kind:11s} GATE estimator ladder monotone from ~0? "
              f"{'PASS' if mono else 'FAIL'} ({' < '.join(f'{x:.1f}' for x in est_ladder)})")

        # 3. SECONDARY (not a gate): FD(ref, decoded+noise). Conflates estimator
        # behaviour with FID's noise-vs-texture confusion -- see the docstring.
        dec_ladder = []
        for pct in (0.01, 0.05, 0.20):
            nf = _inception_feats(_noisy(d_f, pct), device, model)
            fd, t = _fd_with_terms(ref_feats, nf)
            dec_ladder.append(fd)
            rows.append((kind, f"secondary_decoded_plus_noise_{int(pct*100)}pct", fd, t['mean_term'], t['cov_term'], n_side, D))
            print(f"[FID validity] {kind:11s} secondary decoded+{int(pct*100):2d}% = {fd:.2f} "
                  f"(mean {t['mean_term']:.2f} / cov {t['cov_term']:.2f})")
        dmono = all(dec_ladder[i] < dec_ladder[i+1] for i in range(len(dec_ladder)-1))
        print(f"[FID validity] {kind:11s} secondary decoded+noise monotone? {'yes' if dmono else 'NO'} "
              f"({' -> '.join(f'{x:.1f}' for x in dec_ladder)})"
              f"{'' if dmono else '  <- FID reading noise as texture, NOT an N failure'}")

        # 3. unpaired split -- DIAGNOSTIC ONLY, never a floor
        idx = rng.permutation(n_side)
        half = n_side // 2
        if half >= 2:
            fd, t = _fd_with_terms(ref_feats[idx[:half]], ref_feats[idx[half:2*half]])
            rows.append((kind, "_diagnostic_unpaired_split", fd, t['mean_term'], t['cov_term'], half, D))
            print(f"[FID validity] {kind:11s} _diagnostic_unpaired_split = {fd:.2f} at N={half}/side "
                  f"-- NOT a floor for the paired scores above")

    with open(out_path, 'w') as f:
        f.write("kind\trow\tfd\tmean_term\tcov_term\tn_per_side\td\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"[FID validity] wrote {out_path}")


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
                        help='Append overall DISTS + true mask-weighted FG/BG DISTS (dists_fg/dists_bg) to existing result.json files')
    parser.add_argument('--backfill-fid', action='store_true',
                        help='Append overall FID + per-frame tight-bbox fid_fg_bbox (best-effort locality, NOT a foreground metric) to existing result.json files')
    parser.add_argument('--backfill-fvmd', action='store_true',
                        help='Append overall + FG-crop per-video FVMD to existing result.json files (internal signal; not the paper metric)')
    parser.add_argument('--fvmd-setlevel', type=str, default=None, metavar='GROUPS_JSON',
                        help='Compute paper-grade set-level FVMD: JSON file mapping set-name -> list of experiment hashes')
    parser.add_argument('--fvmd-out', type=str, default='fvmd_setlevel.tsv',
                        help='Output table path for --fvmd-setlevel/--fvmd-validity')
    parser.add_argument('--fvmd-validity', type=str, default=None, metavar='GROUPS_JSON',
                        help='Set-level FVMD plus its validity gate (identity check, leave-one-video-out '
                             'jackknife = the real uncertainty at n=6, mean/cov term split) — run this '
                             'before citing any set-level score')

    parser.add_argument('--fid-validity', type=str, default=None, metavar='EXPERIMENT_HASH',
                        help='FID small-sample validity gate on one experiment (identity check, paired-noise '
                             'monotonicity, unpaired-split diagnostic) at N~60-90 vs D=2048 — run this before '
                             'citing any per-experiment FID')
    parser.add_argument('--fid-out', type=str, default='scratch/fid_validity.tsv',
                        help='Output table path for --fid-validity')
    parser.add_argument('--drop-unionbbox-keys', action='store_true',
                        help='One-shot: delete the superseded union-bbox foreground.dists_mean/foreground.fid '
                             'keys (not foreground metrics; see TECHNICAL_REPORT_PIPELINE_INFRA.md 2026-07-16). '
                             'No recomputation; idempotent')

    parser.add_argument('--force', action='store_true',
                        help='With a --backfill-* flag, recompute even if the metric is already present')
    parser.add_argument('--only', type=str, default=None, metavar='HASH',
                        help='With a --backfill-* / --drop-unionbbox-keys flag, act on a single experiment '
                             '(for verification runs)')
    parser.add_argument('--shard', type=str, default=None,
                        help='Shard the evaluation (e.g. 0/2)')
    args = parser.parse_args()
    if args.only:
        # Single-experiment verification path: run the one hash through the same
        # per-experiment function the *_all drivers call.
        fns = {
            'backfill_lpips': backfill_lpips, 'backfill_vmaf': backfill_vmaf,
            'backfill_dists': backfill_dists, 'backfill_fid': backfill_fid,
            'backfill_fvmd': backfill_fvmd,
        }
        selected = [f for f in fns if getattr(args, f)]
        if args.drop_unionbbox_keys:
            selected.append('drop_unionbbox_keys')
        if not selected:
            parser.error('--only requires a --backfill-* or --drop-unionbbox-keys flag')
        # Fail loudly rather than silently honouring only the first: the *_all path is an
        # elif chain, so passing two flags there already drops one, and --only is a
        # verification flag where a silently skipped metric is exactly the wrong outcome.
        if len(selected) > 1:
            parser.error(f'--only takes a single action, got: {", ".join(selected)}')
        if selected[0] == 'drop_unionbbox_keys':
            print(drop_unionbbox_keys(args.only, args.results_dir))
        else:
            print(fns[selected[0]](args.only, args.results_dir, args.cache_dir,
                                   args.dataset_dir, force=args.force))
        return
    if args.fid_validity:
        fid_validity_report(args.fid_validity, args.results_dir, args.cache_dir,
                            args.dataset_dir, args.fid_out)
    elif args.drop_unionbbox_keys:
        drop_unionbbox_keys_all(args.results_dir)
    elif args.backfill_lpips:
        backfill_lpips_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_vmaf:
        backfill_vmaf_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_dists:
        backfill_dists_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_fid:
        backfill_fid_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_fvmd:
        backfill_fvmd_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force, shard=args.shard)
    elif args.fvmd_validity:
        fvmd_validity_report(args.fvmd_validity, args.results_dir, args.cache_dir, args.dataset_dir, args.fvmd_out)
    elif args.fvmd_setlevel:
        fvmd_setlevel_report(args.fvmd_setlevel, args.results_dir, args.cache_dir, args.dataset_dir, args.fvmd_out)
    else:
        evaluate_all(args.results_dir, args.cache_dir, args.dataset_dir, fast=args.fast_metrics)

if __name__ == "__main__":
    main()
