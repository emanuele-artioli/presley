import tempfile, json, uuid, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import lpips
import os
import sys
import math
from presley.utils import *
from presley.degradation import filter_frame_downsample, filter_frame_gaussian
from presley.restoration import restore_downsample_opencv_lanczos, restore_blur_opencv_unsharp_mask
from presley.encoding import encode_video, _decode_video_to_frames, _encode_frames_to_video
from presley.io import _load_resized_masks
_FVMD_DEVICE_LOCKS = {}

from typing import NamedTuple, Any

class _EvaluationContext(NamedTuple):
    reference_frames: Any
    masked_reference_fg_frames: Any
    masked_reference_bg_frames: Any
    fg_masks: Any
    bg_masks: Any
    roi_slice: Any
    crop_width: Any
    crop_height: Any
    crop_filter: Any
    framerate: Any
    block_size: Any
    masked_videos_dir: Any
    fvmd_log_root: Any
    enable_fvmd: Any
    fvmd_device_locks: Any

def _initialise_evaluation_worker(context):
    global _EVALUATION_CONTEXT
    _EVALUATION_CONTEXT = context

from skimage.metrics import structural_similarity as ssim
from fvmd.datasets.video_datasets import VideoDataset
from fvmd.keypoint_tracking import track_keypoints
from fvmd.extract_motion_features import calc_hist
from fvmd.frechet_distance import calculate_fd_given_vectors
import glob
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union, Callable, Sequence, Iterator, TextIO

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from presley.config import *

def _get_lpips_model(device: str='cuda' if torch.cuda.is_available() else 'cpu') -> lpips.LPIPS:
    """Lazy-load and cache LPIPS models per device."""
    normalized_device = device
    if normalized_device.startswith('cuda') and (not torch.cuda.is_available()):
        normalized_device = 'cpu'
    cached = _LPIPS_MODEL_CACHE.get(normalized_device)
    if cached is None:
        cached = lpips.LPIPS(net='alex').to(normalized_device)
        _LPIPS_MODEL_CACHE[normalized_device] = cached
    return cached
def _compute_mask_union_bbox(masks: Sequence[np.ndarray], width: int, height: int, padding_ratio: float=0.05) -> Tuple[int, int, int, int]:
    """Compute a padded bounding box over the union of provided masks."""
    if not masks:
        return (0, 0, width, height)
    union_mask = np.zeros((height, width), dtype=bool)
    for mask in masks:
        if mask is not None:
            union_mask |= mask
    if not np.any(union_mask):
        return (0, 0, width, height)
    (ys, xs) = np.where(union_mask)
    (min_y, max_y) = (int(ys.min()), int(ys.max()))
    (min_x, max_x) = (int(xs.min()), int(xs.max()))
    bbox_height = max_y - min_y + 1
    bbox_width = max_x - min_x + 1
    pad_y = max(1, int(bbox_height * padding_ratio))
    pad_x = max(1, int(bbox_width * padding_ratio))
    y = max(0, min_y - pad_y)
    x = max(0, min_x - pad_x)
    h = min(height - y, bbox_height + 2 * pad_y)
    w = min(width - x, bbox_width + 2 * pad_x)
    return (x, y, w, h)
def _apply_binary_mask(frame: np.ndarray, mask: np.ndarray, invert: bool=False) -> np.ndarray:
    """Return a copy of frame with pixels outside mask zeroed."""
    if frame is None or mask is None:
        return frame
    mask_bool = mask if not invert else ~mask
    masked_frame = np.zeros_like(frame)
    masked_frame[mask_bool] = frame[mask_bool]
    return masked_frame
def _masked_psnr(ref: np.ndarray, dec: np.ndarray, mask: Optional[np.ndarray]=None) -> float:
    """Compute PSNR restricted to masked pixels."""
    if ref is None or dec is None:
        return 0.0
    ref_f = ref.astype(np.float32)
    dec_f = dec.astype(np.float32)
    if mask is not None:
        valid = mask.astype(bool)
        if not np.any(valid):
            return 100.0
        diff = ref_f[valid] - dec_f[valid]
    else:
        diff = ref_f - dec_f
    mse = float(np.mean(diff ** 2)) if diff.size else 0.0
    if mse < 1e-10:
        return 100.0
    max_pixel_value = 255.0
    psnr_val = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return float(min(psnr_val, 100.0))
def _masked_mse(ref: np.ndarray, dec: np.ndarray, mask: Optional[np.ndarray]=None) -> float:
    """Compute MSE (Mean Squared Error) restricted to masked pixels."""
    if ref is None or dec is None:
        return 0.0
    ref_f = ref.astype(np.float32)
    dec_f = dec.astype(np.float32)
    if mask is not None:
        valid = mask.astype(bool)
        if not np.any(valid):
            return 0.0
        diff = ref_f[valid] - dec_f[valid]
    else:
        diff = ref_f - dec_f
    mse = float(np.mean(diff ** 2)) if diff.size else 0.0
    return mse
def _masked_ssim(ref: np.ndarray, dec: np.ndarray, mask: Optional[np.ndarray]=None) -> float:
    """Compute SSIM on luminance channel within the mask."""
    if ref is None or dec is None:
        return 0.0
    ref_y = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    dec_y = cv2.cvtColor(dec, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    if mask is not None:
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return 1.0
        (ys, xs) = np.where(mask_bool)
        (y1, y2) = (ys.min(), ys.max() + 1)
        (x1, x2) = (xs.min(), xs.max() + 1)
        ref_y = ref_y[y1:y2, x1:x2].copy()
        dec_y = dec_y[y1:y2, x1:x2].copy()
        mask_crop = mask_bool[y1:y2, x1:x2]
        ref_y[~mask_crop] = 0
        dec_y[~mask_crop] = 0
    (h, w) = ref_y.shape[:2]
    smallest_dim = min(h, w)
    if smallest_dim < 3:
        return 1.0
    if smallest_dim < 7:
        win_size = smallest_dim if smallest_dim % 2 == 1 else max(3, smallest_dim - 1)
    else:
        win_size = 7
    return float(ssim(ref_y, dec_y, data_range=255, gaussian_weights=True, win_size=win_size))
def generate_opencv_benchmarks(reference_frames: Sequence[np.ndarray], strength_maps: Optional[Dict[str, np.ndarray]], block_size: int, framerate: float, width: int, height: int, temp_dir: str, video_bitrates: Dict[str, float]) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Generate OpenCV restoration benchmark videos for downstream evaluation."""
    if not strength_maps:
        return ({}, {})
    print('\n' + '=' * 80)
    print('GENERATING OPENCV RESTORATION BENCHMARKS')
    print('=' * 80)
    benchmarks_dir = os.path.join(temp_dir, 'opencv_benchmarks')
    os.makedirs(benchmarks_dir, exist_ok=True)
    opencv_benchmarks: Dict[str, str] = {}
    updated_bitrates = dict(video_bitrates)
    for (method_name, maps) in strength_maps.items():
        if maps is None:
            continue
        print(f'\nProcessing benchmarks for: {method_name}')
        target_bitrate = video_bitrates.get(method_name, 1000000)
        normalized_name = method_name.lower()
        if 'downsample' in normalized_name or 'realesrgan' in normalized_name:
            print('  - Generating Lanczos restoration benchmark...')
            benchmark_frames_lanczos: List[np.ndarray] = []
            for (frame_idx, frame) in enumerate(reference_frames):
                map_frame = maps[frame_idx]
                normalizer = np.max(map_frame)
                normalized_map = map_frame / normalizer if normalizer > 0 else map_frame
                (downsampled_frame, _) = filter_frame_downsample(frame, normalized_map, block_size)
                restored_frame = restore_downsample_opencv_lanczos(downsampled_frame, map_frame, block_size)
                benchmark_frames_lanczos.append(restored_frame)
            lanczos_frames_dir = os.path.join(benchmarks_dir, f'{method_name}_lanczos_frames')
            os.makedirs(lanczos_frames_dir, exist_ok=True)
            for (i, frame) in enumerate(benchmark_frames_lanczos):
                cv2.imwrite(os.path.join(lanczos_frames_dir, f'{i + 1:05d}.png'), frame)
            lanczos_video = os.path.join(benchmarks_dir, f'{method_name}_lanczos.mp4')
            encode_video(lanczos_frames_dir, lanczos_video, framerate, width, height, target_bitrate=target_bitrate)
            key = APPROACH_PRESLEY_LANCZOS if method_name == APPROACH_PRESLEY_REALESRGAN else f'{method_name} Lanczos'
            opencv_benchmarks[key] = lanczos_video
            updated_bitrates[key] = video_bitrates.get(method_name, 0.0)
        elif 'gaussian' in normalized_name or 'blur' in normalized_name or 'instantir' in normalized_name:
            print('  - Generating unsharp mask restoration benchmark...')
            benchmark_frames_unsharp: List[np.ndarray] = []
            for (frame_idx, frame) in enumerate(reference_frames):
                map_frame = maps[frame_idx]
                normalizer = np.max(map_frame)
                normalized_map = map_frame / normalizer if normalizer > 0 else map_frame
                (blurred_frame, _) = filter_frame_gaussian(frame, normalized_map, block_size)
                restored_frame = restore_blur_opencv_unsharp_mask(blurred_frame, map_frame, block_size)
                benchmark_frames_unsharp.append(restored_frame)
            unsharp_frames_dir = os.path.join(benchmarks_dir, f'{method_name}_unsharp_frames')
            os.makedirs(unsharp_frames_dir, exist_ok=True)
            for (i, frame) in enumerate(benchmark_frames_unsharp):
                cv2.imwrite(os.path.join(unsharp_frames_dir, f'{i + 1:05d}.png'), frame)
            unsharp_video = os.path.join(benchmarks_dir, f'{method_name}_unsharp.mp4')
            encode_video(unsharp_frames_dir, unsharp_video, framerate, width, height, target_bitrate=target_bitrate)
            key = APPROACH_PRESLEY_UNSHARP if method_name == APPROACH_PRESLEY_INSTANTIR else f'{method_name} Unsharp'
            opencv_benchmarks[key] = unsharp_video
            updated_bitrates[key] = video_bitrates.get(method_name, 0.0)
    print(f'\nGenerated {len(opencv_benchmarks)} OpenCV restoration benchmarks.')
    print('=' * 80 + '\n')
    return (opencv_benchmarks, updated_bitrates)
def calculate_lpips_per_frame(reference_frames: List[np.ndarray], decoded_frames: List[np.ndarray], device: str='cuda' if torch.cuda.is_available() else 'cpu') -> List[float]:
    """Calculate LPIPS over an aligned list of frame pairs."""
    if not reference_frames or not decoded_frames:
        return []
    lpips_model = _get_lpips_model(device)
    model_device = next(lpips_model.parameters()).device
    lpips_scores: List[float] = []
    with torch.no_grad():
        for (ref_frame, dec_frame) in zip(reference_frames, decoded_frames):
            if ref_frame is None or dec_frame is None:
                continue
            ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            dec_rgb = cv2.cvtColor(dec_frame, cv2.COLOR_BGR2RGB)
            ref_tensor = torch.from_numpy(np.ascontiguousarray(ref_rgb)).permute(2, 0, 1).unsqueeze(0).float()
            dec_tensor = torch.from_numpy(np.ascontiguousarray(dec_rgb)).permute(2, 0, 1).unsqueeze(0).float()
            ref_tensor = ref_tensor.to(model_device) / 127.5 - 1.0
            dec_tensor = dec_tensor.to(model_device) / 127.5 - 1.0
            lpips_score = lpips_model(ref_tensor, dec_tensor).item()
            lpips_scores.append(lpips_score)
    return lpips_scores
def calculate_vmaf(reference_video: str, distorted_video: str, width: int, height: int, framerate: float, model_path: str=None, frame_stride: int=1) -> Dict[str, float]:
    """
    Calculate VMAF (Video Multimethod Assessment Fusion) using the standalone vmaf command-line tool.
    Automatically converts videos to YUV format if needed.
    
    Args:
        reference_video: Path to the reference video file
        distorted_video: Path to the distorted/encoded video file
        width: Video width
        height: Video height
        framerate: Video framerate
        model_path: Optional path to VMAF model file
    
    Returns:
        Dictionary containing VMAF statistics (mean, min, max, etc.)
    """

    def _convert_to_yuv(video_path: str, output_yuv: str, width: int, height: int, stride: int=1) -> bool:
        """Convert a video to YUV420p format."""
        try:
            vf_filters: List[str] = []
            if stride > 1:
                vf_filters.append(f"select='not(mod(n,{stride}))'")
                vf_filters.append('setpts=N/(FRAME_RATE*TB)')
            vf_filters.append(f'scale={width}:{height}')
            filter_arg = ','.join(vf_filters)
            convert_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video_path, '-vf', filter_arg, '-pix_fmt', 'yuv420p', '-y', output_yuv]
            result = subprocess.run(convert_cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f'Error converting video to YUV: {e.stderr}')
            return False
    try:
        ref_yuv = reference_video
        dist_yuv = distorted_video
        temp_ref_yuv = None
        temp_dist_yuv = None
        if not reference_video.endswith('.yuv'):
            temp_ref_yuv = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
            temp_ref_yuv.close()
            ref_yuv = temp_ref_yuv.name
            print(f'  - Converting reference video to YUV format...')
            if not _convert_to_yuv(reference_video, ref_yuv, width, height, frame_stride):
                return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
        if not distorted_video.endswith('.yuv'):
            temp_dist_yuv = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
            temp_dist_yuv.close()
            dist_yuv = temp_dist_yuv.name
            print(f'  - Converting distorted video to YUV format...')
            if not _convert_to_yuv(distorted_video, dist_yuv, width, height, frame_stride):
                return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
            output_json = temp_file.name
        vmaf_cmd = ['/opt/local/bin/vmaf', '-r', ref_yuv, '-d', dist_yuv, '-w', str(width), '-h', str(height), '-p', '420', '-b', '8', '--json', '-o', output_json]
        if model_path:
            vmaf_cmd.extend(['--model', model_path])
        result = subprocess.run(vmaf_cmd, capture_output=True, text=True, check=True)
        with open(output_json, 'r') as f:
            vmaf_data = json.load(f)
        if 'frames' in vmaf_data:
            vmaf_scores = [frame['metrics']['vmaf'] for frame in vmaf_data['frames']]
        elif 'pooled_metrics' in vmaf_data:
            pooled = vmaf_data['pooled_metrics']['vmaf']
            return {'mean': pooled.get('mean', 0), 'min': pooled.get('min', 0), 'max': pooled.get('max', 0), 'std': pooled.get('stddev', 0), 'harmonic_mean': pooled.get('harmonic_mean', 0)}
        else:
            print(f'Warning: Unexpected VMAF output format for {distorted_video}')
            return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
        vmaf_array = np.array(vmaf_scores)
        harmonic_mean = len(vmaf_scores) / np.sum([1.0 / max(score, 0.001) for score in vmaf_scores])
        os.unlink(output_json)
        if temp_ref_yuv:
            os.unlink(temp_ref_yuv.name)
        if temp_dist_yuv:
            os.unlink(temp_dist_yuv.name)
        return {'mean': float(np.mean(vmaf_array)), 'min': float(np.min(vmaf_array)), 'max': float(np.max(vmaf_array)), 'std': float(np.std(vmaf_array)), 'harmonic_mean': float(harmonic_mean)}
    except subprocess.CalledProcessError as e:
        print(f'Error running VMAF command: {e.stderr}')
        return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
    except Exception as e:
        print(f'Error calculating VMAF: {str(e)}')
        return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
    finally:
        if 'output_json' in locals() and os.path.exists(output_json):
            try:
                os.unlink(output_json)
            except:
                pass
        if 'temp_ref_yuv' in locals() and temp_ref_yuv and os.path.exists(temp_ref_yuv.name):
            try:
                os.unlink(temp_ref_yuv.name)
            except:
                pass
        if 'temp_dist_yuv' in locals() and temp_dist_yuv and os.path.exists(temp_dist_yuv.name):
            try:
                os.unlink(temp_dist_yuv.name)
            except:
                pass
def calculate_fvmd(reference_frames: List[np.ndarray], decoded_frames: List[np.ndarray], log_root: Optional[str]=None, stride: int=1, max_frames: Optional[int]=None, early_stop_delta: float=0.002, early_stop_window: int=50, device: Optional[int]=None, verbose: bool=True) -> Tuple[float, float]:
    """Calculate FVMD statistics. Returns tuple of (fvmd_value, std_dev)."""
    printer = _safe_print if verbose else lambda *args, **kwargs: None
    if not reference_frames or not decoded_frames:
        raise ValueError('Both reference_frames and decoded_frames must contain at least one frame.')
    total_frames = min(len(reference_frames), len(decoded_frames))
    if total_frames < 2:
        raise ValueError('FVMD requires at least two frames in both reference and decoded sequences.')
    base_stride = max(1, stride)

    def _build_indices(stride_value: int) -> List[int]:
        idxs = list(range(0, total_frames, stride_value))
        if len(idxs) < 2 and total_frames >= 2:
            idxs = [0, total_frames - 1]
        if max_frames is not None and max_frames > 0:
            idxs = idxs[:max_frames]
        unique: List[int] = []
        seen: set[int] = set()
        for idx in idxs:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
        return unique

    class _FvmdNoTrajectories(RuntimeError):
        pass

    def _render_frames(frame_indices: Sequence[int], gt_clip: Path, gen_clip: Path) -> None:
        for (idx, frame_idx) in enumerate(frame_indices, start=1):
            ref_frame = reference_frames[frame_idx]
            dec_frame = decoded_frames[frame_idx]
            if ref_frame is None or dec_frame is None:
                raise ValueError('Frames must not be None when computing FVMD.')
            ref_frame_contig = np.ascontiguousarray(ref_frame)
            dec_frame_contig = np.ascontiguousarray(dec_frame)
            if ref_frame_contig.dtype != np.uint8:
                ref_frame_contig = np.clip(ref_frame_contig, 0, 255).astype(np.uint8)
            if dec_frame_contig.dtype != np.uint8:
                dec_frame_contig = np.clip(dec_frame_contig, 0, 255).astype(np.uint8)
            ref_path = gt_clip / f'{idx:05d}.png'
            dec_path = gen_clip / f'{idx:05d}.png'
            if not cv2.imwrite(str(ref_path), ref_frame_contig):
                raise RuntimeError(f'Failed to write reference frame for FVMD: {ref_path}')
            if not cv2.imwrite(str(dec_path), dec_frame_contig):
                raise RuntimeError(f'Failed to write decoded frame for FVMD: {dec_path}')
    min_required_frames = 10

    def _run_fvmd_once(frame_indices: Sequence[int]) -> float:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            gt_root = tmp_path / 'gt'
            gen_root = tmp_path / 'gen'
            clip_name = 'clip_0001'
            gt_clip = gt_root / clip_name
            gen_clip = gen_root / clip_name
            gt_clip.mkdir(parents=True, exist_ok=True)
            gen_clip.mkdir(parents=True, exist_ok=True)
            _render_frames(frame_indices, gt_clip, gen_clip)
            if log_root is None:
                logs_root_path = tmp_path / 'fvmd_logs'
            else:
                logs_root_path = Path(log_root)
            logs_root_path.mkdir(parents=True, exist_ok=True)
            run_log_dir = logs_root_path / f'run_{uuid.uuid4().hex}'
            run_log_dir.mkdir(parents=True, exist_ok=True)
            if len(frame_indices) < min_required_frames:
                raise _FvmdNoTrajectories(f'Only {len(frame_indices)} frame(s) sampled; FVMD requires at least {min_required_frames}.')
            clip_seq_len = max(min_required_frames, min(16, len(frame_indices)))
            gen_dataset = VideoDataset(str(gen_root), seq_len=clip_seq_len, stride=1)
            gt_dataset = VideoDataset(str(gt_root), seq_len=clip_seq_len, stride=1)
            if len(gen_dataset) == 0 or len(gt_dataset) == 0:
                raise _FvmdNoTrajectories('Insufficient frames after sampling for FVMD evaluation.')
            if not torch.cuda.is_available():
                raise RuntimeError('FVMD evaluation requires a CUDA-capable GPU, but none were detected.')
            available_gpus = torch.cuda.device_count()
            if device is not None:
                if device < 0 or device >= available_gpus:
                    raise ValueError(f'Requested FVMD device index {device} is out of range (found {available_gpus}).')
                device_ids = [int(device)]
            else:
                device_ids = [0]
            torch.cuda.set_device(device_ids[0])
            device_label = f'cuda:{device_ids[0]}'
            printer(f'    FVMD evaluating {len(frame_indices)} frame(s) on {device_label}')
            try:
                with _silence_console_output():
                    (velo_gen, velo_gt, acc_gen, acc_gt) = track_keypoints(log_dir=str(run_log_dir), gen_dataset=gen_dataset, gt_dataset=gt_dataset, v_stride=1, S=clip_seq_len, device_ids=device_ids)
            except RuntimeError as exc:
                raise _FvmdNoTrajectories(str(exc)) from exc
            if any((arr.size == 0 for arr in (velo_gen, velo_gt, acc_gen, acc_gt))):
                raise _FvmdNoTrajectories('FVMD keypoint tracking returned empty trajectories.')
            B = velo_gen.shape[0]
            if B == 0:
                raise _FvmdNoTrajectories('FVMD keypoint tracking produced zero batches.')
            try:
                gt_v_hist = calc_hist(velo_gt).reshape(B, -1)
                gen_v_hist = calc_hist(velo_gen).reshape(B, -1)
                gt_a_hist = calc_hist(acc_gt).reshape(B, -1)
                gen_a_hist = calc_hist(acc_gen).reshape(B, -1)
            except ValueError as exc:
                raise _FvmdNoTrajectories(f'Histogram computation failed: {exc}') from exc
            gt_hist = np.concatenate((gt_v_hist, gt_a_hist), axis=1)
            gen_hist = np.concatenate((gen_v_hist, gen_a_hist), axis=1)
            fvmd_value = calculate_fd_given_vectors(gt_hist, gen_hist)
            if not np.isfinite(fvmd_value):
                raise RuntimeError('FVMD produced a non-finite score.')
        return float(fvmd_value)

    def _compute_std(selected_indices: Sequence[int]) -> float:
        if len(selected_indices) < 2:
            return 0.0
        scores: List[float] = []
        warned = False
        window_span = min(len(selected_indices), max(min_required_frames, 8))
        for start in range(0, len(selected_indices) - window_span + 1):
            window_indices = selected_indices[start:start + window_span]
            try:
                scores.append(_run_fvmd_once(window_indices))
            except _FvmdNoTrajectories as exc:
                if not warned:
                    printer(f'  Warning: FVMD could not compute variability for one or more windows; first failure involving frames {window_indices}: {exc}')
                    warned = True
        if len(scores) <= 1:
            return 0.0
        return float(np.std(scores, ddof=1))
    window = max(1, early_stop_window)
    attempt_stride = base_stride
    while True:
        indices = _build_indices(attempt_stride)
        if len(indices) < min_required_frames:
            if attempt_stride == 1:
                raise ValueError(f'FVMD requires at least {min_required_frames} sampled frames; provide more frames or disable stride.')
            next_stride = max(1, attempt_stride // 2)
            if next_stride == attempt_stride:
                next_stride = attempt_stride - 1
            printer(f'  Warning: FVMD sampling with stride {attempt_stride} yielded only {len(indices)} frame(s); retrying with stride {next_stride}.')
            attempt_stride = next_stride
            continue
        processed = 0
        last_score: Optional[float] = None
        used_indices: Sequence[int] = []
        try:
            while processed < len(indices):
                next_count = min(len(indices), processed + window)
                current_indices = indices[:next_count]
                current_score = _run_fvmd_once(current_indices)
                used_indices = list(current_indices)
                if last_score is not None:
                    baseline = max(abs(last_score), 1e-06)
                    delta = abs(current_score - last_score) / baseline
                    if delta < early_stop_delta:
                        std_value = _compute_std(used_indices)
                        if attempt_stride != base_stride:
                            printer(f'  Info: FVMD used effective stride {attempt_stride} instead of requested {base_stride}.')
                        return (current_score, std_value)
                last_score = current_score
                processed = next_count
            assert last_score is not None
            std_value = _compute_std(used_indices if used_indices else indices)
            if attempt_stride != base_stride:
                printer(f'  Info: FVMD used effective stride {attempt_stride} instead of requested {base_stride}.')
            return (last_score, std_value)
        except _FvmdNoTrajectories as exc:
            if attempt_stride == 1:
                raise RuntimeError('FVMD failed to track keypoints even with stride=1. Consider reviewing the input frames or masks.') from exc
            next_stride = max(1, attempt_stride // 2)
            if next_stride == attempt_stride:
                next_stride = attempt_stride - 1
            printer(f'  Warning: FVMD tracking failed with stride {attempt_stride}; retrying with stride {next_stride}.')
            attempt_stride = next_stride
def analyze_encoding_performance(reference_frames: List[np.ndarray], encoded_videos: Dict[str, str], block_size: int, width: int, height: int, temp_dir: str, masks_dir: str, video_bitrates: Dict[str, float]={}, framerate: float=30.0, metric_stride: int=1, fvmd_stride: int=1, fvmd_max_frames: Optional[int]=None, fvmd_early_stop_delta: float=0.002, fvmd_early_stop_window: int=50, vmaf_stride: int=1, enable_fvmd: bool=True) -> Dict:
    """Analyze encoded videos with mask-aware metrics."""
    metric_stride = max(1, metric_stride)
    fvmd_stride = max(1, fvmd_stride)
    vmaf_stride = max(1, vmaf_stride)
    os.makedirs(temp_dir, exist_ok=True)
    masked_videos_dir = os.path.join(temp_dir, 'masked_videos')
    fvmd_log_root = os.path.join(temp_dir, 'fvmd_logs')
    os.makedirs(masked_videos_dir, exist_ok=True)
    os.makedirs(fvmd_log_root, exist_ok=True)
    if not os.path.isdir(masks_dir):
        print(f"Warning: Masks directory not found at '{masks_dir}'. Cannot perform FG/BG analysis.")
        return {}
    total_reference_frames = len(reference_frames)
    if total_reference_frames == 0:
        print('Warning: No reference frames provided. Skipping analysis.')
        return {}
    (fg_masks, bg_masks) = _load_resized_masks(masks_dir, width, height, total_reference_frames)
    fg_bbox = _compute_mask_union_bbox(fg_masks, width, height)
    (bbox_x, bbox_y, bbox_w, bbox_h) = fg_bbox
    y_start = bbox_y
    y_stop = min(height, bbox_y + max(1, bbox_h))
    x_start = bbox_x
    x_stop = min(width, bbox_x + max(1, bbox_w))
    roi_slice = (slice(y_start, y_stop), slice(x_start, x_stop))
    crop_width = max(1, x_stop - x_start)
    crop_height = max(1, y_stop - y_start)
    crop_filter = f'crop={crop_width}:{crop_height}:{x_start}:{y_start}'
    masked_reference_fg_frames = [_apply_binary_mask(reference_frames[idx], fg_masks[idx]) for idx in range(total_reference_frames)]
    masked_reference_bg_frames = [_apply_binary_mask(reference_frames[idx], bg_masks[idx]) for idx in range(total_reference_frames)]
    try:
        mp_ctx = multiprocessing.get_context('spawn')
    except ValueError:
        mp_ctx = multiprocessing.get_context()
    gpu_device_ids: List[Optional[int]] = []
    if enable_fvmd and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_device_ids = list(range(gpu_count))
    if not gpu_device_ids:
        gpu_device_ids = [None]
    unique_device_ids: List[Optional[int]] = []
    for device_id in gpu_device_ids:
        if device_id not in unique_device_ids:
            unique_device_ids.append(device_id)
    fvmd_device_locks: Dict[Optional[int], Any] = {}
    for device_id in unique_device_ids:
        fvmd_device_locks[device_id] = mp_ctx.Semaphore(1)
    global _EVALUATION_CONTEXT, _FVMD_DEVICE_LOCKS
    evaluation_context = _EvaluationContext(reference_frames=reference_frames, masked_reference_fg_frames=masked_reference_fg_frames, masked_reference_bg_frames=masked_reference_bg_frames, fg_masks=fg_masks, bg_masks=bg_masks, roi_slice=roi_slice, crop_width=crop_width, crop_height=crop_height, crop_filter=crop_filter, framerate=framerate, block_size=block_size, masked_videos_dir=masked_videos_dir, fvmd_log_root=fvmd_log_root, enable_fvmd=enable_fvmd, fvmd_device_locks=fvmd_device_locks)
    _EVALUATION_CONTEXT = evaluation_context
    analysis_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    videos_to_process: List[Tuple[str, str]] = []
    for (video_name, video_path) in encoded_videos.items():
        if not os.path.exists(video_path):
            print(f"\nProcessing '{video_name}'...")
            print('  - Video not found, skipping.')
            continue
        videos_to_process.append((video_name, video_path))
    if not videos_to_process:
        print('No encoded videos available for analysis.')

        _EVALUATION_CONTEXT = None
        _FVMD_DEVICE_LOCKS = {}
        return analysis_results
    cpu_count = multiprocessing.cpu_count() if hasattr(multiprocessing, 'cpu_count') else os.cpu_count()
    max_workers = min(len(videos_to_process), cpu_count or len(videos_to_process) or 1)
    effective_gpu_workers = len([device for device in gpu_device_ids if device is not None])
    if enable_fvmd and effective_gpu_workers > 0:
        max_workers = min(max_workers, effective_gpu_workers)
    max_workers = max(1, max_workers)
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx, initializer=_initialise_evaluation_worker, initargs=(evaluation_context,)) as executor:
        for (idx, (video_name, video_path)) in enumerate(videos_to_process):
            fvmd_device = gpu_device_ids[idx % len(gpu_device_ids)] if gpu_device_ids else None
            bitrate = video_bitrates.get(video_name, 0.0)
            future = executor.submit(_evaluate_single_video_metrics, video_name, video_path, metric_stride, fvmd_stride, fvmd_max_frames, fvmd_early_stop_delta, fvmd_early_stop_window, vmaf_stride, bitrate, fvmd_device)
            futures[future] = video_name
        for future in as_completed(futures):
            video_name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f"Error during analysis of '{video_name}'") from exc
            if result is not None:
                analysis_results[video_name] = result

    _EVALUATION_CONTEXT = None
    _FVMD_DEVICE_LOCKS = {}
    if not analysis_results:
        print('No results to display.')
    print(f'\nAnalysis complete. Masked videos saved to: {masked_videos_dir}')
    return analysis_results
def _evaluate_single_video_metrics(video_name: str, video_path: str, metric_stride: int, fvmd_stride: int, fvmd_max_frames: Optional[int], fvmd_early_stop_delta: float, fvmd_early_stop_window: int, vmaf_stride: int, bitrate_bps: float, fvmd_device: Optional[int]) -> Optional[Dict[str, Dict[str, float]]]:
    """Evaluate quality metrics for a single encoded approach in an isolated process."""
    if _EVALUATION_CONTEXT is None:
        raise RuntimeError('Evaluation context was not initialised before spawning workers.')
    ctx = _EVALUATION_CONTEXT
    print(f"\nProcessing '{video_name}'...")
    if not os.path.exists(video_path):
        print('  - Video not found, skipping.')
        return None
    decoded_frames = _decode_video_to_frames(video_path)
    reference_frames = ctx.reference_frames
    total_reference_frames = len(reference_frames)
    frame_count = min(total_reference_frames, len(decoded_frames))
    if frame_count == 0:
        print('  - No decoded frames available, skipping.')
        return None
    frame_indices = list(range(0, frame_count, metric_stride))
    if not frame_indices:
        frame_indices = [0]
    if frame_indices[-1] != frame_count - 1:
        frame_indices.append(frame_count - 1)
    frame_indices = sorted(set(frame_indices))
    slug = _slugify_name(video_name)
    block_size = ctx.block_size
    fg_masks = ctx.fg_masks
    bg_masks = ctx.bg_masks
    roi_slice = ctx.roi_slice
    masked_reference_fg_frames = ctx.masked_reference_fg_frames[:frame_count]
    masked_reference_bg_frames = ctx.masked_reference_bg_frames[:frame_count]
    masked_decoded_fg_frames = [_apply_binary_mask(decoded_frames[idx], fg_masks[idx]) for idx in range(frame_count)]
    masked_decoded_bg_frames = [_apply_binary_mask(decoded_frames[idx], bg_masks[idx]) for idx in range(frame_count)]
    fg_psnr_vals: List[float] = []
    fg_ssim_vals: List[float] = []
    fg_mse_vals: List[float] = []
    fg_ref_lpips_frames: List[np.ndarray] = []
    fg_dec_lpips_frames: List[np.ndarray] = []
    bg_psnr_vals: List[float] = []
    bg_ssim_vals: List[float] = []
    bg_mse_vals: List[float] = []
    bg_ref_lpips_frames: List[np.ndarray] = []
    bg_dec_lpips_frames: List[np.ndarray] = []
    for idx in frame_indices:
        ref_frame = reference_frames[idx]
        dec_frame = decoded_frames[idx]
        fg_mask = fg_masks[idx]
        bg_mask = bg_masks[idx]
        ref_roi = ref_frame[roi_slice]
        dec_roi = dec_frame[roi_slice]
        fg_mask_roi = fg_mask[roi_slice]
        fg_psnr_vals.append(_masked_psnr(ref_roi, dec_roi, fg_mask_roi))
        fg_ssim_vals.append(_masked_ssim(ref_roi, dec_roi, fg_mask_roi))
        fg_mse_vals.append(_masked_mse(ref_roi, dec_roi, fg_mask_roi))
        fg_ref_lpips_frames.append(masked_reference_fg_frames[idx][roi_slice])
        fg_dec_lpips_frames.append(masked_decoded_fg_frames[idx][roi_slice])
        bg_psnr_vals.append(_masked_psnr(ref_frame, dec_frame, bg_mask))
        bg_ssim_vals.append(_masked_ssim(ref_frame, dec_frame, bg_mask))
        bg_mse_vals.append(_masked_mse(ref_frame, dec_frame, bg_mask))
        bg_ref_lpips_frames.append(masked_reference_bg_frames[idx])
        bg_dec_lpips_frames.append(masked_decoded_bg_frames[idx])
    result: Dict[str, Dict[str, float]] = {'foreground': {'psnr_mean': float(np.mean(fg_psnr_vals)) if fg_psnr_vals else 0.0, 'psnr_std': float(np.std(fg_psnr_vals)) if fg_psnr_vals else 0.0, 'ssim_mean': float(np.mean(fg_ssim_vals)) if fg_ssim_vals else 0.0, 'ssim_std': float(np.std(fg_ssim_vals)) if fg_ssim_vals else 0.0, 'mse_mean': float(np.mean(fg_mse_vals)) if fg_mse_vals else 0.0, 'mse_std': float(np.std(fg_mse_vals)) if fg_mse_vals else 0.0}, 'background': {'psnr_mean': float(np.mean(bg_psnr_vals)) if bg_psnr_vals else 0.0, 'psnr_std': float(np.std(bg_psnr_vals)) if bg_psnr_vals else 0.0, 'ssim_mean': float(np.mean(bg_ssim_vals)) if bg_ssim_vals else 0.0, 'ssim_std': float(np.std(bg_ssim_vals)) if bg_ssim_vals else 0.0, 'mse_mean': float(np.mean(bg_mse_vals)) if bg_mse_vals else 0.0, 'mse_std': float(np.std(bg_mse_vals)) if bg_mse_vals else 0.0}, 'bitrate_mbps': bitrate_bps / 1000000}
    result['foreground']['fvmd'] = float('nan')
    result['foreground']['fvmd_std'] = float('nan')
    result['background']['fvmd'] = float('nan')
    result['background']['fvmd_std'] = float('nan')
    fg_lpips_scores = calculate_lpips_per_frame(fg_ref_lpips_frames, fg_dec_lpips_frames)
    bg_lpips_scores = calculate_lpips_per_frame(bg_ref_lpips_frames, bg_dec_lpips_frames)
    result['foreground']['lpips_mean'] = float(np.mean(fg_lpips_scores)) if fg_lpips_scores else 0.0
    result['foreground']['lpips_std'] = float(np.std(fg_lpips_scores)) if fg_lpips_scores else 0.0
    result['background']['lpips_mean'] = float(np.mean(bg_lpips_scores)) if bg_lpips_scores else 0.0
    result['background']['lpips_std'] = float(np.std(bg_lpips_scores)) if bg_lpips_scores else 0.0
    ref_fg_video_path = os.path.join(ctx.masked_videos_dir, f'{slug}_reference_fg_{frame_count:05d}.mp4')
    if not os.path.exists(ref_fg_video_path):
        _encode_frames_to_video(masked_reference_fg_frames, ref_fg_video_path, ctx.framerate, filter_chain=ctx.crop_filter, extra_codec_args=['-g', '1'])
    ref_bg_video_path = os.path.join(ctx.masked_videos_dir, f'{slug}_reference_bg_{frame_count:05d}.mp4')
    if not os.path.exists(ref_bg_video_path):
        _encode_frames_to_video(masked_reference_bg_frames, ref_bg_video_path, ctx.framerate, extra_codec_args=['-g', '1'])
    enc_fg_video_path = os.path.join(ctx.masked_videos_dir, f'{slug}_fg_{frame_count:05d}.mp4')
    _encode_frames_to_video(masked_decoded_fg_frames, enc_fg_video_path, ctx.framerate, filter_chain=ctx.crop_filter, extra_codec_args=['-g', '1'])
    enc_bg_video_path = os.path.join(ctx.masked_videos_dir, f'{slug}_bg_{frame_count:05d}.mp4')
    _encode_frames_to_video(masked_decoded_bg_frames, enc_bg_video_path, ctx.framerate, extra_codec_args=['-g', '1'])
    (frame_height, frame_width) = reference_frames[0].shape[:2]
    vmaf_fg = calculate_vmaf(ref_fg_video_path, enc_fg_video_path, ctx.crop_width, ctx.crop_height, ctx.framerate, frame_stride=vmaf_stride)
    vmaf_bg = calculate_vmaf(ref_bg_video_path, enc_bg_video_path, frame_width, frame_height, ctx.framerate, frame_stride=vmaf_stride)
    result['foreground']['vmaf_mean'] = float(vmaf_fg.get('mean', 0))
    result['foreground']['vmaf_std'] = float(vmaf_fg.get('std', 0))
    result['background']['vmaf_mean'] = float(vmaf_bg.get('mean', 0))
    result['background']['vmaf_std'] = float(vmaf_bg.get('std', 0))
    if ctx.enable_fvmd:
        min_fvmd_samples = 10
        total_available_frames = frame_count
        if fvmd_max_frames is not None and fvmd_max_frames > 0:
            total_available_frames = min(total_available_frames, fvmd_max_frames)
        effective_stride = max(1, fvmd_stride)
        if total_available_frames >= min_fvmd_samples:
            max_stride_for_min_samples = max(1, total_available_frames // min_fvmd_samples)
            effective_stride = min(effective_stride, max_stride_for_min_samples)
        else:
            effective_stride = 1
        if effective_stride != fvmd_stride:
            _safe_print(f"    Adjusted FVMD stride from {fvmd_stride} to {effective_stride} for '{video_name}' to sample enough frames.")
        fvmd_indices = list(range(0, frame_count, effective_stride))
        if not fvmd_indices:
            fvmd_indices = [0]
        if fvmd_max_frames is not None and fvmd_max_frames > 0:
            fvmd_indices = fvmd_indices[:fvmd_max_frames]
        if len(fvmd_indices) < min_fvmd_samples:
            _safe_print(f"    Skipping FVMD for '{video_name}': only {len(fvmd_indices)} sampled frame(s); need at least {min_fvmd_samples}.")
        else:
            ref_fg_fvmd_frames = [masked_reference_fg_frames[i] for i in fvmd_indices]
            dec_fg_fvmd_frames = [masked_decoded_fg_frames[i] for i in fvmd_indices]
            ref_bg_fvmd_frames = [masked_reference_bg_frames[i] for i in fvmd_indices]
            dec_bg_fvmd_frames = [masked_decoded_bg_frames[i] for i in fvmd_indices]
            fvmd_log_dir = os.path.join(ctx.fvmd_log_root, slug)
            os.makedirs(fvmd_log_dir, exist_ok=True)
            fvmd_lock = None
            lock_acquired = False
            if _FVMD_DEVICE_LOCKS:
                if fvmd_device in _FVMD_DEVICE_LOCKS:
                    fvmd_lock = _FVMD_DEVICE_LOCKS[fvmd_device]
                elif None in _FVMD_DEVICE_LOCKS:
                    fvmd_lock = _FVMD_DEVICE_LOCKS[None]
            try:
                if fvmd_lock is not None:
                    if fvmd_lock.acquire(block=False):
                        lock_acquired = True
                    else:
                        device_label = f'cuda:{fvmd_device}' if fvmd_device is not None else 'cpu'
                        _safe_print(f'    Waiting for FVMD device {device_label} to become available...')
                        fvmd_lock.acquire()
                        lock_acquired = True
                (fg_fvmd_mean, fg_fvmd_std) = calculate_fvmd(ref_fg_fvmd_frames, dec_fg_fvmd_frames, log_root=fvmd_log_dir, stride=1, max_frames=None, early_stop_delta=fvmd_early_stop_delta, early_stop_window=fvmd_early_stop_window, device=fvmd_device, verbose=False)
                (bg_fvmd_mean, bg_fvmd_std) = calculate_fvmd(ref_bg_fvmd_frames, dec_bg_fvmd_frames, log_root=fvmd_log_dir, stride=1, max_frames=None, early_stop_delta=fvmd_early_stop_delta, early_stop_window=fvmd_early_stop_window, device=fvmd_device, verbose=False)
            finally:
                if lock_acquired and fvmd_lock is not None:
                    fvmd_lock.release()
            result['foreground']['fvmd'] = fg_fvmd_mean
            result['foreground']['fvmd_std'] = fg_fvmd_std
            result['background']['fvmd'] = bg_fvmd_mean
            result['background']['fvmd_std'] = bg_fvmd_std
    print(f"  ✓ Completed evaluation for '{video_name}'.")
    return result
