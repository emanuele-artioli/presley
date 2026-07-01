import lpips
import json
import argparse
import contextlib
import os
import sys
import math
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
from presley.io import *
from presley.metrics import *
from presley.encode_utils import *
from presley.degradation import *
from presley.restoration import *
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
import builtins

os.environ.setdefault('LOGURU_LEVEL', 'WARNING')
try:
    from diffusers.utils import logging as _diffusers_logging
    _diffusers_logging.disable_progress_bar()
    _diffusers_logging.set_verbosity_error()
except ImportError:
    _diffusers_logging = None
@dataclass
class ChunkSpec:
    """Specification for a processing chunk."""
    start: int
    end: int
    device: torch.device
    chunk_id: int = 0
class _NullStream:
    """Lightweight write-only stream that safely discards all data."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass

    def writelines(self, lines: Sequence[str]) -> None:
        for line in lines:
            self.write(line)

    def close(self) -> None:
        pass

    @property
    def closed(self) -> bool:
        return False

    def isatty(self) -> bool:
        return False
@contextlib.contextmanager
def _silence_console_output() -> Iterator[None]:
    """Redirect stdout/stderr to a resilient null stream for noisy calls."""
    null_stream = _NullStream()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = null_stream
        sys.stderr = null_stream
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
def _safe_print(*args: Any, **kwargs: Any) -> None:
    """Print helper resilient to closed stdout/stderr streams."""
    target_stream = kwargs.get('file', sys.stdout)
    try:
        builtins.print(*args, **kwargs)
    except (ValueError, OSError):
        fallback = getattr(sys, '__stdout__', None)
        if fallback is None or fallback is target_stream:
            return
        kwargs['file'] = fallback
        try:
            builtins.print(*args, **kwargs)
        except (ValueError, OSError):
            pass
_LPIPS_MODEL_CACHE: Dict[str, lpips.LPIPS] = {}
def _configure_fvmd_logging() -> None:
    """Restrict FVMD's internal logger to warnings/errors to reduce noise."""
    try:
        from loguru import logger as _loguru_logger
    except ImportError:
        return
    try:
        _loguru_logger.disable('fvmd')
    except Exception:
        try:
            _loguru_logger.remove()
            _loguru_logger.add(sys.stderr, level='WARNING')
        except Exception:
            pass
_configure_fvmd_logging()
def _slugify_name(name: str) -> str:
    """Generate filesystem-friendly identifier from a video name."""
    import re
    slug = re.sub('[^\\w\\-]', '_', name.strip())
    return slug.strip('_') or 'video'
_REALESRGAN_UPSAMPLER_CACHE: Dict[str, 'RealESRGANer'] = {}
def _print_summary_report(results: Dict) -> None:
    """Prints a unified summary report with all metrics in one table."""
    print(f"\n{'=' * 180}")
    print(f"{'COMPREHENSIVE ANALYSIS SUMMARY':^180}")
    print(f"{'=' * 180}")
    if not results:
        print('No results to display.')
        return

    def _fmt(value: Optional[float], precision: int=2) -> str:
        return 'N/A' if value is None or not math.isfinite(value) else f'{value:.{precision}f}'

    def _format_pair(fg: Optional[float], bg: Optional[float], prec: int=2) -> str:
        return f'{_fmt(fg, prec)} / {_fmt(bg, prec)}'

    def _format_change(value: Optional[float]) -> str:
        return 'N/A' if value is None or not math.isfinite(value) else f'{value:+.2f}%'
    print(f"\n{'QUALITY METRICS (Foreground / Background)':^200}")
    print(f"{'Method':<20} {'PSNR (dB)':<25} {'SSIM':<25} {'MSE':<25} {'LPIPS':<25} {'FVMD':<25} {'VMAF':<25} {'Bitrate (Mbps)':<15}")
    print(f"{'-' * 200}")
    for (video_name, data) in results.items():
        fg_data = data['foreground']
        bg_data = data['background']
        psnr_str = _format_pair(fg_data.get('psnr_mean'), bg_data.get('psnr_mean'), precision_fg=2)
        ssim_str = _format_pair(fg_data.get('ssim_mean'), bg_data.get('ssim_mean'), precision_fg=4, precision_bg=4)
        mse_str = _format_pair(fg_data.get('mse_mean'), bg_data.get('mse_mean'), precision_fg=2, precision_bg=2)
        lpips_str = _format_pair(fg_data.get('lpips_mean'), bg_data.get('lpips_mean'), precision_fg=4, precision_bg=4)
        fvmd_str = _format_pair(fg_data.get('fvmd'), bg_data.get('fvmd'), precision_fg=2)
        vmaf_str = _format_pair(fg_data.get('vmaf_mean'), bg_data.get('vmaf_mean'), precision_fg=2)
        bitrate_str = _fmt(data.get('bitrate_mbps'), precision=2)
        print(f'{video_name:<20} {psnr_str:<25} {ssim_str:<25} {mse_str:<25} {lpips_str:<25} {fvmd_str:<25} {vmaf_str:<25} {bitrate_str:<15}')
    print(f"{'-' * 200}")
    if len(results) > 1:
        baseline_name = list(results.keys())[0]
        print(f"\n{'TRADE-OFF ANALYSIS (vs. ' + baseline_name + ')':^200}")
        print(f"{'Method':<20} {'PSNR FG %':<15} {'PSNR BG %':<15} {'SSIM FG %':<15} {'SSIM BG %':<15} {'MSE FG %':<15} {'MSE BG %':<15} {'LPIPS FG %':<15} {'LPIPS BG %':<15} {'FVMD FG %':<15} {'FVMD BG %':<15} {'VMAF FG %':<15} {'VMAF BG %':<15}")
        print(f"{'-' * 200}")
        for video_name in list(results.keys())[1:]:
            psnr_fg_change = math.nan
            psnr_bg_change = math.nan
            ssim_fg_change = math.nan
            ssim_bg_change = math.nan
            mse_fg_change = math.nan
            mse_bg_change = math.nan
            lpips_fg_change = math.nan
            lpips_bg_change = math.nan
            fvmd_fg_change = math.nan
            fvmd_bg_change = math.nan
            vmaf_fg_change = math.nan
            vmaf_bg_change = math.nan
            for metric in ['psnr', 'ssim', 'mse', 'lpips', 'vmaf']:
                for region in ['foreground', 'background']:
                    baseline_val = results[baseline_name][region].get(f'{metric}_mean')
                    current_val = results[video_name][region].get(f'{metric}_mean')
                    change = math.nan
                    if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)) and math.isfinite(baseline_val) and math.isfinite(current_val) and (baseline_val != 0):
                        if metric == 'lpips':
                            if current_val > 0:
                                change = (baseline_val / current_val - 1) * 100
                        else:
                            change = (current_val / baseline_val - 1) * 100
                    if metric == 'psnr' and region == 'foreground':
                        psnr_fg_change = change
                    elif metric == 'psnr' and region == 'background':
                        psnr_bg_change = change
                    elif metric == 'ssim' and region == 'foreground':
                        ssim_fg_change = change
                    elif metric == 'ssim' and region == 'background':
                        ssim_bg_change = change
                    elif metric == 'mse' and region == 'foreground':
                        mse_fg_change = change
                    elif metric == 'mse' and region == 'background':
                        mse_bg_change = change
                    elif metric == 'lpips' and region == 'foreground':
                        lpips_fg_change = change
                    elif metric == 'lpips' and region == 'background':
                        lpips_bg_change = change
                    elif metric == 'vmaf' and region == 'foreground':
                        vmaf_fg_change = change
                    elif metric == 'vmaf' and region == 'background':
                        vmaf_bg_change = change
            for region in ['foreground', 'background']:
                baseline_fvmd = results[baseline_name][region].get('fvmd')
                current_fvmd = results[video_name][region].get('fvmd')
                change = math.nan
                if isinstance(baseline_fvmd, (int, float)) and isinstance(current_fvmd, (int, float)) and math.isfinite(baseline_fvmd) and math.isfinite(current_fvmd) and (baseline_fvmd > 0) and (current_fvmd > 0):
                    change = (baseline_fvmd / current_fvmd - 1) * 100
                if region == 'foreground':
                    fvmd_fg_change = change
                else:
                    fvmd_bg_change = change
            psnr_fg_change_str = _format_change(psnr_fg_change)
            psnr_bg_change_str = _format_change(psnr_bg_change)
            ssim_fg_change_str = _format_change(ssim_fg_change)
            ssim_bg_change_str = _format_change(ssim_bg_change)
            mse_fg_change_str = _format_change(mse_fg_change)
            mse_bg_change_str = _format_change(mse_bg_change)
            lpips_fg_change_str = _format_change(lpips_fg_change)
            lpips_bg_change_str = _format_change(lpips_bg_change)
            fvmd_fg_change_str = _format_change(fvmd_fg_change)
            fvmd_bg_change_str = _format_change(fvmd_bg_change)
            vmaf_fg_change_str = _format_change(vmaf_fg_change)
            vmaf_bg_change_str = _format_change(vmaf_bg_change)
            print(f'{video_name:<20} {psnr_fg_change_str:<15} {psnr_bg_change_str:<15} {ssim_fg_change_str:<15} {ssim_bg_change_str:<15} {mse_fg_change_str:<15} {mse_bg_change_str:<15} {lpips_fg_change_str:<15} {lpips_bg_change_str:<15} {fvmd_fg_change_str:<15} {fvmd_bg_change_str:<15} {vmaf_fg_change_str:<15} {vmaf_bg_change_str:<15}')
        print(f"{'-' * 180}")
def run_presley(config: PresleyConfig) -> Dict[str, Any]:
    """Execute the full Elvis pipeline with the supplied configuration."""
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    script_dir = Path(__file__).resolve().parent
    os.chdir(str(script_dir))
    reference_video = config.reference_video
    (width, height) = (config.width, config.height)
    block_size = config.block_size
    shrink_amount = config.shrink_amount
    video_name = Path(reference_video).stem
    if config.experiment_dir:
        experiment_dir = os.path.abspath(config.experiment_dir)
    else:
        experiment_dir = os.path.abspath(f'experiment_{video_name}_w{width}_h{height}_bs{block_size}_shrink{shrink_amount}')
    os.makedirs(experiment_dir, exist_ok=True)
    execution_times: Dict[str, float] = {}
    approach_times = defaultdict(float)
    cap = cv2.VideoCapture(reference_video)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not framerate or framerate <= 0:
        framerate = 30.0
    target_bitrate = config.target_bitrate_override
    if target_bitrate is None:
        target_bitrate = calculate_target_bitrate(width, height, framerate, quality_factor=config.quality_factor)
    config_dict = asdict(config)
    pipeline_params: Dict[str, Any] = {'config': config_dict, 'derived': {'framerate': framerate, 'target_bitrate': target_bitrate, 'experiment_dir': experiment_dir, 'quality_factor': config.quality_factor}, 'functions': {'calculate_removability_scores': {'alpha': config.removability_alpha, 'smoothing_beta': config.removability_smoothing_beta, 'block_size': block_size}, 'apply_selective_removal': {'shrink_amount': shrink_amount}, 'inpaint_with_propainter': {'resize_ratio': config.propainter_resize_ratio, 'ref_stride': config.propainter_ref_stride, 'neighbor_length': config.propainter_neighbor_length, 'subvideo_length': config.propainter_subvideo_length, 'mask_dilation': config.propainter_mask_dilation, 'raft_iter': config.propainter_raft_iter, 'fp16': config.propainter_fp16, 'devices': list(config.propainter_devices) if config.propainter_devices else None, 'parallel_chunk_length': config.propainter_parallel_chunk_length, 'chunk_overlap': config.propainter_chunk_overlap}, 'inpaint_with_e2fgvi': {'ref_stride': config.e2fgvi_ref_stride, 'neighbor_stride': config.e2fgvi_neighbor_stride, 'num_ref': config.e2fgvi_num_ref, 'mask_dilation': config.e2fgvi_mask_dilation, 'devices': list(config.e2fgvi_devices) if config.e2fgvi_devices else None, 'parallel_chunk_length': config.e2fgvi_parallel_chunk_length, 'chunk_overlap': config.e2fgvi_chunk_overlap}, 'restore_downsampled_with_realesrgan': {'denoise_strength': config.realesrgan_denoise_strength, 'tile': config.realesrgan_tile, 'tile_pad': config.realesrgan_tile_pad, 'pre_pad': config.realesrgan_pre_pad, 'fp32': config.realesrgan_fp32, 'devices': list(config.realesrgan_devices) if config.realesrgan_devices else None, 'parallel_chunk_length': config.realesrgan_parallel_chunk_length, 'per_device_workers': config.realesrgan_per_device_workers}, 'restore_with_instantir_adaptive': {'cfg': config.instantir_cfg, 'creative_start': config.instantir_creative_start, 'preview_start': config.instantir_preview_start, 'seed': config.instantir_seed, 'devices': list(config.instantir_devices) if config.instantir_devices else None, 'batch_size': config.instantir_batch_size, 'parallel_chunk_length': config.instantir_parallel_chunk_length}, 'analyze_encoding_performance': {'generate_opencv_benchmarks': config.generate_opencv_benchmarks, 'metric_stride': config.metric_stride, 'fvmd_stride': config.fvmd_stride, 'fvmd_max_frames': config.fvmd_max_frames, 'fvmd_processes': config.fvmd_processes, 'fvmd_early_stop_delta': config.fvmd_early_stop_delta, 'fvmd_early_stop_window': config.fvmd_early_stop_window, 'vmaf_stride': config.vmaf_stride, 'enable_fvmd': config.enable_fvmd}}}
    encode_function_params = {'preset': config.encode_preset, 'pix_fmt': config.encode_pix_fmt, 'target_bitrate': target_bitrate}
    pipeline_params['functions']['encode_video'] = encode_function_params
    print(f'Processing video: {reference_video}')
    print(f'Target resolution: {width}x{height}')
    print(f'Calculated target bitrate: {target_bitrate} bps ({target_bitrate / 1000000:.1f} Mbps) for {width}x{height}@{framerate:.1f}fps')
    start = time.time()
    frames_dir = os.path.join(experiment_dir, 'frames')
    reference_frames_dir = os.path.join(frames_dir, 'reference')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(reference_frames_dir, exist_ok=True)
    print('Converting video to raw YUV format...')
    raw_video_path = os.path.join(experiment_dir, 'reference_raw.yuv')
    subprocess.run(f'ffmpeg -hide_banner -loglevel error -y -i {reference_video} -vf scale={width}:{height} -c:v rawvideo -pix_fmt yuv420p {raw_video_path}', shell=True, check=False)
    print('Extracting reference frames...')
    subprocess.run(f'ffmpeg -hide_banner -loglevel error -y -video_size {width}x{height} -r {framerate} -pixel_format yuv420p -i {raw_video_path} -q:v 2 {reference_frames_dir}/%05d.png', shell=True, check=False)
    frame_files = sorted([f for f in os.listdir(reference_frames_dir) if f.endswith('.png')])
    reference_frames = [cv2.imread(os.path.join(reference_frames_dir, f)) for f in frame_files]
    end = time.time()
    execution_times['Preprocessing'] = end - start
    print(f'Video preprocessing completed in {end - start:.2f} seconds.\n')
    start = time.time()
    print(f'Calculating removability scores with block size: {block_size}x{block_size}')
    removability_scores = calculate_removability_scores(raw_video_file=raw_video_path, reference_frames_folder=reference_frames_dir, width=width, height=height, block_size=block_size, alpha=config.removability_alpha, working_dir=experiment_dir, smoothing_beta=config.removability_smoothing_beta)
    end = time.time()
    execution_times['Removability Calculation'] = end - start
    print(f'Removability scores calculation completed in {end - start:.2f} seconds.\n')
    start = time.time()
    print('Encoding reference frames with two-pass for baseline comparison...')
    baseline_video = os.path.join(experiment_dir, 'baseline.mp4')
    encode_video(input_frames_dir=reference_frames_dir, output_video=baseline_video, framerate=framerate, width=width, height=height, target_bitrate=target_bitrate, preset=config.encode_preset, pix_fmt=config.encode_pix_fmt)
    end = time.time()
    duration = end - start
    approach_times[APPROACH_BASELINE] += duration
    print(f'Baseline encoding completed in {duration:.2f} seconds.\n')
    start = time.time()
    print(f'Shrinking and encoding frames with {APPROACH_ELVIS}...')
    shrunk_frames_dir = os.path.join(experiment_dir, 'frames', 'shrunk')
    os.makedirs(shrunk_frames_dir, exist_ok=True)
    (shrunk_frames, removal_masks, block_coords_to_remove) = zip(*[apply_selective_removal(img, scores, block_size, shrink_amount=shrink_amount) for (img, scores) in zip(reference_frames, removability_scores)])
    for (i, frame) in enumerate(shrunk_frames):
        cv2.imwrite(os.path.join(shrunk_frames_dir, f'{i + 1:05d}.png'), frame)
    shrunk_video = os.path.join(experiment_dir, 'shrunk.mp4')
    shrunk_width = shrunk_frames[0].shape[1]
    encode_video(input_frames_dir=shrunk_frames_dir, output_video=shrunk_video, framerate=framerate, width=shrunk_width, height=height, target_bitrate=target_bitrate, preset=config.encode_preset, pix_fmt=config.encode_pix_fmt)
    removal_masks_np = np.array(removal_masks, dtype=np.uint8)
    masks_packed = np.packbits(removal_masks_np)
    np.savez(os.path.join(experiment_dir, f'shrink_masks_{block_size}.npz'), packed=masks_packed, shape=removal_masks_np.shape)
    end = time.time()
    duration = end - start
    approach_times[APPROACH_ELVIS] += duration
    print(f'{APPROACH_ELVIS} shrinking completed in {duration:.2f} seconds.\n')
    start = time.time()
    print('Encoding frames with ROI-based adaptive quantization...')
    adaptive_video = os.path.join(experiment_dir, 'adaptive.mp4')
    maps_dir = os.path.join(experiment_dir, 'maps')
    qp_maps_dir = os.path.join(maps_dir, 'qp_maps')
    os.makedirs(maps_dir, exist_ok=True)
    valid_ctu_sizes = [16, 32, 64]
    roi_ctu_size = min(valid_ctu_sizes, key=lambda x: abs(x - block_size))
    pipeline_params['functions']['encode_with_roi'] = {'target_bitrate': target_bitrate, 'ctu_size': roi_ctu_size}
    encode_with_roi(input_frames_dir=reference_frames_dir, output_video=adaptive_video, removability_scores=removability_scores, block_size=block_size, framerate=framerate, width=width, height=height, target_bitrate=target_bitrate, save_qp_maps=True, qp_maps_dir=qp_maps_dir)
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_QP] += duration
    print(f'{APPROACH_PRESLEY_QP} encoding completed in {duration:.2f} seconds.\n')
    start = time.time()
    print(f'Applying {APPROACH_PRESLEY_REALESRGAN} adaptive filtering and encoding...')
    downsampled_frames_dir = os.path.join(experiment_dir, 'frames', 'downsampled')
    os.makedirs(downsampled_frames_dir, exist_ok=True)
    (downsampled_frames, downsample_maps) = zip(*[filter_frame_downsample(img, scores, block_size) for (img, scores) in zip(reference_frames, removability_scores)])
    for (i, frame) in enumerate(downsampled_frames):
        cv2.imwrite(os.path.join(downsampled_frames_dir, f'{i + 1:05d}.png'), frame)
    downsampled_video = os.path.join(experiment_dir, 'downsampled_encoded.mp4')
    encode_video(input_frames_dir=downsampled_frames_dir, output_video=downsampled_video, framerate=framerate, width=width, height=height, target_bitrate=target_bitrate, preset=config.encode_preset, pix_fmt=config.encode_pix_fmt)
    downsample_maps_file = os.path.join(maps_dir, 'downsample_maps.npz')
    encode_strength_maps_to_npz(strength_maps=list(downsample_maps), output_path=downsample_maps_file)
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_REALESRGAN] += duration
    print(f'{APPROACH_PRESLEY_REALESRGAN} filtering and encoding completed in {duration:.2f} seconds.\n')
    start = time.time()
    print(f'Applying {APPROACH_PRESLEY_INSTANTIR} adaptive filtering and encoding...')
    gaussian_frames_dir = os.path.join(experiment_dir, 'frames', 'gaussian')
    os.makedirs(gaussian_frames_dir, exist_ok=True)
    (gaussian_frames, gaussian_maps) = zip(*[filter_frame_gaussian(img, scores, block_size) for (img, scores) in zip(reference_frames, removability_scores)])
    for (i, frame) in enumerate(gaussian_frames):
        cv2.imwrite(os.path.join(gaussian_frames_dir, f'{i + 1:05d}.png'), frame)
    gaussian_video = os.path.join(experiment_dir, 'gaussian_encoded.mp4')
    encode_video(input_frames_dir=gaussian_frames_dir, output_video=gaussian_video, framerate=framerate, width=width, height=height, target_bitrate=target_bitrate, preset=config.encode_preset, pix_fmt=config.encode_pix_fmt)
    gaussian_maps_file = os.path.join(maps_dir, 'gaussian_maps.npz')
    encode_strength_maps_to_npz(strength_maps=list(gaussian_maps), output_path=gaussian_maps_file)
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_INSTANTIR] += duration
    print(f'{APPROACH_PRESLEY_INSTANTIR} filtering and encoding completed in {duration:.2f} seconds.\n')
    start = time.time()
    print(f'Decoding and stretching {APPROACH_ELVIS} video...')
    removal_masks_file = np.load(os.path.join(experiment_dir, f'shrink_masks_{block_size}.npz'))
    removal_masks_loaded = np.unpackbits(removal_masks_file['packed'])
    removal_masks = removal_masks_loaded[:np.prod(removal_masks_file['shape'])].reshape(removal_masks_file['shape'])
    stretched_frames_dir = os.path.join(experiment_dir, 'frames', 'stretched')
    if not decode_video(shrunk_video, stretched_frames_dir, framerate=framerate):
        raise RuntimeError(f'Failed to decode shrunk video: {shrunk_video}')
    num_masks = len(removal_masks)
    stretched_frames = [stretch_frame(cv2.imread(os.path.join(stretched_frames_dir, f'{i + 1:05d}.png')), removal_masks[i], block_size) for i in range(num_masks)]
    for (i, frame) in enumerate(stretched_frames):
        cv2.imwrite(os.path.join(stretched_frames_dir, f'{i + 1:05d}.png'), frame)
    removal_masks_dir = os.path.join(maps_dir, 'removal_masks')
    os.makedirs(removal_masks_dir, exist_ok=True)
    for (i, mask) in enumerate(removal_masks):
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(removal_masks_dir, f'{i + 1:05d}.png'), mask_img)
    (num_blocks_y, num_blocks_x) = removal_masks[0].shape
    print(f'Removal masks saved at block resolution ({num_blocks_y}x{num_blocks_x})')
    removal_masks_fullres_dir = os.path.join(experiment_dir, 'frames', 'removal_masks_fullres')
    os.makedirs(removal_masks_fullres_dir, exist_ok=True)
    for (i, mask) in enumerate(removal_masks):
        mask_img = (mask * 255).astype(np.uint8)
        mask_fullres = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(removal_masks_fullres_dir, f'{i + 1:05d}.png'), mask_fullres)
    print(f'Full-resolution masks for inpainting saved to {removal_masks_fullres_dir}')
    end = time.time()
    duration = end - start
    approach_times[APPROACH_ELVIS] += duration
    print(f'{APPROACH_ELVIS} stretching completed in {duration:.2f} seconds.\n')
    stretched_video = os.path.join(experiment_dir, 'stretched.mp4')
    encode_video(input_frames_dir=stretched_frames_dir, output_video=stretched_video, framerate=framerate, width=width, height=height, target_bitrate=None)
    start = time.time()
    print('Inpainting stretched frames with CV2...')
    inpainted_cv2_frames_dir = os.path.join(experiment_dir, 'frames', 'inpainted_cv2')
    os.makedirs(inpainted_cv2_frames_dir, exist_ok=True)
    for i in range(len(removal_masks)):
        stretched_frame = cv2.imread(os.path.join(stretched_frames_dir, f'{i + 1:05d}.png'))
        mask_img = cv2.imread(os.path.join(removal_masks_fullres_dir, f'{i + 1:05d}.png'), cv2.IMREAD_GRAYSCALE)
        inpainted_frame = cv2.inpaint(stretched_frame, mask_img, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(inpainted_cv2_frames_dir, f'{i + 1:05d}.png'), inpainted_frame)
    end = time.time()
    duration = end - start
    approach_times[APPROACH_ELVIS_CV2] += duration
    print(f'{APPROACH_ELVIS_CV2} inpainting completed in {duration:.2f} seconds.\n')
    inpainted_cv2_video = os.path.join(experiment_dir, 'inpainted_cv2.mp4')
    encode_video(input_frames_dir=inpainted_cv2_frames_dir, output_video=inpainted_cv2_video, framerate=framerate, width=width, height=height, target_bitrate=None)
    start = time.time()
    print('Inpainting stretched frames with ProPainter...')
    inpainted_frames_dir = os.path.join(experiment_dir, 'frames', 'inpainted')
    inpaint_with_propainter(stretched_frames_dir=stretched_frames_dir, removal_masks_dir=removal_masks_fullres_dir, output_frames_dir=inpainted_frames_dir, width=width, height=height, framerate=framerate, resize_ratio=config.propainter_resize_ratio, ref_stride=config.propainter_ref_stride, neighbor_length=config.propainter_neighbor_length, subvideo_length=config.propainter_subvideo_length, mask_dilation=config.propainter_mask_dilation, raft_iter=config.propainter_raft_iter, fp16=config.propainter_fp16, devices=list(config.propainter_devices) if config.propainter_devices else None, parallel_chunk_length=config.propainter_parallel_chunk_length, chunk_overlap=config.propainter_chunk_overlap)
    end = time.time()
    duration = end - start
    approach_times[APPROACH_ELVIS_PROP] += duration
    print(f'{APPROACH_ELVIS_PROP} inpainting completed in {duration:.2f} seconds.\n')
    inpainted_video = os.path.join(experiment_dir, 'inpainted_propainter.mp4')
    encode_video(input_frames_dir=inpainted_frames_dir, output_video=inpainted_video, framerate=framerate, width=width, height=height, target_bitrate=None)
    start = time.time()
    print('Inpainting stretched frames with E2FGVI...')
    inpainted_e2fgvi_frames_dir = os.path.join(experiment_dir, 'frames', 'inpainted_e2fgvi')
    inpaint_with_e2fgvi(stretched_frames_dir=stretched_frames_dir, removal_masks_dir=removal_masks_fullres_dir, output_frames_dir=inpainted_e2fgvi_frames_dir, width=width, height=height, framerate=framerate, ref_stride=config.e2fgvi_ref_stride, neighbor_stride=config.e2fgvi_neighbor_stride, num_ref=config.e2fgvi_num_ref, mask_dilation=config.e2fgvi_mask_dilation, devices=config.e2fgvi_devices, parallel_chunk_length=config.e2fgvi_parallel_chunk_length, chunk_overlap=config.e2fgvi_chunk_overlap)
    end = time.time()
    duration = end - start
    approach_times[APPROACH_ELVIS_E2FGVI] += duration
    print(f'{APPROACH_ELVIS_E2FGVI} inpainting completed in {duration:.2f} seconds.\n')
    inpainted_e2fgvi_video = os.path.join(experiment_dir, 'inpainted_e2fgvi.mp4')
    encode_video(input_frames_dir=inpainted_e2fgvi_frames_dir, output_video=inpainted_e2fgvi_video, framerate=framerate, width=width, height=height, target_bitrate=None)
    start = time.time()
    print(f'Decoding {APPROACH_PRESLEY_REALESRGAN} video and strength maps...')
    downsampled_frames_decoded_dir = os.path.join(experiment_dir, 'frames', 'downsampled_decoded')
    if not decode_video(downsampled_video, downsampled_frames_decoded_dir, framerate=framerate):
        raise RuntimeError(f'Failed to decode downsampled video: {downsampled_video}')
    downsample_maps_file = os.path.join(maps_dir, 'downsample_maps.npz')
    strength_maps = decode_strength_maps_from_npz(downsample_maps_file)
    downsampled_maps_decoded_dir = os.path.join(experiment_dir, 'maps', 'downsampled_maps_decoded')
    os.makedirs(downsampled_maps_decoded_dir, exist_ok=True)
    for (i, map_frame) in enumerate(strength_maps):
        map_img = np.clip(map_frame.astype(np.float32) * 25.5, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(downsampled_maps_decoded_dir, f'{i + 1:05d}.png'), map_img)
    print(f'  Decoded downsample maps saved to {downsampled_maps_decoded_dir} at block resolution ({strength_maps.shape[1]}x{strength_maps.shape[2]})')
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_REALESRGAN] += duration
    print(f'Decoding completed in {duration:.2f} seconds.\n')
    start = time.time()
    print(f'Applying adaptive upsampling restoration for {APPROACH_PRESLEY_REALESRGAN}...')
    downsampled_restored_frames_dir = os.path.join(experiment_dir, 'frames', 'downsampled_restored')
    os.makedirs(downsampled_restored_frames_dir, exist_ok=True)
    restore_downsampled_with_realesrgan(input_frames_dir=downsampled_frames_decoded_dir, output_frames_dir=downsampled_restored_frames_dir, downscale_maps=strength_maps, block_size=block_size, denoise_strength=config.realesrgan_denoise_strength, tile=config.realesrgan_tile, tile_pad=config.realesrgan_tile_pad, pre_pad=config.realesrgan_pre_pad, fp32=config.realesrgan_fp32, devices=list(config.realesrgan_devices) if config.realesrgan_devices else None, parallel_chunk_length=config.realesrgan_parallel_chunk_length, per_device_workers=config.realesrgan_per_device_workers)
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_REALESRGAN] += duration
    print(f'{APPROACH_PRESLEY_REALESRGAN} restoration completed in {duration:.2f} seconds.\n')
    downsampled_restored_video = os.path.join(experiment_dir, 'downsampled_restored.mp4')
    encode_video(input_frames_dir=downsampled_restored_frames_dir, output_video=downsampled_restored_video, framerate=framerate, width=width, height=height, target_bitrate=None)
    start = time.time()
    print(f'Decoding {APPROACH_PRESLEY_INSTANTIR} video and strength maps...')
    gaussian_frames_decoded_dir = os.path.join(experiment_dir, 'frames', 'gaussian_decoded')
    if not decode_video(gaussian_video, gaussian_frames_decoded_dir, framerate=framerate):
        raise RuntimeError(f'Failed to decode Gaussian video: {gaussian_video}')
    gaussian_maps_file = os.path.join(maps_dir, 'gaussian_maps.npz')
    strength_maps_gaussian = decode_strength_maps_from_npz(gaussian_maps_file)
    gaussian_maps_decoded_dir = os.path.join(experiment_dir, 'maps', 'gaussian_maps_decoded')
    os.makedirs(gaussian_maps_decoded_dir, exist_ok=True)
    for (i, map_frame) in enumerate(strength_maps_gaussian):
        map_img = np.clip(map_frame.astype(np.float32) * 25.5, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(gaussian_maps_decoded_dir, f'{i + 1:05d}.png'), map_img)
    print(f'  Decoded gaussian maps saved to {gaussian_maps_decoded_dir} at block resolution ({strength_maps_gaussian.shape[1]}x{strength_maps_gaussian.shape[2]})')
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_INSTANTIR] += duration
    print(f'Decoding completed in {duration:.2f} seconds.\n')
    start = time.time()
    print(f'Applying adaptive deblurring restoration for {APPROACH_PRESLEY_INSTANTIR}...')
    instantir_work_dir = os.path.join(experiment_dir, 'instantir_work')
    gaussian_instantir_input_dir = os.path.join(instantir_work_dir, 'gaussian_decoded')
    os.makedirs(gaussian_instantir_input_dir, exist_ok=True)
    decoded_gaussian_frames = [f for f in os.listdir(gaussian_frames_decoded_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f'  Copying {len(decoded_gaussian_frames)} frames to InstantIR input directory...')
    for frame_file in decoded_gaussian_frames:
        shutil.copy2(os.path.join(gaussian_frames_decoded_dir, frame_file), os.path.join(gaussian_instantir_input_dir, frame_file))
    restore_with_instantir_adaptive(input_frames_dir=gaussian_instantir_input_dir, blur_maps=strength_maps_gaussian, block_size=block_size, cfg=config.instantir_cfg, creative_start=config.instantir_creative_start, preview_start=config.instantir_preview_start, seed=config.instantir_seed, devices=list(config.instantir_devices) if config.instantir_devices else None, batch_size=config.instantir_batch_size, parallel_chunk_length=config.instantir_parallel_chunk_length)
    gaussian_restored_frames_dir = os.path.join(experiment_dir, 'frames', 'gaussian_restored')
    os.makedirs(gaussian_restored_frames_dir, exist_ok=True)
    print('  Copying restored frames to output directory...')
    for frame_file in os.listdir(gaussian_instantir_input_dir):
        if frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(os.path.join(gaussian_instantir_input_dir, frame_file), os.path.join(gaussian_restored_frames_dir, frame_file))
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_INSTANTIR] += duration
    print(f'{APPROACH_PRESLEY_INSTANTIR} restoration completed in {duration:.2f} seconds.\n')
    gaussian_restored_video = os.path.join(experiment_dir, 'gaussian_restored.mp4')
    encode_video(input_frames_dir=gaussian_restored_frames_dir, output_video=gaussian_restored_video, framerate=framerate, width=width, height=height, target_bitrate=None)
    print('Evaluating and comparing encoding performance...')
    start = time.time()
    if config.strength_maps_use_npz:
        downsample_maps_path = os.path.join(maps_dir, 'downsample_maps.npz')
        gaussian_maps_path = os.path.join(maps_dir, 'gaussian_maps.npz')
    else:
        downsample_maps_path = os.path.join(maps_dir, 'downsample_encoded.mp4')
        gaussian_maps_path = os.path.join(maps_dir, 'gaussian_encoded.mp4')
    video_sizes = {APPROACH_BASELINE: os.path.getsize(baseline_video), APPROACH_ELVIS: os.path.getsize(shrunk_video) + os.path.getsize(os.path.join(experiment_dir, f'shrink_masks_{block_size}.npz')), APPROACH_PRESLEY_QP: os.path.getsize(adaptive_video), APPROACH_PRESLEY_REALESRGAN: os.path.getsize(downsampled_video) + os.path.getsize(downsample_maps_path), APPROACH_PRESLEY_INSTANTIR: os.path.getsize(gaussian_video) + os.path.getsize(gaussian_maps_path)}
    frame_count = len(frame_files)
    duration = frame_count / framerate if framerate else frame_count
    bitrates = {key: size * 8 / duration for (key, size) in video_sizes.items()}
    print(f'\nEncoding Results (Target Bitrate: {target_bitrate} bps / {target_bitrate / 1000000:.1f} Mbps):')
    for (key, bitrate) in bitrates.items():
        print(f'{key} bitrate: {bitrate / 1000000:.2f} Mbps')
    encoded_videos = {APPROACH_BASELINE: baseline_video, APPROACH_PRESLEY_QP: adaptive_video, APPROACH_ELVIS_CV2: inpainted_cv2_video, APPROACH_ELVIS_PROP: inpainted_video, APPROACH_ELVIS_E2FGVI: inpainted_e2fgvi_video, APPROACH_PRESLEY_REALESRGAN: downsampled_restored_video, APPROACH_PRESLEY_INSTANTIR: gaussian_restored_video}
    ufo_masks_dir = os.path.join(experiment_dir, 'maps', 'ufo_masks')
    strength_maps_dict = {APPROACH_PRESLEY_REALESRGAN: strength_maps, APPROACH_PRESLEY_INSTANTIR: strength_maps_gaussian}
    if config.generate_opencv_benchmarks:
        (opencv_benchmarks, opencv_bitrates) = generate_opencv_benchmarks(reference_frames=reference_frames, strength_maps=strength_maps_dict, block_size=block_size, framerate=framerate, width=width, height=height, temp_dir=experiment_dir, video_bitrates=bitrates)
        encoded_videos.update(opencv_benchmarks)
        bitrates.update(opencv_bitrates)
    analysis_results = analyze_encoding_performance(reference_frames=reference_frames, encoded_videos=encoded_videos, block_size=block_size, width=width, height=height, temp_dir=experiment_dir, masks_dir=ufo_masks_dir, video_bitrates=bitrates, framerate=framerate, metric_stride=config.metric_stride, fvmd_stride=config.fvmd_stride, fvmd_max_frames=config.fvmd_max_frames, fvmd_early_stop_delta=config.fvmd_early_stop_delta, fvmd_early_stop_window=config.fvmd_early_stop_window, vmaf_stride=config.vmaf_stride, enable_fvmd=config.enable_fvmd)
    end = time.time()
    execution_times['Performance Evaluation'] = end - start
    for (approach, total) in approach_times.items():
        execution_times[approach] = total
    analysis_results['execution_times_seconds'] = execution_times
    analysis_results['video_name'] = reference_video
    analysis_results['video_length_seconds'] = duration
    analysis_results['video_framerate'] = framerate
    analysis_results['video_resolution'] = f'{width}x{height}'
    analysis_results['block_size'] = block_size
    analysis_results['target_bitrate_bps'] = target_bitrate
    results_json_path = os.path.join(experiment_dir, 'analysis_results.json')
    pipeline_params['derived']['analysis_results_path'] = results_json_path
    analysis_results['parameters'] = pipeline_params
    analysis_results['experiment_dir'] = experiment_dir
    analysis_results['analysis_results_path'] = results_json_path
    with open(results_json_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    print(f'Analysis results saved to: {results_json_path}')
    return analysis_results
def _load_config_from_cli() -> PresleyConfig:
    parser = argparse.ArgumentParser(description='Run the ELVIS pipeline with configurable parameters.')
    parser.add_argument('--config', type=str, help='Path to a JSON file containing PresleyConfig fields.')
    parser.add_argument('--reference-video', type=str, help='Path to the input reference video.')
    parser.add_argument('--width', type=int, help='Target frame width.')
    parser.add_argument('--height', type=int, help='Target frame height.')
    parser.add_argument('--block-size', type=int, help='Processing block size.')
    parser.add_argument('--shrink-amount', type=float, help='Shrink amount for ELVIS.')
    parser.add_argument('--quality-factor', type=float, help='Quality factor for target bitrate calculation.')
    parser.add_argument('--target-bitrate', type=int, help='Override target bitrate in bits per second')
    parser.add_argument('--removability-alpha', type=float, help='Alpha parameter for removability scoring.')
    parser.add_argument('--removability-smoothing-beta', type=float, help='Smoothing beta for removability scoring.')
    parser.add_argument('--encode-preset', type=str, help='FFmpeg preset for encoding (e.g., medium, fast, slow).')
    parser.add_argument('--encode-pix-fmt', type=str, help='Pixel format for encoding (e.g., yuv420p).')
    parser.add_argument('--generate-opencv-benchmarks', dest='generate_opencv_benchmarks', action='store_true', help='Enable OpenCV baseline generation.')
    parser.add_argument('--disable-opencv-benchmarks', dest='generate_opencv_benchmarks', action='store_false', help='Disable OpenCV baseline generation.')
    parser.set_defaults(generate_opencv_benchmarks=None)
    parser.add_argument('--metric-stride', type=int, help='Stride for PSNR/SSIM/LPIPS metrics.')
    parser.add_argument('--fvmd-stride', type=int, help='Stride for FVMD computation.')
    parser.add_argument('--fvmd-max-frames', type=int, help='Maximum frames for FVMD computation.')
    parser.add_argument('--fvmd-processes', type=int, help='Number of FVMD worker processes.')
    parser.add_argument('--fvmd-early-stop-delta', type=float, help='Early stop delta for FVMD.')
    parser.add_argument('--fvmd-early-stop-window', type=int, help='Early stop window for FVMD.')
    parser.add_argument('--vmaf-stride', type=int, help='Stride for VMAF computation.')
    args = parser.parse_args()
    config_data: Dict[str, Any] = asdict(PresleyConfig())
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config_data.update(file_config)
    overrides = {'reference_video': args.reference_video, 'width': args.width, 'height': args.height, 'block_size': args.block_size, 'shrink_amount': args.shrink_amount, 'quality_factor': args.quality_factor, 'target_bitrate_override': args.target_bitrate, 'removability_alpha': args.removability_alpha, 'removability_smoothing_beta': args.removability_smoothing_beta, 'encode_preset': args.encode_preset, 'encode_pix_fmt': args.encode_pix_fmt, 'metric_stride': args.metric_stride, 'fvmd_stride': args.fvmd_stride, 'fvmd_max_frames': args.fvmd_max_frames, 'fvmd_processes': args.fvmd_processes, 'fvmd_early_stop_delta': args.fvmd_early_stop_delta, 'fvmd_early_stop_window': args.fvmd_early_stop_window, 'vmaf_stride': args.vmaf_stride}
    for (key, value) in overrides.items():
        if value is not None:
            config_data[key] = value
    if args.generate_opencv_benchmarks is not None:
        config_data['generate_opencv_benchmarks'] = args.generate_opencv_benchmarks
    return PresleyConfig(**config_data)
def main() -> None:
    config = _load_config_from_cli()
    results = run_presley(config)
    path = results.get('analysis_results_path')
    if path:
        print(f'\nFinal analysis JSON: {path}')
if __name__ == '__main__':
    main()
