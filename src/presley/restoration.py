import tempfile, multiprocessing, functools, gc
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple
from instantir import InstantIRRuntime, load_runtime, restore_images_batch
from PIL import Image
import os
import sys
import math
from presley.utils import *
from presley.degradation import split_image_into_blocks, combine_blocks_into_image
from presley.io import clear_directory, load_frame, save_frame, get_frame_paths
from presley.concurrency import chunk_for_devices, _resolve_device_list
from presley.encoding import decode_video
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

def stretch_frame(shrunk_frame: np.ndarray, binary_mask: np.ndarray, block_size: int) -> np.ndarray:
    """Reconstruct full-resolution frame from shrunk version using removal mask."""
    (h_shrunk, w_shrunk, channels) = shrunk_frame.shape
    pad_y = (block_size - h_shrunk % block_size) % block_size
    pad_x = (block_size - w_shrunk % block_size) % block_size
    if pad_y > 0 or pad_x > 0:
        shrunk_frame = np.pad(shrunk_frame, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        binary_mask = np.pad(binary_mask, ((0, 1 if pad_y > 0 else 0), (0, 1 if pad_x > 0 else 0)), mode='edge')

    (num_blocks_y, num_blocks_x) = binary_mask.shape
    final_blocks = np.zeros((num_blocks_y, num_blocks_x, block_size, block_size, channels), dtype=shrunk_frame.dtype)
    shrunk_blocks = split_image_into_blocks(shrunk_frame, block_size)
    final_blocks[binary_mask == 0] = shrunk_blocks.reshape(-1, block_size, block_size, channels)
    reconstructed_image = combine_blocks_into_image(final_blocks)

    if pad_y > 0:
        reconstructed_image = reconstructed_image[:-pad_y, :, :]
    if pad_x > 0:
        reconstructed_image = reconstructed_image[:, :-pad_x, :]
    return reconstructed_image
def inpaint_with_propainter(stretched_frames_dir: str, removal_masks_dir: str, output_frames_dir: str, width: int, height: int, framerate: float, resize_ratio: float=1.0, ref_stride: int=20, neighbor_length: int=4, subvideo_length: int=40, mask_dilation: int=4, raft_iter: int=20, fp16: bool=True, devices: Optional[Sequence[Union[int, str, torch.device]]]=None, parallel_chunk_length: Optional[int]=None, chunk_overlap: Optional[int]=None) -> None:
    """Use ProPainter to inpaint stretched frames with removed blocks."""
    original_dir = os.getcwd()
    stretched_frames_abs = os.path.abspath(stretched_frames_dir)
    removal_masks_abs = os.path.abspath(removal_masks_dir)
    output_frames_abs = os.path.abspath(output_frames_dir)
    output_frames_path = Path(output_frames_abs)
    os.makedirs(output_frames_abs, exist_ok=True)
    for stale_frame in output_frames_path.glob('*.png'):
        if stale_frame.is_file():
            stale_frame.unlink()
    try:
        import propainter as _propainter
    except ImportError as exc:
        raise RuntimeError('propainter package is not installed. Install it with `pip install propainter`.') from exc
    try:
        frame_files = sorted([f for f in os.listdir(stretched_frames_abs) if f.lower().endswith(('.jpg', '.png'))])
        mask_files = sorted([f for f in os.listdir(removal_masks_abs) if f.lower().endswith(('.jpg', '.png'))])
        if len(frame_files) == 0 or len(frame_files) != len(mask_files):
            raise ValueError('Frame and mask counts must match and be non-zero for ProPainter input.')
        total_frames = len(frame_files)
        frame_paths = [Path(stretched_frames_abs) / f for f in frame_files]
        mask_paths = [Path(removal_masks_abs) / f for f in mask_files]
        resolved_devices = _resolve_device_list(devices, prefer_cuda=True, allow_cpu_fallback=True)
        effective_chunk = parallel_chunk_length if parallel_chunk_length is not None else subvideo_length
        if effective_chunk is None or effective_chunk <= 0:
            effective_chunk = total_frames
        effective_chunk = max(1, min(effective_chunk, total_frames))
        effective_overlap = chunk_overlap if chunk_overlap is not None else neighbor_length
        effective_overlap = max(0, effective_overlap)
        if effective_chunk == 1:
            effective_overlap = 0
        else:
            effective_overlap = min(effective_overlap, effective_chunk // 2)
        total_chunks = max(1, math.ceil(total_frames / effective_chunk))
        propainter_entry = [sys.executable, '-m', 'propainter.inference_propainter']
        run_cwd = original_dir
        base_flags = ['--width', str(width), '--height', str(height), '--resize_ratio', str(resize_ratio), '--ref_stride', str(ref_stride), '--neighbor_length', str(neighbor_length), '--mask_dilation', str(mask_dilation), '--raft_iter', str(raft_iter), '--save_fps', str(int(framerate))]
        if fp16:
            base_flags.append('--fp16')

        class _PropainterChunk(NamedTuple):
            job_id: int
            start: int
            end: int
            expanded_start: int
            expanded_end: int
        chunks: List[_PropainterChunk] = []
        cursor = 0
        job_id = 0
        while cursor < total_frames:
            end = min(total_frames, cursor + effective_chunk)
            expanded_start = max(0, cursor - effective_overlap)
            expanded_end = min(total_frames, end + effective_overlap)
            if expanded_end <= expanded_start:
                expanded_end = min(total_frames, expanded_start + effective_chunk)
            chunks.append(_PropainterChunk(job_id=job_id, start=cursor, end=end, expanded_start=expanded_start, expanded_end=expanded_end))
            job_id += 1
            cursor = end
        device_summary = ', '.join((str(dev) for dev in resolved_devices))
        print(f'Using ProPainter on devices: {device_summary} | chunk length: {effective_chunk} | overlap: {effective_overlap}')
        print(f'Total frames: {total_frames} | parallel chunks: {len(chunks)}')

        def _visible_device_token(device: torch.device) -> Optional[str]:
            if device.type == 'cuda':
                return str(device.index if device.index is not None else 0)
            return None

        def _run_chunk(chunk: _PropainterChunk, device: torch.device) -> None:
            visible_token = _visible_device_token(device)
            env = os.environ.copy()
            if visible_token is not None:
                env['CUDA_VISIBLE_DEVICES'] = visible_token
            else:
                env.pop('CUDA_VISIBLE_DEVICES', None)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                video_input_dir = temp_dir_path / f'propainter_job_{chunk.job_id:04d}'
                mask_input_dir = temp_dir_path / f'propainter_masks_{chunk.job_id:04d}'
                output_root_dir = temp_dir_path / f'propainter_output_{chunk.job_id:04d}'
                video_input_dir.mkdir(parents=True, exist_ok=True)
                mask_input_dir.mkdir(parents=True, exist_ok=True)
                output_root_dir.mkdir(parents=True, exist_ok=True)
                expanded_indices = list(range(chunk.expanded_start, chunk.expanded_end))
                for (local_idx, frame_idx) in enumerate(expanded_indices):
                    shutil.copy(frame_paths[frame_idx], video_input_dir / f'{local_idx:04d}.png')
                    shutil.copy(mask_paths[frame_idx], mask_input_dir / f'{local_idx:04d}.png')
                chunk_length = max(1, chunk.expanded_end - chunk.expanded_start)
                sub_len = subvideo_length if subvideo_length and subvideo_length > 0 else chunk_length
                chunk_specific_flags = base_flags + ['--subvideo_length', str(max(1, min(sub_len, chunk_length)))]
                cmd = propainter_entry + ['--video', str(video_input_dir), '--mask', str(mask_input_dir), '--output', str(output_root_dir)] + chunk_specific_flags
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=run_cwd, env=env)
                if result.returncode != 0:
                    print(f'ProPainter stdout (chunk {chunk.job_id}): {result.stdout}')
                    print(f'ProPainter stderr (chunk {chunk.job_id}): {result.stderr}')
                    raise RuntimeError(f'ProPainter inference failed for chunk {chunk.job_id}. See logs above for details.')
                video_name = video_input_dir.name
                generated_frames_dir = output_root_dir / video_name / 'frames'
                if not generated_frames_dir.exists():
                    raise RuntimeError(f'ProPainter did not emit frames for chunk {chunk.job_id} at {generated_frames_dir}')
                generated_files = sorted([p for p in generated_frames_dir.iterdir() if p.suffix.lower() == '.png'])
                if not generated_files:
                    raise RuntimeError(f'No frames produced by ProPainter for chunk {chunk.job_id} in {generated_frames_dir}')
                skip_prefix = chunk.start - chunk.expanded_start
                keep_count = chunk.end - chunk.start
                selected_files = generated_files[skip_prefix:skip_prefix + keep_count]
                if len(selected_files) != keep_count:
                    raise RuntimeError(f'Unexpected frame count for chunk {chunk.job_id}: expected {keep_count}, got {len(selected_files)}')
                for (offset, frame_path) in enumerate(selected_files):
                    dst_frame = output_frames_path / f'{chunk.start + offset + 1:05d}.png'
                    shutil.copy(frame_path, dst_frame)
                print(f'  ✓ ProPainter chunk {chunk.job_id + 1}/{len(chunks)} frames {chunk.start + 1}-{chunk.end} on {device}')
        max_workers = min(len(resolved_devices), len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_iter = iter(chunks)
            while True:
                tasks = []
                for device in resolved_devices:
                    chunk = next(chunk_iter, None)
                    if chunk is None:
                        break
                    tasks.append(executor.submit(_run_chunk, chunk, device))
                if not tasks:
                    break
                for future in tasks:
                    future.result()
        print(f'Inpainted frames saved to {output_frames_abs}')
    except Exception as exc:
        print(f'Error in inpaint_with_propainter: {exc}')
        raise
    finally:
        os.chdir(original_dir)
def inpaint_with_e2fgvi(stretched_frames_dir: str, removal_masks_dir: str, output_frames_dir: str, width: int, height: int, framerate: float, ref_stride: int=10, neighbor_stride: int=5, num_ref: int=-1, mask_dilation: int=4, devices: Optional[Sequence[Union[int, str, torch.device]]]=None, parallel_chunk_length: Optional[int]=None, chunk_overlap: Optional[int]=None) -> None:
    """Use E2FGVI to inpaint stretched frames with removed blocks. Supports multi-GPU parallelism."""
    stretched_frames_abs = os.path.abspath(stretched_frames_dir)
    removal_masks_abs = os.path.abspath(removal_masks_dir)
    output_frames_abs = os.path.abspath(output_frames_dir)
    frames_path = Path(stretched_frames_abs)
    masks_path = Path(removal_masks_abs)
    output_path = Path(output_frames_abs)
    if not frames_path.is_dir():
        raise ValueError(f'Stretched frames directory does not exist: {stretched_frames_abs}')
    if not masks_path.is_dir():
        raise ValueError(f'Removal masks directory does not exist: {removal_masks_abs}')
    output_path.mkdir(parents=True, exist_ok=True)
    valid_suffixes = ('.png', '.jpg', '.jpeg')
    frame_paths = sorted([p for p in frames_path.iterdir() if p.suffix.lower() in valid_suffixes])
    if not frame_paths:
        raise ValueError(f'No frames found in {stretched_frames_abs}')
    mask_paths = sorted([p for p in masks_path.iterdir() if p.suffix.lower() in valid_suffixes])
    if len(frame_paths) != len(mask_paths):
        raise ValueError(f'Frame count ({len(frame_paths)}) does not match mask count ({len(mask_paths)}).')
    for (frame_file, mask_file) in zip(frame_paths, mask_paths):
        if frame_file.stem != mask_file.stem:
            raise ValueError(f'Frame/mask mismatch: {frame_file.name} vs {mask_file.name}')
    for stale_file in output_path.iterdir():
        if stale_file.is_file() and stale_file.suffix.lower() in valid_suffixes:
            stale_file.unlink()
    try:
        import e2fgvi as e2fgvi_pkg
    except ImportError as exc:
        raise RuntimeError('E2FGVI package is not installed. Install it with `pip install e2fgvi`.') from exc
    package_dir = Path(e2fgvi_pkg.__file__).resolve().parent
    base_cmd_prefix = [sys.executable, '-m', 'e2fgvi']
    ckpt_path = package_dir / 'release_model' / 'E2FGVI-HQ-CVPR22.pth'
    if framerate is None:
        raise ValueError('`framerate` must be provided for E2FGVI inference.')

    def _build_command(video_dir: str, mask_dir: str, save_dir: str) -> List[str]:
        cmd = list(base_cmd_prefix)
        cmd.extend(['--model', 'e2fgvi_hq', '--video', video_dir, '--mask', mask_dir, '--ckpt', str(ckpt_path), '--step', str(ref_stride), '--num_ref', str(num_ref), '--neighbor_stride', str(neighbor_stride), '--set_size', '--width', str(width), '--height', str(height), '--savefps', str(int(framerate)), '--save_frames', save_dir])
        return cmd
    total_frames = len(frame_paths)
    resolved_devices = _resolve_device_list(devices, prefer_cuda=True, allow_cpu_fallback=True)
    cuda_devices = [dev for dev in resolved_devices if dev.type == 'cuda']
    preferred_device: Optional[torch.device] = None
    if cuda_devices:
        preferred_device = cuda_devices[0]
    elif resolved_devices:
        preferred_device = resolved_devices[0]

    def _device_label(device_obj: Optional[torch.device]) -> str:
        if device_obj is None:
            return 'default'
        if device_obj.type == 'cuda':
            idx = device_obj.index if device_obj.index is not None else 0
            return f'cuda:{idx}'
        return str(device_obj)

    def _run_single(device_override: Optional[torch.device]) -> None:
        print('Running E2FGVI inference...')
        cmd = _build_command(stretched_frames_abs, removal_masks_abs, output_frames_abs)
        env = os.environ.copy()
        if device_override is not None:
            if device_override.type == 'cuda':
                cuda_idx = device_override.index if device_override.index is not None else 0
                env['CUDA_VISIBLE_DEVICES'] = str(cuda_idx)
            else:
                env['CUDA_VISIBLE_DEVICES'] = ''
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f'E2FGVI stdout: {result.stdout}')
            print(f'E2FGVI stderr: {result.stderr}')
            raise RuntimeError(f'E2FGVI inference failed: {result.stderr}')
        if result.stdout:
            print(f'E2FGVI output: {result.stdout}')
        generated_frames = list(output_path.glob('*.png'))
        if generated_frames:
            print(f'E2FGVI inpainted frames saved to {output_frames_abs}')
            return
        results_dir = package_dir / 'results'
        if not results_dir.exists():
            raise RuntimeError(f'E2FGVI results directory not found at {results_dir}')
        result_videos = [f for f in results_dir.iterdir() if f.suffix == '.mp4']
        if not result_videos:
            raise RuntimeError(f'No result video found in {results_dir}')
        result_videos.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        result_video_path = result_videos[0]
        print('Decoding E2FGVI result video to frames...')
        if not decode_video(str(result_video_path), output_frames_abs, framerate=framerate, start_number=1, quality=1):
            raise RuntimeError(f'Failed to decode E2FGVI result video: {result_video_path}')
        print(f'E2FGVI inpainted frames saved to {output_frames_abs}')
        try:
            result_video_path.unlink()
        except OSError:
            pass

    def _run_parallel(device_list: Sequence[torch.device]) -> None:
        default_overlap = max(neighbor_stride, 1) * 2
        overlap = int(chunk_overlap) if chunk_overlap is not None else default_overlap
        overlap = max(0, overlap)
        if total_frames == 1:
            overlap = 0
        else:
            overlap = min(overlap, total_frames - 1)
        if parallel_chunk_length is None or parallel_chunk_length <= 0:
            chunk_len = math.ceil(total_frames / len(device_list))
        else:
            chunk_len = int(parallel_chunk_length)
        chunk_len = max(1, min(chunk_len, total_frames))
        if chunk_len <= overlap:
            chunk_len = min(total_frames, overlap + 1)

        class _E2FGVIChunk(NamedTuple):
            index: int
            chunk_start: int
            chunk_end: int
            core_start: int
            core_end: int
        chunks: List[_E2FGVIChunk] = []
        cursor = 0
        idx = 0
        while cursor < total_frames:
            core_end = min(total_frames, cursor + chunk_len)
            chunk_start = max(0, cursor - (overlap if cursor > 0 else 0))
            chunk_end = min(total_frames, core_end + (overlap if core_end < total_frames else 0))
            chunks.append(_E2FGVIChunk(index=idx, chunk_start=chunk_start, chunk_end=chunk_end, core_start=cursor, core_end=core_end))
            cursor = core_end
            idx += 1
        device_labels = [_device_label(dev) for dev in device_list]
        print('Running E2FGVI inference across multiple devices...')
        print(f"  Devices: {', '.join(device_labels)} | chunk length: {chunk_len} | overlap: {overlap} | total chunks: {len(chunks)}")

        def _link_or_copy(src: Path, dst: Path) -> None:
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        def _process_chunk(chunk: _E2FGVIChunk, device_obj: torch.device) -> None:
            device_label = _device_label(device_obj)
            print(f'    -> E2FGVI chunk {chunk.index + 1}/{len(chunks)} frames {chunk.core_start + 1}-{chunk.core_end} on {device_label}')
            with tempfile.TemporaryDirectory(prefix=f'e2fgvi_chunk_{chunk.index:03d}_') as tmp_root:
                tmp_root_path = Path(tmp_root)
                chunk_frames_path = tmp_root_path / 'frames'
                chunk_masks_path = tmp_root_path / 'masks'
                chunk_output_path = tmp_root_path / 'output'
                chunk_frames_path.mkdir(parents=True, exist_ok=True)
                chunk_masks_path.mkdir(parents=True, exist_ok=True)
                chunk_output_path.mkdir(parents=True, exist_ok=True)
                chunk_indices = list(range(chunk.chunk_start, chunk.chunk_end))
                for (seq_idx, original_idx) in enumerate(chunk_indices, start=1):
                    frame_src = frame_paths[original_idx]
                    mask_src = mask_paths[original_idx]
                    frame_dest = chunk_frames_path / f'{seq_idx:05d}{frame_src.suffix}'
                    mask_dest = chunk_masks_path / f'{seq_idx:05d}{mask_src.suffix}'
                    _link_or_copy(frame_src, frame_dest)
                    _link_or_copy(mask_src, mask_dest)
                cmd = _build_command(str(chunk_frames_path), str(chunk_masks_path), str(chunk_output_path))
                env = os.environ.copy()
                if device_obj.type == 'cuda':
                    cuda_idx = device_obj.index if device_obj.index is not None else 0
                    env['CUDA_VISIBLE_DEVICES'] = str(cuda_idx)
                else:
                    env['CUDA_VISIBLE_DEVICES'] = ''
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if result.returncode != 0:
                    print(f'E2FGVI stdout (chunk {chunk.index}): {result.stdout}')
                    print(f'E2FGVI stderr (chunk {chunk.index}): {result.stderr}')
                    raise RuntimeError(f'E2FGVI inference failed for chunk {chunk.index}: {result.stderr}')
                output_files = sorted([p for p in chunk_output_path.iterdir() if p.suffix.lower() in valid_suffixes], key=lambda p: p.name)
                if len(output_files) != len(chunk_indices):
                    raise RuntimeError(f'Mismatch between produced frames ({len(output_files)}) and expected count ({len(chunk_indices)}) for E2FGVI chunk {chunk.index}.')
                for (rel_idx, original_idx) in enumerate(chunk_indices):
                    if original_idx < chunk.core_start or original_idx >= chunk.core_end:
                        continue
                    output_file = output_files[rel_idx]
                    final_path = output_path / frame_paths[original_idx].name
                    if final_path.exists():
                        final_path.unlink()
                    shutil.copy2(output_file, final_path)
        max_workers = len(device_list)
        if max_workers <= 0:
            raise RuntimeError('No devices available for E2FGVI parallel execution.')
        chunk_iter = iter(chunks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                futures = []
                for device_obj in device_list:
                    chunk = next(chunk_iter, None)
                    if chunk is None:
                        break
                    futures.append(executor.submit(_process_chunk, chunk, device_obj))
                if not futures:
                    break
                for future in futures:
                    future.result()
        missing = [path.name for path in frame_paths if not (output_path / path.name).exists()]
        if missing:
            raise RuntimeError(f"E2FGVI parallel execution missing {len(missing)} frame(s); examples: {', '.join(missing[:5])}")
        print(f'E2FGVI inpainted frames saved to {output_frames_abs}')
    if len(cuda_devices) >= 2 and total_frames > 1:
        _run_parallel(cuda_devices)
    else:
        _run_single(preferred_device)
def upscale_realesrgan_2x(image: np.ndarray, realesrgan_dir: str=None, temp_dir: str=None) -> np.ndarray:
    """Apply Real-ESRGAN 2x upscaling. Returns upscaled image (2*H, 2*W, C) in BGR format."""
    cleanup_temp = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        cleanup_temp = True
    original_dir = os.getcwd()
    try:
        temp_dir_abs = os.path.abspath(temp_dir)
        input_dir = os.path.join(temp_dir_abs, 'input')
        output_dir = os.path.join(temp_dir_abs, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        input_path = os.path.join(input_dir, 'input.png')
        cv2.imwrite(input_path, image)
        run_args = None
        use_package = False
        if realesrgan_dir is None:
            try:
                import importlib
                importlib.import_module('realesrgan.entrypoints')
                use_package = True
            except ImportError:
                use_package = False
        if use_package:
            run_args = [sys.executable, '-m', 'realesrgan.entrypoints', '-n', 'RealESRGAN_x4plus', '-i', input_dir, '-o', output_dir, '-s', '2', '--suffix', 'out', '--ext', 'png']
        else:
            if realesrgan_dir is None:
                raise RuntimeError("The 'realesrgan' package is not installed and no local realesrgan_dir was provided.")
            realesrgan_dir_abs = os.path.abspath(realesrgan_dir)
            inference_script = os.path.join(realesrgan_dir_abs, 'inference_realesrgan.py')
            if not os.path.exists(inference_script):
                raise FileNotFoundError(f'Real-ESRGAN inference script not found at: {inference_script}')
            os.chdir(realesrgan_dir_abs)
            run_args = [sys.executable, inference_script, '-n', 'RealESRGAN_x4plus', '-i', input_dir, '-o', output_dir, '-s', '2', '--suffix', 'out', '--ext', 'png']
        result = subprocess.run(run_args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'Real-ESRGAN failed: {result.stderr}\nStdout: {result.stdout}')
        output_path = os.path.join(output_dir, 'input_out.png')
        if not os.path.exists(output_path):
            raise RuntimeError(f'Failed to find upscaled image at {output_path}')
        upscaled = cv2.imread(output_path)
        if upscaled is None:
            raise RuntimeError(f'Failed to read upscaled image from {output_path}')
        return upscaled
    finally:
        os.chdir(original_dir)
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
def _instantiate_realesrgan_upsampler(model_name: str, device: torch.device, *, denoise_strength: float=1.0, tile: int=0, tile_pad: int=10, pre_pad: int=0, fp32: bool=False) -> 'RealESRGANer':
    """Create and warm a Real-ESRGAN upsampler on the specified device."""
    try:
        import realesrgan
        from realesrgan.utils import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        from realesrgan.inference import DEFAULT_RELEASE_SUBDIR, DEFAULT_WEIGHTS_SUBDIR, _resolve_existing_model_path
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
    except ImportError as exc:
        raise RuntimeError('Real-ESRGAN python package with its dependencies is required. Install it with `pip install realesrgan basicsr`.') from exc
    model_name = model_name.split('.')[0]
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth']
    else:
        raise ValueError(f"Unsupported Real-ESRGAN model '{model_name}'.")
    package_dir = Path(realesrgan.__file__).resolve().parent
    release_dir = package_dir / DEFAULT_RELEASE_SUBDIR
    weights_dir = package_dir / DEFAULT_WEIGHTS_SUBDIR
    cwd_weights_dir = Path.cwd() / DEFAULT_WEIGHTS_SUBDIR
    search_dirs: List[Path] = [release_dir, weights_dir, cwd_weights_dir]
    existing = _resolve_existing_model_path(model_name, search_dirs)
    if existing is None:
        weights_dir.mkdir(parents=True, exist_ok=True)
        root_dir = package_dir
        for url in file_urls:
            load_file_from_url(url=url, model_dir=os.path.join(root_dir, DEFAULT_WEIGHTS_SUBDIR), progress=True, file_name=None)
        existing = _resolve_existing_model_path(model_name, search_dirs)
        if existing is None:
            raise RuntimeError(f"Unable to locate Real-ESRGAN weights for model '{model_name}'.")
        if model_name == 'realesr-general-x4v3':
            wdn_existing = _resolve_existing_model_path('realesr-general-wdn-x4v3', search_dirs)
            if wdn_existing is None:
                raise RuntimeError('Missing realesr-general-wdn-x4v3 weights required for DNI mode.')
    resolved_model_path: Union[str, List[str]] = str(existing)
    dni_weight: Optional[List[float]] = None
    if model_name == 'realesr-general-x4v3' and (not math.isclose(denoise_strength, 1.0)):
        wdn_path = _resolve_existing_model_path('realesr-general-wdn-x4v3', search_dirs)
        if wdn_path is None:
            raise RuntimeError('Unable to locate realesr-general-wdn-x4v3 weights for DNI upsampling.')
        resolved_model_path = [resolved_model_path, str(wdn_path)]
        dni_weight = [denoise_strength, 1 - denoise_strength]
    half_precision = device.type == 'cuda' and (not fp32)
    upsampler = RealESRGANer(scale=netscale, model_path=resolved_model_path, dni_weight=dni_weight, model=model, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, half=half_precision, device=device)
    return upsampler
def _device_slot_key(device_obj: torch.device, slot_id: int) -> str:
    idx = device_obj.index if device_obj.type == 'cuda' and device_obj.index is not None else None
    base = f'cuda:{idx}' if idx is not None else str(device_obj)
    return base if slot_id <= 0 else f'{base}#{slot_id}'
def _format_device_slot(device_obj: torch.device, slot_id: int) -> str:
    key = _device_slot_key(device_obj, slot_id)
    return key
def _upsample_with_realesrgan(upsampler: 'RealESRGANer', image: np.ndarray, *, device_obj: Optional[torch.device]=None, outscale: float=2.0) -> np.ndarray:
    try:
        (output, _) = upsampler.enhance(image, outscale=outscale)
        return output
    except RuntimeError as exc:
        device_label = str(device_obj) if device_obj is not None else 'unknown device'
        raise RuntimeError(f'Real-ESRGAN failed on {device_label}: {exc}') from exc
def upscale_realesrgan_adaptive(downsampled_image: np.ndarray, downscale_maps: np.ndarray, block_size: int, realesrgan_dir: str=None, *, upsample_fn: Optional[Callable[[np.ndarray], np.ndarray]]=None) -> np.ndarray:
    """
    Applies adaptive Real-ESRGAN upscaling to restore an image where different blocks
    were downsampled by different factors (powers of 2).

    The algorithm works in multiple stages:
    1. Find max downscaling factor and downscale image to that resolution.
    2. Apply Real-ESRGAN 2x to entire frame and update maps.
    3. Restore blocks that were originally downsampled by a factor smaller or equal to the current stage.
    4. Repeat for next stage until full resolution is reached and all blocks are restored.
    
    This allows blocks to see their neighbors during upscaling for proper context,
    while avoiding applying unnecessary upscaling artifacts to higher-quality blocks.
    
    Args:
        downsampled_image: The downsampled image (non-uniform block sizes) in BGR format
        downscale_maps: 2D array (num_blocks_y, num_blocks_x) indicating the downscale factor applied to each block.
        block_size: The side length of each block in the original resolution
        realesrgan_dir: Path to Real-ESRGAN directory
        upsample_fn: Optional callable performing a single 2x upscale. Defaults to calling upscale_realesrgan_2x.
    
    Returns:
        The adaptively upscaled image at original resolution
    """
    if upsample_fn is None:
        upsample_fn = functools.partial(upscale_realesrgan_2x, realesrgan_dir=realesrgan_dir)
    downsampled_image, downscale_maps, pad_y, pad_x = _pad_for_restoration(downsampled_image, downscale_maps, block_size)
    downscale_maps = np.power(2, downscale_maps).astype(np.int32)
    max_factor = int(downscale_maps.max())
    (height, width, _) = downsampled_image.shape
    current_image = cv2.resize(downsampled_image, (width // max_factor, height // max_factor), interpolation=cv2.INTER_AREA)
    (num_blocks_y, num_blocks_x) = downscale_maps.shape
    current_factor = max_factor / 2
    while current_factor >= 1:
        current_block_size = block_size // int(current_factor)
        current_image = upsample_fn(current_image)
        blocks = split_image_into_blocks(current_image, current_block_size)
        downscaled_image = cv2.resize(downsampled_image, (current_image.shape[1], current_image.shape[0]), interpolation=cv2.INTER_AREA)
        downsampled_blocks = split_image_into_blocks(downscaled_image, current_block_size)
        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                block_factor = downscale_maps[i, j]
                if block_factor <= current_factor:
                    blocks[i, j] = downsampled_blocks[i, j]
                else:
                    downscale_maps[i, j] = current_factor
        current_image = combine_blocks_into_image(blocks)
        current_factor /= 2
    return _crop_after_restoration(current_image, pad_y, pad_x)
import threading
_REALESRGAN_UPSAMPLER_LOCK = threading.Lock()

def get_realesrgan_upsampler(device: torch.device, *, model_name: str='RealESRGAN_x4plus', denoise_strength: float=1.0, tile: int=0, tile_pad: int=10, pre_pad: int=0, fp32: bool=False) -> 'RealESRGANer':
    """Get or create a cached RealESRGAN upsampler for the given device."""
    key = f'{device}_{model_name}_{denoise_strength}_{tile}_{tile_pad}_{pre_pad}_{fp32}'
    with _REALESRGAN_UPSAMPLER_LOCK:
        upsampler = _REALESRGAN_UPSAMPLER_CACHE.get(key)
        if upsampler is None:
            _safe_print(f'    -> Warming Real-ESRGAN runtime on {device}...')
            upsampler = _instantiate_realesrgan_upsampler(model_name=model_name, device=device, denoise_strength=denoise_strength, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, fp32=fp32)
            _REALESRGAN_UPSAMPLER_CACHE[key] = upsampler
    return upsampler
def restore_frames_realesrgan(frames: List[np.ndarray], downscale_maps: np.ndarray, block_size: int, device: torch.device, *, model_name: str='RealESRGAN_x4plus', denoise_strength: float=1.0, tile: int=0, tile_pad: int=10, pre_pad: int=0, fp32: bool=False) -> List[np.ndarray]:
    """
    Pure restoration function: restore frames using RealESRGAN.
    
    Takes frames and downscale maps as numpy arrays, returns restored frames.
    No file IO, no parallelization - just core restoration logic.
    """
    upsampler = get_realesrgan_upsampler(device, model_name=model_name, denoise_strength=denoise_strength, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, fp32=fp32)

    def _enhance_once(img: np.ndarray) -> np.ndarray:
        return _upsample_with_realesrgan(upsampler, img, device_obj=device, outscale=2.0)
    restored_frames = []
    for (idx, frame) in enumerate(frames):
        restored = upscale_realesrgan_adaptive(frame, downscale_maps[idx], block_size, upsample_fn=_enhance_once)
        restored_frames.append(restored)
    return restored_frames
def restore_downsampled_with_realesrgan(input_frames_dir: str, output_frames_dir: str, downscale_maps: np.ndarray, block_size: int, *, model_name: str='RealESRGAN_x4plus', denoise_strength: float=1.0, tile: int=0, tile_pad: int=10, pre_pad: int=0, fp32: bool=False, devices: Optional[Sequence[Union[int, str, torch.device]]]=None, parallel_chunk_length: Optional[int]=None, per_device_workers: int=1) -> None:
    """Parallel adaptive Real-ESRGAN restoration over a directory of frames."""
    frame_paths = get_frame_paths(input_frames_dir)
    if not frame_paths:
        raise ValueError(f'No frames found in {input_frames_dir}')
    downscale_maps = np.asarray(downscale_maps)
    num_frames = len(frame_paths)
    if downscale_maps.shape[0] != num_frames:
        raise ValueError(f'Downscale maps length ({downscale_maps.shape[0]}) does not match frame count ({num_frames}).')
    clear_directory(output_frames_dir)
    os.makedirs(output_frames_dir, exist_ok=True)
    resolved_devices = _resolve_device_list(devices, prefer_cuda=True, allow_cpu_fallback=True)
    device_summary = ', '.join((str(dev) for dev in resolved_devices))
    tile_desc = str(tile) if tile and tile > 0 else 'full-frame'
    _safe_print(f'  Using Real-ESRGAN on devices: {device_summary} | tile: {tile_desc}')
    _safe_print(f'  Total frames: {num_frames}')

    def process_chunk(chunk_frames: List[np.ndarray], device: torch.device) -> List[np.ndarray]:
        return restore_frames_realesrgan(chunk_frames, downscale_maps[:len(chunk_frames)], block_size, device, model_name=model_name, denoise_strength=denoise_strength, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, fp32=fp32)
    chunks = chunk_for_devices(num_frames, resolved_devices)
    all_restored: List[np.ndarray] = []
    for chunk in chunks:
        chunk_frames = [load_frame(str(frame_paths[i])) for i in range(chunk.start, chunk.end)]
        chunk_maps = downscale_maps[chunk.start:chunk.end]
        _safe_print(f'    -> Real-ESRGAN frames {chunk.start + 1}-{chunk.end} on {chunk.device}')
        restored = restore_frames_realesrgan(chunk_frames, chunk_maps, block_size, chunk.device, model_name=model_name, denoise_strength=denoise_strength, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, fp32=fp32)
        all_restored.extend(restored)
    for (idx, restored_frame) in enumerate(all_restored):
        output_path = os.path.join(output_frames_dir, frame_paths[idx].name)
        save_frame(restored_frame, output_path)
def _pad_for_restoration(image: np.ndarray, maps: np.ndarray, block_size: int):
    (h_orig, w_orig, _) = image.shape
    pad_y = (block_size - h_orig % block_size) % block_size
    pad_x = (block_size - w_orig % block_size) % block_size
    if pad_y > 0 or pad_x > 0:
        image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        target_h = image.shape[0] // block_size
        target_w = image.shape[1] // block_size
        pad_h = target_h - maps.shape[0]
        pad_w = target_w - maps.shape[1]
        if pad_h > 0 or pad_w > 0:
            maps = np.pad(maps, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode='constant', constant_values=0)
    return image, maps, pad_y, pad_x

def _crop_after_restoration(image: np.ndarray, pad_y: int, pad_x: int):
    if pad_y > 0:
        image = image[:-pad_y, :, :]
    if pad_x > 0:
        image = image[:, :-pad_x, :]
    return image

def restore_downsample_opencv_lanczos(downsampled_image: np.ndarray, downscale_maps: np.ndarray, block_size: int) -> np.ndarray:
    """
    Restores a downsampled image using OpenCV's Lanczos interpolation.
    This is a simple client-side restoration benchmark that doesn't require any ML models.
    Lanczos generally provides better quality than bilinear/bicubic for upscaling.
    
    Args:
        downsampled_image: The downsampled image (non-uniform block sizes) in BGR format
        downscale_maps: 2D array (num_blocks_y, num_blocks_x) indicating the downscale factor applied to each block
        block_size: The side length of each block in the original resolution
    
    Returns:
        The restored image at original resolution using Lanczos upscaling
    """
    downsampled_image, downscale_maps, pad_y, pad_x = _pad_for_restoration(downsampled_image, downscale_maps, block_size)
    downscale_factors = np.power(2, downscale_maps).astype(np.int32)
    max_factor = int(downscale_factors.max())
    if max_factor == 1:
        return _crop_after_restoration(downsampled_image, pad_y, pad_x)
    (height, width, _) = downsampled_image.shape
    (num_blocks_y, num_blocks_x) = downscale_maps.shape
    blocks = split_image_into_blocks(downsampled_image, block_size)
    restored_blocks = np.zeros_like(blocks)
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            factor = downscale_factors[i, j]
            if factor > 1:
                block = blocks[i, j]
                small_size = max(1, block_size // factor)
                small_block = cv2.resize(block, (small_size, small_size), interpolation=cv2.INTER_AREA)
                restored_block = cv2.resize(small_block, (block_size, block_size), interpolation=cv2.INTER_LANCZOS4)
                restored_blocks[i, j] = restored_block
            else:
                restored_blocks[i, j] = blocks[i, j]
    restored_image = combine_blocks_into_image(restored_blocks)
    return _crop_after_restoration(restored_image, pad_y, pad_x)
def restore_blur_opencv_unsharp_mask(blurred_image: np.ndarray, blur_maps: np.ndarray, block_size: int) -> np.ndarray:
    """
    Restores a blurred image using OpenCV's unsharp masking technique.
    This is a simple client-side restoration benchmark that doesn't require any ML models.
    
    Unsharp masking works by:
    1. Blurring the image
    2. Subtracting the blurred version from the original to get high-frequency details
    3. Adding these details back to enhance sharpness
    
    Args:
        blurred_image: The blurred image in BGR format
        blur_maps: 2D array (num_blocks_y, num_blocks_x) indicating blur rounds applied to each block
        block_size: The side length of each block
    
    Returns:
        The restored image with adaptive unsharp masking applied
    """
    blurred_image, blur_maps, pad_y, pad_x = _pad_for_restoration(blurred_image, blur_maps, block_size)
    max_rounds = int(blur_maps.max())
    if max_rounds == 0:
        return _crop_after_restoration(blurred_image, pad_y, pad_x)
    (height, width, _) = blurred_image.shape
    (num_blocks_y, num_blocks_x) = blur_maps.shape
    blocks = split_image_into_blocks(blurred_image, block_size)
    restored_blocks = np.zeros_like(blocks)
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            block = blocks[i, j]
            blur_strength = int(blur_maps[i, j])
            if blur_strength > 0:
                amount = blur_strength * 0.5
                radius = max(1, blur_strength)
                blurred = cv2.GaussianBlur(block, (0, 0), radius)
                sharpened = cv2.addWeighted(block, 1.0 + amount, blurred, -amount, 0)
                restored_blocks[i, j] = np.clip(sharpened, 0, 255).astype(np.uint8)
            else:
                restored_blocks[i, j] = block
    restored_image = combine_blocks_into_image(restored_blocks)
    return _crop_after_restoration(restored_image, pad_y, pad_x)
def _instantir_chunk_worker(frames_dir: str, frame_names: Sequence[str], blur_maps: np.ndarray, block_size: int, weights_dir: str, cfg: float, creative_start: float, preview_start: float, batch_size: int, device_str: str, seed: Optional[int], chunk_index: int, total_chunks: int, global_start: int, global_end: int) -> None:
    """Worker entry point that restores a contiguous frame chunk on a single device."""
    device = torch.device(device_str)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    weights_path = Path(weights_dir).expanduser()
    frame_paths = [os.path.join(frames_dir, name) for name in frame_names]
    chunk_frames: List[np.ndarray] = []
    original_blocks: List[np.ndarray] = []
    if not frame_paths: return
    first_frame = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    _, dummy_map, pad_y, pad_x = _pad_for_restoration(first_frame, blur_maps[0], block_size)
    target_h = (first_frame.shape[0] + pad_y) // block_size
    target_w = (first_frame.shape[1] + pad_x) // block_size
    pad_h = target_h - blur_maps.shape[1]
    pad_w = target_w - blur_maps.shape[2]
    if pad_h > 0 or pad_w > 0:
        blur_maps = np.pad(blur_maps, ((0, 0), (0, max(0, pad_h)), (0, max(0, pad_w))), mode='constant', constant_values=0)
    for path in frame_paths:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f'Failed to load frame for InstantIR restoration: {path}')
        if pad_y > 0 or pad_x > 0:
            frame = np.pad(frame, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        chunk_frames.append(frame)
        original_blocks.append(split_image_into_blocks(frame, block_size))
    chunk_original_blocks = np.stack(original_blocks, axis=0)
    chunk_blur_maps = np.asarray(blur_maps, dtype=np.int32).copy()
    local_length = len(chunk_frames)
    max_rounds = int(chunk_blur_maps.max()) if chunk_blur_maps.size else 0
    device_label = device_str
    _safe_print(f'    -> InstantIR chunk {chunk_index + 1}/{total_chunks} frames {global_start + 1}-{global_end} on {device_label} (max rounds: {max_rounds})')
    runtime: Optional[InstantIRRuntime] = None
    try:
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        with _silence_console_output():
            runtime = load_runtime(instantir_path=weights_path, device=device, torch_dtype=dtype, map_location='cpu')
        if hasattr(runtime, 'pipe') and hasattr(runtime.pipe, 'set_progress_bar_config'):
            runtime.pipe.set_progress_bar_config(disable=True)
        if max_rounds <= 0:
            _safe_print(f'       No blur detected for chunk {chunk_index + 1}; skipping restoration.')
        else:

            def _iter_batches(indices: Sequence[int], batch_len: int) -> Iterator[List[int]]:
                step = max(1, batch_len)
                for offset in range(0, len(indices), step):
                    yield list(indices[offset:offset + step])
            for round_idx in range(max_rounds):
                active_indices = [idx for idx in range(local_length) if np.any(chunk_blur_maps[idx] > 0)]
                if not active_indices:
                    break
                _safe_print(f'       Round {round_idx + 1}/{max_rounds}: processing {len(active_indices)} frame(s)')
                for batch_indices in _iter_batches(active_indices, batch_size):
                    pil_batch = []
                    for local_idx in batch_indices:
                        frame_rgb = cv2.cvtColor(chunk_frames[local_idx], cv2.COLOR_BGR2RGB)
                        pil_batch.append(Image.fromarray(frame_rgb))
                    with _silence_console_output():
                        restored_pils = restore_images_batch(runtime, pil_batch, num_inference_steps=1, cfg=cfg, preview_start=preview_start, creative_start=creative_start)
                    for (local_idx, restored_pil) in zip(batch_indices, restored_pils):
                        restored_bgr = cv2.cvtColor(np.array(restored_pil), cv2.COLOR_RGB2BGR)
                        restored_blocks = split_image_into_blocks(restored_bgr, block_size)
                        completed_mask = chunk_blur_maps[local_idx] <= 0
                        if np.any(completed_mask):
                            restored_blocks[completed_mask] = chunk_original_blocks[local_idx][completed_mask]
                        chunk_frames[local_idx] = combine_blocks_into_image(restored_blocks)
                positive_mask = chunk_blur_maps > 0
                chunk_blur_maps[positive_mask] -= 1
        for (path, frame) in zip(frame_paths, chunk_frames):
            frame = _crop_after_restoration(frame, pad_y, pad_x)
            if not cv2.imwrite(path, frame):
                raise RuntimeError(f'Failed to write restored frame: {path}')
    finally:
        if runtime is not None:
            del runtime
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        gc.collect()
def restore_with_instantir_adaptive(input_frames_dir: str, blur_maps: np.ndarray, block_size: int, cfg: float=7.0, creative_start: float=1.0, preview_start: float=0.0, seed: Optional[int]=42, devices: Optional[Sequence[Union[int, str, torch.device]]]=None, batch_size: int=4, parallel_chunk_length: Optional[int]=None) -> None:
    """Apply adaptive InstantIR blind restoration with simple per-device chunking."""
    _ = parallel_chunk_length
    if batch_size < 1:
        raise ValueError('`batch_size` must be at least 1.')
    _safe_print('  Preparing InstantIR workers...')
    weights_dir = Path('./InstantIR/models').expanduser()
    weights_dir.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    resolved_devices = _resolve_device_list(devices, prefer_cuda=True, allow_cpu_fallback=True)
    cuda_devices = [dev for dev in resolved_devices if dev.type == 'cuda']
    worker_devices = cuda_devices if cuda_devices else [resolved_devices[0]]
    frames_files = sorted([f for f in os.listdir(input_frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    num_frames = len(frames_files)
    if num_frames == 0:
        raise ValueError(f'No frames found in {input_frames_dir}')
    if num_frames != blur_maps.shape[0]:
        raise ValueError(f"Number of frames ({num_frames}) doesn't match blur_maps shape ({blur_maps.shape[0]})")
    if np.max(blur_maps) == 0:
        _safe_print('  No blurring detected, skipping restoration.')
        return

    def _split_ranges(total: int, parts: int) -> List[Tuple[int, int]]:
        if parts <= 0:
            return [(0, total)]
        base = total // parts
        remainder = total % parts
        ranges: List[Tuple[int, int]] = []
        start = 0
        for idx in range(parts):
            length = base + (1 if idx < remainder else 0)
            end = start + length
            ranges.append((start, end))
            start = end
        return ranges
    initial_ranges = _split_ranges(num_frames, len(worker_devices))
    jobs: List[Dict[str, Any]] = []
    for (device, (start, end)) in zip(worker_devices, initial_ranges):
        if start >= end:
            continue
        frames_subset = frames_files[start:end]
        jobs.append({'device': device, 'start': start, 'end': end, 'frames': frames_subset, 'blur': np.array(blur_maps[start:end], copy=True)})
    if not jobs:
        _safe_print('  No frame chunks were assigned; skipping InstantIR restoration.')
        return
    for (idx, job) in enumerate(jobs):
        device = job['device']
        if device.type == 'cuda':
            dev_idx = device.index if device.index is not None else 0
            job['device_str'] = f'cuda:{dev_idx}'
        else:
            job['device_str'] = str(device)
        job['chunk_index'] = idx
    device_summary = ', '.join((job['device_str'] for job in jobs))
    _safe_print(f'  Using InstantIR on devices: {device_summary} | batch size per device: {batch_size}')
    total_chunks = len(jobs)
    chunk_shapes = ', '.join((f"{job['start'] + 1}-{job['end']} ({job['end'] - job['start']} frames)" for job in jobs))
    _safe_print(f'  Assigned frame spans per worker: {chunk_shapes}')
    if total_chunks == 1:
        job = jobs[0]
        worker_seed = seed
        _instantir_chunk_worker(input_frames_dir, job['frames'], job['blur'], block_size, str(weights_dir), cfg, creative_start, preview_start, batch_size, job['device_str'], worker_seed, job['chunk_index'], total_chunks, job['start'], job['end'])
    else:
        ctx = multiprocessing.get_context('spawn')
        processes: List[multiprocessing.Process] = []
        for job in jobs:
            worker_seed = seed + job['chunk_index'] if seed is not None else None
            proc = ctx.Process(target=_instantir_chunk_worker, args=(input_frames_dir, job['frames'], job['blur'], block_size, str(weights_dir), cfg, creative_start, preview_start, batch_size, job['device_str'], worker_seed, job['chunk_index'], total_chunks, job['start'], job['end']))
            proc.start()
            processes.append(proc)
        errors: List[int] = []
        for proc in processes:
            proc.join()
            if proc.exitcode not in (0, None):
                errors.append(proc.exitcode)
        if errors:
            raise RuntimeError(f'InstantIR worker(s) exited with non-zero code(s): {errors}')
    _safe_print(f'  Adaptive InstantIR restoration complete. Frames saved to {input_frames_dir}')
