import platform
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

def _decode_video_to_frames(video_path: str, max_frames: Optional[int]=None) -> List[np.ndarray]:
    """Decode a video into a list of BGR frames using OpenCV."""
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Warning: Unable to open video for decoding: {video_path}')
        return frames
    total = 0
    while True:
        (ret, frame) = cap.read()
        if not ret:
            break
        frames.append(frame)
        total += 1
        if max_frames is not None and total >= max_frames:
            break
    cap.release()
    return frames
def _encode_frames_to_video(frames: Sequence[np.ndarray], output_path: str, framerate: float, filter_chain: Optional[str]=None, codec: str='libx264', preset: str='ultrafast', pix_fmt: str='yuv420p', extra_codec_args: Optional[Sequence[str]]=None) -> None:
    """Encode a sequence of frames to video via FFmpeg piping."""
    if not frames:
        raise ValueError('No frames provided for video encoding.')
    (height, width) = frames[0].shape[:2]
    pix_fmt_lower = (pix_fmt or '').lower()
    requires_even_dims = False
    if pix_fmt_lower:
        chroma_tokens = ('420', 'nv12', 'nv21', 'p010', 'p016')
        if any((token in pix_fmt_lower for token in chroma_tokens)):
            requires_even_dims = True
    else:
        requires_even_dims = True
    pad_right = 1 if requires_even_dims and width % 2 != 0 else 0
    pad_bottom = 1 if requires_even_dims and height % 2 != 0 else 0
    even_pad_filter: Optional[str] = None
    if requires_even_dims:
        even_pad_filter = 'pad=ceil(iw/2)*2:ceil(ih/2)*2:x=0:y=0:color=black'
        if filter_chain:
            filter_chain = f'{filter_chain},{even_pad_filter}'
        else:
            filter_chain = even_pad_filter
    adjusted_width = width + pad_right
    adjusted_height = height + pad_bottom
    if pad_right or pad_bottom:
        print(f"  - Adjusting frame dimensions from {width}x{height} to {adjusted_width}x{adjusted_height} for {pix_fmt or 'default'} encoding")
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{adjusted_width}x{adjusted_height}', '-r', f'{framerate}', '-i', '-']
    if filter_chain:
        cmd.extend(['-vf', filter_chain])
    cmd.extend(['-c:v', codec, '-preset', preset])
    if codec == 'libx264' and '-crf' not in (extra_codec_args or []):
        cmd.extend(['-crf', '0'])
    if pix_fmt:
        cmd.extend(['-pix_fmt', pix_fmt])
    if extra_codec_args:
        cmd.extend(extra_codec_args)
    cmd.append(output_path)
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            if frame is None:
                continue
            frame_to_write = frame
            if pad_right or pad_bottom:
                frame_to_write = cv2.copyMakeBorder(frame_to_write, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_REPLICATE)
            frame_bytes = np.ascontiguousarray(frame_to_write.astype(np.uint8)).tobytes()
            process.stdin.write(frame_bytes)
    finally:
        if process.stdin:
            process.stdin.close()
    ret_code = process.wait()
    if ret_code != 0:
        raise RuntimeError(f'FFmpeg failed with exit code {ret_code} while encoding {output_path}')
def calculate_target_bitrate(width: int, height: int, framerate: float, quality_factor: float=1.0) -> int:
    """Calculate target bitrate based on video characteristics. Returns bitrate in bps."""
    pixels_per_second = width * height * framerate
    bits_per_pixel = 0.01 * quality_factor
    target_bps = int(pixels_per_second * bits_per_pixel)
    return target_bps
def encode_video(input_frames_dir: str, output_video: str, framerate: float, width: int, height: int, target_bitrate: int=None, preset: str='medium', pix_fmt: str='yuv420p', **extra_params) -> None:
    """Encode video using two-pass libx265. Supports lossy (with target_bitrate) or lossless mode."""
    temp_dir = os.path.dirname(output_video) or '.'
    os.makedirs(temp_dir, exist_ok=True)
    passlog_file = os.path.join(temp_dir, f'ffmpeg_2pass_log_{os.path.basename(output_video)}')
    null_device = 'NUL' if platform.system() == 'Windows' else '/dev/null'
    try:
        extra_params = {key: value for (key, value) in extra_params.items() if value is not None}
        pass1_extra_params = {key: value for (key, value) in extra_params.items() if key != 'qpfile'}

        def _extend_x265_params(base: str, params: Dict[str, Any]) -> str:
            if not params:
                return base
            suffix = ''.join((f':{key}={value}' for (key, value) in params.items()))
            return f'{base}{suffix}'
        base_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-framerate', str(framerate), '-i', f'{input_frames_dir}/%05d.png', '-vf', f'scale={width}:{height}:flags=lanczos,format={pix_fmt}']
        if target_bitrate is None:
            preset = 'fast'
            x265_base_params = 'lossless=1'
            pass1_params = _extend_x265_params(f'{x265_base_params}:pass=1:stats={passlog_file}', pass1_extra_params)
            pass1_cmd = base_cmd + ['-c:v', 'libx265', '-preset', preset, '-x265-params', pass1_params, '-f', 'mp4', '-y', null_device]
            subprocess.run(pass1_cmd, check=True, capture_output=True, text=True)
            pass2_params = _extend_x265_params(f'{x265_base_params}:pass=2:stats={passlog_file}', extra_params)
            pass2_cmd = base_cmd + ['-c:v', 'libx265', '-preset', preset, '-x265-params', pass2_params, '-y', output_video]
            result = subprocess.run(pass2_cmd, check=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f'Error in pass 2 encoding: {result.stderr}')
        else:
            pass1_params = _extend_x265_params(f'pass=1:stats={passlog_file}', pass1_extra_params)
            pass1_cmd = base_cmd + ['-c:v', 'libx265', '-b:v', str(target_bitrate), '-minrate', str(int(target_bitrate * 0.9)), '-maxrate', str(int(target_bitrate * 1.1)), '-bufsize', str(target_bitrate), '-preset', preset, '-g', str(framerate), '-x265-params', pass1_params, '-f', 'mp4', '-y', null_device]
            subprocess.run(pass1_cmd, check=True, capture_output=True, text=True)
            pass2_params = _extend_x265_params(f'pass=2:stats={passlog_file}', extra_params)
            pass2_cmd = base_cmd + ['-c:v', 'libx265', '-b:v', str(target_bitrate), '-minrate', str(int(target_bitrate * 0.9)), '-maxrate', str(int(target_bitrate * 1.1)), '-bufsize', str(target_bitrate), '-preset', preset, '-g', str(framerate), '-x265-params', pass2_params, '-y', output_video]
            result = subprocess.run(pass2_cmd, check=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f'Error in pass 2 encoding: {result.stderr}')
    except subprocess.CalledProcessError as e:
        print('--- FFMPEG COMMAND FAILED ---')
        print('STDOUT:', e.stdout)
        print('STDERR:', e.stderr)
        raise RuntimeError(f'FFmpeg command failed with exit code {e.returncode}') from e
    finally:
        import glob
        log_pattern = os.path.join(temp_dir, f'ffmpeg_2pass_log_{os.path.basename(output_video)}*')
        for f in glob.glob(log_pattern):
            try:
                os.remove(f)
            except:
                pass
def decode_video(video_path: str, output_dir: str, framerate: float=None, start_number: int=1, quality: int=1) -> bool:
    """Decode video to PNG frames. Returns True on success."""
    os.makedirs(output_dir, exist_ok=True)
    decode_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video_path, '-pix_fmt', 'rgb24', '-q:v', str(quality)]
    if framerate is not None:
        decode_cmd.extend(['-r', str(framerate)])
    decode_cmd.extend(['-f', 'image2', '-start_number', str(start_number), '-y', os.path.join(output_dir, '%05d.png')])
    result = subprocess.run(decode_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'Error decoding {video_path}: {result.stderr}')
        return False
    return True
def encode_with_roi(input_frames_dir: str, output_video: str, removability_scores: np.ndarray, block_size: int, framerate: float, width: int, height: int, target_bitrate: int=1000000, save_qp_maps: bool=False, qp_maps_dir: str=None) -> None:
    """Encode video with per-block QP control based on removability scores, using two-pass encoding."""
    (num_frames, num_blocks_y, num_blocks_x) = removability_scores.shape
    temp_dir = os.path.dirname(output_video) or '.'
    os.makedirs(temp_dir, exist_ok=True)
    qpfile_path = os.path.join(temp_dir, 'qpfile_per_block.txt')
    passlog_file = os.path.join(temp_dir, 'ffmpeg_2pass_log')
    null_device = 'NUL' if platform.system() == 'Windows' else '/dev/null'
    try:
        print('Generating detailed qpfile for per-block quality control...')
        qp_maps = np.clip(removability_scores * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)
        valid_ctu_sizes = [16, 32, 64]
        largest_dimension = max(width, height)
        min_ctu_by_resolution = 16
        if largest_dimension >= 4320:
            min_ctu_by_resolution = 64
        elif largest_dimension >= 2160:
            min_ctu_by_resolution = 32
        nearest_ctu = min(valid_ctu_sizes, key=lambda size: abs(size - block_size))
        if nearest_ctu < block_size:
            larger_sizes = [size for size in valid_ctu_sizes if size >= block_size]
            ctu_size = larger_sizes[0] if larger_sizes else valid_ctu_sizes[-1]
        else:
            ctu_size = nearest_ctu
        if ctu_size < min_ctu_by_resolution:
            compliant_sizes = [size for size in valid_ctu_sizes if size >= min_ctu_by_resolution]
            if compliant_sizes:
                ctu_size = compliant_sizes[0]
            else:
                ctu_size = valid_ctu_sizes[-1]
        ctu_cols = math.ceil(width / ctu_size)
        ctu_rows = math.ceil(height / ctu_size)
        qp_maps_aligned = np.empty((num_frames, ctu_rows, ctu_cols), dtype=np.float32)
        if (ctu_rows, ctu_cols) != (num_blocks_y, num_blocks_x):
            print(f'Resizing per-block QP maps from {num_blocks_y}x{num_blocks_x} blocks to CTU grid {ctu_rows}x{ctu_cols} (CTU={ctu_size}).')
        for frame_idx in range(num_frames):
            frame_map = qp_maps[frame_idx]
            if frame_map.shape == (ctu_rows, ctu_cols):
                qp_maps_aligned[frame_idx] = frame_map
                continue
            interpolation = cv2.INTER_AREA if ctu_size >= block_size else cv2.INTER_LINEAR
            qp_maps_aligned[frame_idx] = cv2.resize(frame_map, (ctu_cols, ctu_rows), interpolation=interpolation)
        with open(qpfile_path, 'w') as f:
            for frame_idx in range(num_frames):
                line_parts = [f'{frame_idx} P -1']
                qp_frame = qp_maps_aligned[frame_idx]
                block_qps = [f'{bx},{by},{qp_frame[by, bx]:.4f}' for by in range(ctu_rows) for bx in range(ctu_cols)]
                line_parts.extend(block_qps)
                f.write(' '.join(line_parts) + '\n')
        print(f'qpfile generated at {qpfile_path}')
        if save_qp_maps:
            if qp_maps_dir is None:
                qp_maps_dir = os.path.join(temp_dir, 'qp_maps')
            os.makedirs(qp_maps_dir, exist_ok=True)
            for frame_idx in range(num_frames):
                qp_map_block_res = qp_maps[frame_idx]
                qp_map_image = np.clip((qp_map_block_res + 1.0) * 127.5, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(qp_maps_dir, f'qp_map_{frame_idx:05d}.png'), qp_map_image)
            print(f'QP maps saved to {qp_maps_dir} at block resolution ({num_blocks_y}x{num_blocks_x})')
        print(f'Starting two-pass encoding with per-block QP control (CTU {ctu_size})...')
        encode_video(input_frames_dir=input_frames_dir, output_video=output_video, framerate=framerate, width=width, height=height, target_bitrate=target_bitrate, ctu=ctu_size, qpfile=qpfile_path)
        print(f'\nTwo-pass per-block encoding complete. Output saved to {output_video}')
    except subprocess.CalledProcessError as e:
        print('--- FFMPEG COMMAND FAILED ---')
        print('STDOUT:', e.stdout)
        print('STDERR:', e.stderr)
        raise RuntimeError(f'FFmpeg command failed with exit code {e.returncode}') from e
    finally:
        print('Cleaning up temporary files...')
        if os.path.exists(qpfile_path):
            os.remove(qpfile_path)
        import glob
        log_pattern = os.path.join(temp_dir, f'{os.path.basename(passlog_file)}*')
        for f in glob.glob(log_pattern):
            try:
                os.remove(f)
            except:
                pass
def encode_strength_maps(strength_maps: np.ndarray, output_video: str, framerate: float, target_bitrate: int=50000) -> None:
    """Encode strength maps as grayscale video. Maps normalized to 0-255 for encoding."""
    min_val = np.min(strength_maps)
    max_val = np.max(strength_maps)
    normalized_maps = ((strength_maps - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    frames_dir = os.path.splitext(output_video)[0]
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(normalized_maps.shape[0]):
        map_img = normalized_maps[i]
        cv2.imwrite(os.path.join(frames_dir, f'{i + 1:05d}.png'), map_img)
    encode_video(input_frames_dir=frames_dir, output_video=output_video, framerate=framerate, width=normalized_maps.shape[2], height=normalized_maps.shape[1], target_bitrate=target_bitrate, pix_fmt='gray')
def decode_strength_maps(video_path: str, block_size: int, frames_dir: str) -> np.ndarray:
    """Decode strength maps from compressed video. Returns 3D array (frames, blocks_y, blocks_x)."""
    if 'gaussian' in video_path:
        (min_val, max_val) = (0.0, 10.0)
    elif 'downsample' in video_path:
        (min_val, max_val) = (0.0, int(np.log2(block_size)))
    os.makedirs(frames_dir, exist_ok=True)
    decode_video(video_path, frames_dir, quality=1)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    strength_maps = []
    for frame_file in frame_files:
        img = cv2.imread(os.path.join(frames_dir, frame_file), cv2.IMREAD_GRAYSCALE)
        strength_map = img.astype(np.float32) / 255.0 * (max_val - min_val) + min_val
        strength_map = np.round(strength_map).astype(np.uint8)
        strength_maps.append(strength_map)
    return np.stack(strength_maps, axis=0)
def encode_strength_maps_to_npz(strength_maps: np.ndarray, output_path: str) -> None:
    """Encode strength maps as compressed .npz file. Stored as uint8 for minimal size."""
    if isinstance(strength_maps, list):
        strength_maps = np.stack(strength_maps, axis=0)
    if strength_maps.dtype != np.uint8:
        strength_maps = strength_maps.astype(np.uint8)
    np.savez_compressed(output_path, strength_maps=strength_maps)
    print(f'  Strength maps saved to {output_path} ({os.path.getsize(output_path) / 1024:.2f} KB)')
def decode_strength_maps_from_npz(npz_path: str) -> np.ndarray:
    """Decode strength maps from .npz file. Returns 3D array (frames, blocks_y, blocks_x)."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'Strength maps file not found: {npz_path}')
    data = np.load(npz_path)
    strength_maps = data['strength_maps']
    print(f'  Strength maps loaded from {npz_path} ({os.path.getsize(npz_path) / 1024:.2f} KB)')
    return strength_maps
