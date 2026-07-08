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



def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalizes a NumPy array to the range [0, 1]."""
    (min_val, max_val) = (arr.min(), arr.max())
    return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr
def calculate_removability_scores(raw_video_file: str, reference_frames_folder: str, width: int, height: int, block_size: int, alpha: float=0.5, working_dir: str='.', smoothing_beta: float=1) -> np.ndarray:
    """Compute removability scores via EVCA and UFO. Returns 3D array (frames, blocks_y, blocks_x) in [0,1]."""
    working_dir_abs = os.path.abspath(working_dir)
    maps_dir = os.path.join(working_dir_abs, 'maps')
    ufo_masks_dir = os.path.join(maps_dir, 'ufo_masks')
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(ufo_masks_dir, exist_ok=True)
    frame_count = len(os.listdir(reference_frames_folder))
    raw_video_abs = os.path.abspath(raw_video_file)
    reference_frames_abs = os.path.abspath(reference_frames_folder)
    ufo_masks_abs = os.path.abspath(ufo_masks_dir)
    evca_csv_dest = Path(maps_dir) / 'evca.csv'
    evca_tc_dest = Path(maps_dir) / 'evca_TC_blocks.csv'
    evca_sc_dest = Path(maps_dir) / 'evca_SC_blocks.csv'
    try:
        print('Running EVCA for complexity analysis...')
        try:
            import importlib
            evca_pkg = importlib.import_module('evca')
        except ImportError as exc:
            raise RuntimeError("The 'evca' package is not installed in the current environment. Install it via 'pip install evca' (or pip install -e . inside the repo) before running Elvis.") from exc
        evca_root = Path(evca_pkg.__file__).resolve().parent
        package_csv = evca_root / 'evca.csv'
        package_tc = evca_root / 'evca_TC_blocks.csv'
        package_sc = evca_root / 'evca_SC_blocks.csv'
        for path in (package_csv, package_tc, package_sc):
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass
        evca_cmd = [sys.executable, '-m', 'evca.main', '-i', raw_video_abs, '-r', f'{width}x{height}', '-b', str(block_size), '-f', str(frame_count), '-c', './evca.csv', '-bi', '1']
        result = subprocess.run(evca_cmd, capture_output=True, text=True, cwd=working_dir_abs)
        if result.returncode != 0:
            print(f'EVCA command failed: {result.stderr}')
            print(f'EVCA stdout: {result.stdout}')
            raise RuntimeError(f'EVCA execution failed: {result.stderr}')
        print('EVCA completed successfully')
        for dest in (evca_csv_dest, evca_tc_dest, evca_sc_dest):
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass
        try:
            shutil.copy2(package_tc, evca_tc_dest)
            shutil.copy2(package_sc, evca_sc_dest)
            if package_csv.exists():
                shutil.copy2(package_csv, evca_csv_dest)
        except FileNotFoundError as exc:
            raise RuntimeError('EVCA finished without producing the expected CSV outputs inside the package. Verify the installation and rerun the analysis.') from exc
        print('Running UFO for object detection (using installed package if available)...')
        ufo_dataset_root = os.path.join(working_dir_abs, 'ufo_dataset')
        ufo_image_dir = os.path.join(ufo_dataset_root, 'image')
        ufo_class_dir = os.path.join(ufo_image_dir, 'ref')
        shutil.rmtree(ufo_dataset_root, ignore_errors=True)
        os.makedirs(ufo_class_dir, exist_ok=True)
        for fname in sorted(os.listdir(reference_frames_abs)):
            src = os.path.join(reference_frames_abs, fname)
            dst = os.path.join(ufo_class_dir, fname)
            shutil.copy(src, dst)

        def _find_ufo_weights():
            candidate_names = ['model_best.pth', 'video_best.pth', 'ufo_weights.pth', 'video_weights.pth', 'weights.pth']
            try:
                import importlib
                ufo_pkg = importlib.import_module('ufo')
                pkg_weights_dir = Path(ufo_pkg.__file__).parent / 'weights'
            except Exception:
                pkg_weights_dir = Path(working_dir_abs) / 'weights'
            for n in candidate_names:
                p = pkg_weights_dir / n
                if p.exists():
                    return str(p)
            try:
                downloader = importlib.import_module('ufo.download_ufo_weights')
                if hasattr(downloader, 'main'):
                    downloaded = downloader.main()
                    if downloaded:
                        return str(downloaded)
            except Exception:
                pass
            return None
        model_path_for_ufo = _find_ufo_weights() or 'weights/video_best.pth'
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        ran_ufo = False
        try:
            from ufo.test import segment_frames
            import cv2
            import numpy as np

            frame_files = sorted(os.listdir(reference_frames_abs))
            frames_list = []
            for fname in frame_files:
                fpath = os.path.join(reference_frames_abs, fname)
                frames_list.append(cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB))
            
            frames_arr = np.array(frames_list)
            
            masks_arr = segment_frames(
                frames=frames_arr,
                device=device_str,
                model_path=model_path_for_ufo,
                group_size=5,
                img_size=224
            )
            
            os.makedirs(ufo_masks_abs, exist_ok=True)
            for i, fname in enumerate(frame_files):
                mask_path = os.path.join(ufo_masks_abs, fname)
                mask_uint8 = (masks_arr[i] * 255.0).astype(np.uint8)
                cv2.imwrite(mask_path, mask_uint8)
                
            ran_ufo = True
        except Exception as e:
            print(f'Programmatic UFO run failed or package not available: {e}')
            
        if not ran_ufo:
            raise RuntimeError('UFO execution failed: could not run programmatically.')
        for (root, dirs, files) in os.walk(ufo_masks_abs):
            for fname in files:
                fpath = os.path.join(root, fname)
                target = os.path.join(ufo_masks_abs, fname)
                if os.path.dirname(fpath) == ufo_masks_abs:
                    continue
                try:
                    shutil.move(fpath, target)
                except Exception:
                    try:
                        shutil.copy(fpath, target)
                    except Exception:
                        pass
        for (dirpath, dirnames, filenames) in os.walk(ufo_masks_abs, topdown=False):
            if dirpath == ufo_masks_abs:
                continue
            try:
                os.rmdir(dirpath)
            except Exception:
                pass
        shutil.rmtree(ufo_dataset_root, ignore_errors=True)
        temporal_array = np.loadtxt(evca_tc_dest, delimiter=',', skiprows=1)
        spatial_array = np.loadtxt(evca_sc_dest, delimiter=',', skiprows=1)
        num_blocks_x = width // block_size
        num_blocks_y = height // block_size
        num_frames = min(temporal_array.shape[1], spatial_array.shape[1])
        temporal_3d = temporal_array[:, :num_frames].T.reshape(num_frames, num_blocks_y, num_blocks_x)
        spatial_3d = spatial_array[:, :num_frames].T.reshape(num_frames, num_blocks_y, num_blocks_x)
        temporal_3d = normalize_array(temporal_3d)
        spatial_3d = normalize_array(spatial_3d)
        removability_scores = np.zeros_like(spatial_3d)
        removability_scores[:-1] = alpha * spatial_3d[:-1] + (1 - alpha) * temporal_3d[1:]
        removability_scores[-1] = spatial_3d[-1]
        for i in range(num_frames):
            mask_path = os.path.join(ufo_masks_abs, f'{i + 1:05d}.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                resized_mask = cv2.resize(mask, (num_blocks_x, num_blocks_y), interpolation=cv2.INTER_NEAREST)
                background_blocks = resized_mask == 0
                removability_scores[i][background_blocks] *= 10.0
            else:
                print(f'Warning: Mask file not found for frame {i}: {mask_path}')
        if smoothing_beta < 1 and removability_scores.shape[0] >= 2:
            print('Applying temporal smoothing to removability scores...')
            smoothed_scores = np.zeros_like(removability_scores)
            smoothed_scores[0] = removability_scores[0]
            smoothed_scores[1:] = smoothing_beta * removability_scores[1:] + (1 - smoothing_beta) * removability_scores[:-1]
            removability_scores = smoothed_scores
        removability_scores = normalize_array(removability_scores)
        return removability_scores
    except Exception as e:
        print(f'Error in calculate_removability_scores: {e}')
        raise
def split_image_into_blocks(image: np.ndarray, block_size: int) -> np.ndarray:
    """
    Splits an image into a 5D array of blocks.
    Shape: (num_blocks_y, num_blocks_x, block_size, block_size, channels)
    """
    block_size = int(block_size)
    (h, w, c) = image.shape
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size
    blocks = image.reshape(num_blocks_y, block_size, num_blocks_x, block_size, c)
    blocks = blocks.swapaxes(1, 2)
    return blocks
def apply_selective_removal(image: np.ndarray, frame_scores: np.ndarray, block_size: int, shrink_amount: float, cluster_blocks: bool = False) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Remove blocks based on scores. Returns (new_image, removal_mask, removed_coords)."""
    (h, w, c) = image.shape
    pad_y = (block_size - h % block_size) % block_size
    pad_x = (block_size - w % block_size) % block_size
    if pad_y > 0 or pad_x > 0:
        image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        frame_scores = np.pad(frame_scores, ((0, 1 if pad_y > 0 else 0), (0, 1 if pad_x > 0 else 0)), mode='constant', constant_values=0)
    (num_blocks_y, num_blocks_x) = frame_scores.shape
    if shrink_amount < 1.0:
        num_blocks_to_remove = int(shrink_amount * num_blocks_x)
    else:
        num_blocks_to_remove = int(shrink_amount)
    num_blocks_to_remove = min(num_blocks_to_remove, num_blocks_x)
    block_coords_to_remove = []
    
    # Optional clustering: blur the scores so peaks are wider, forcing selected blocks to clump together
    selection_scores = frame_scores
    if cluster_blocks:
        selection_scores = cv2.GaussianBlur(frame_scores.astype(np.float32), (5, 5), 0)
        
    for j in range(num_blocks_y):
        row_scores = selection_scores[j, :]
        indices_to_remove = np.argsort(-row_scores)[:num_blocks_to_remove]
        indices_to_remove.sort()
        block_coords_to_remove.append(indices_to_remove.tolist())
    blocks = split_image_into_blocks(image, block_size)
    removal_mask = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int8)
    rows_indices = np.arange(num_blocks_y).repeat([len(cols) for cols in block_coords_to_remove])
    if len(rows_indices) > 0:
        cols_indices = np.concatenate(block_coords_to_remove)
        removal_mask[rows_indices, cols_indices] = 1
    kept_blocks_list = [blocks[i, np.where(removal_mask[i] == 0)[0]] for i in range(num_blocks_y)]
    kept_blocks = np.stack(kept_blocks_list, axis=0)
    new_image = combine_blocks_into_image(kept_blocks)
    if pad_y > 0:
        new_image = new_image[:-pad_y, :, :]
        removal_mask = removal_mask[:-1, :]
    if pad_x > 0:
        new_image = new_image[:, :-pad_x, :]
        removal_mask = removal_mask[:, :-1]
    return (new_image, removal_mask, block_coords_to_remove)
def select_removal_mask_global(frame_scores: np.ndarray, amount: float, cluster_blocks: bool = True, exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """Select the globally top-k most-removable blocks (no per-row constraint).

    ELVIS's shrink transport needs an equal number of removed blocks per row so
    surviving blocks repack into a rectangle; blackout/freeze keep native
    geometry, so that per-row constraint is pure loss -- it forces removing a
    low-removability block in a foreground-heavy row while sparing a
    high-removability one elsewhere. Selecting the global top-k spends the same
    removal budget on the actually-most-removable blocks. Budget is matched to
    the per-row default (int(amount*num_blocks_x) per row) so a global run is
    directly comparable to the per-row run at the same `amount`.

    ``exclude`` (bool, same shape as frame_scores): blocks that must NEVER be
    selected. The removability model only *softly* protects foreground (BG
    scores x10), which fails on high-motion foregrounds -- measured on
    bmx-trees: FG mean score 0.127 > BG 0.113, so top-k removed 12.8% of FG
    blocks and FG-PSNR collapsed by 3 dB. Passing the UFO foreground blocks
    here makes the protection hard. The budget is capped at the number of
    non-excluded blocks.
    """
    num_blocks_y, num_blocks_x = frame_scores.shape
    per_row = int(amount * num_blocks_x) if amount < 1.0 else int(amount)
    k = min(per_row * num_blocks_y, num_blocks_y * num_blocks_x)
    mask = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int8)
    selection_scores = frame_scores.astype(np.float32)
    if cluster_blocks:
        selection_scores = cv2.GaussianBlur(selection_scores, (5, 5), 0)
    if exclude is not None:
        # Exclude AFTER the cluster blur so smeared FG scores can't re-enter.
        selection_scores = np.where(exclude, -np.inf, selection_scores)
        k = min(k, int((~exclude).sum()))
    if k <= 0:
        return mask
    idx = np.argpartition(-selection_scores.ravel(), k - 1)[:k]
    mask.ravel()[idx] = 1
    return mask


def _selected_blocks(frame_scores: np.ndarray) -> np.ndarray:
    """Binary per-block selection used by the mean_fill/freeze degradations.

    Matches the existing presley_ai filters (downsample/blur), which degrade any
    block with round(score) > 0 (i.e. removability >= 0.5).
    """
    return (np.round(frame_scores).astype(np.int32) > 0)


def filter_frame_mean_fill(image: np.ndarray, frame_scores: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Replace selected blocks with their mean color (flat DC ~ near-free to code).

    A flat block costs the encoder almost nothing (one DC coefficient, strong
    inter/intra prediction), so this pushes bits away from removable/background
    regions the same way blackout does -- but leaves a smooth prior instead of
    black, which the in-painter restores. Returns (image, binary strength map).
    """
    h, w, c = image.shape
    num_blocks_y, num_blocks_x = frame_scores.shape
    sel = _selected_blocks(frame_scores)
    out = image.copy()
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            if sel[by, bx]:
                y0, y1 = by * block_size, min((by + 1) * block_size, h)
                x0, x1 = bx * block_size, min((bx + 1) * block_size, w)
                blk = out[y0:y1, x0:x1]
                out[y0:y1, x0:x1] = blk.reshape(-1, c).mean(0).astype(np.uint8)
    return out, sel.astype(np.int8)


def filter_frame_freeze(image: np.ndarray, frame_scores: np.ndarray, block_size: int, prev_image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Copy selected blocks from the previous degraded frame (inter-skip ~ 0 bits).

    Frame 0 keeps original content (no previous frame), giving the I-frame a
    real-texture prior. Returns (image, binary strength map).
    """
    h, w, c = image.shape
    num_blocks_y, num_blocks_x = frame_scores.shape
    sel = _selected_blocks(frame_scores)
    out = image.copy()
    if prev_image is not None:
        for by in range(num_blocks_y):
            for bx in range(num_blocks_x):
                if sel[by, bx]:
                    y0, y1 = by * block_size, min((by + 1) * block_size, h)
                    x0, x1 = bx * block_size, min((bx + 1) * block_size, w)
                    out[y0:y1, x0:x1] = prev_image[y0:y1, x0:x1]
    return out, sel.astype(np.int8)


def combine_blocks_into_image(blocks: np.ndarray) -> np.ndarray:
    """Combine 5D array of blocks back into single image. Inverse of split_image_into_blocks."""
    (num_blocks_y, num_blocks_x, block_size, _, c) = blocks.shape
    image = blocks.swapaxes(1, 2)
    image = image.reshape(num_blocks_y * block_size, num_blocks_x * block_size, c)
    return image
def filter_frame_downsample(image: np.ndarray, frame_scores: np.ndarray, block_size: int, scale: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptively downsample each block based on removability scores. Returns (image, downsample_maps)."""
    (h, w, c) = image.shape
    pad_y = (block_size - h % block_size) % block_size
    pad_x = (block_size - w % block_size) % block_size
    if pad_y > 0 or pad_x > 0:
        image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        frame_scores = np.pad(frame_scores, ((0, 1 if pad_y > 0 else 0), (0, 1 if pad_x > 0 else 0)), mode='constant', constant_values=0)
    blocks = split_image_into_blocks(image, block_size)
    downsample_maps = np.round(frame_scores).astype(np.int32)
    processed_blocks = blocks.copy()
    (num_blocks_y, num_blocks_x) = (blocks.shape[0], blocks.shape[1])
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            if downsample_maps[by, bx] > 0:
                block = blocks[by, bx]
                small_size = max(1, int(block_size * scale))
                small_block = cv2.resize(block, (small_size, small_size), interpolation=cv2.INTER_AREA)
                upsampled_block = cv2.resize(small_block, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
                processed_blocks[by, bx] = upsampled_block
    new_image = combine_blocks_into_image(processed_blocks)
    if pad_y > 0:
        new_image = new_image[:-pad_y, :, :]
        downsample_maps = downsample_maps[:-1, :]
    if pad_x > 0:
        new_image = new_image[:, :-pad_x, :]
        downsample_maps = downsample_maps[:, :-1]
    return (new_image, downsample_maps)
def filter_frame_gaussian(image: np.ndarray, frame_scores: np.ndarray, block_size: int, kernel_size: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Apply adaptive Gaussian blur per block based on scores. Returns (image, blur_strengths)."""
    (h, w, c) = image.shape
    pad_y = (block_size - h % block_size) % block_size
    pad_x = (block_size - w % block_size) % block_size
    if pad_y > 0 or pad_x > 0:
        image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        frame_scores = np.pad(frame_scores, ((0, 1 if pad_y > 0 else 0), (0, 1 if pad_x > 0 else 0)), mode='constant', constant_values=0)
    blocks = split_image_into_blocks(image, block_size)
    blur_strengths = np.round(frame_scores).astype(np.int32)
    processed_blocks = blocks.copy()
    (num_blocks_y, num_blocks_x) = (blocks.shape[0], blocks.shape[1])
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            if blur_strengths[by, bx] > 0:
                block = blocks[by, bx]
                # Ensure kernel is odd
                k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                blurred_block = cv2.GaussianBlur(block, (k, k), 0)
                processed_blocks[by, bx] = blurred_block
    new_image = combine_blocks_into_image(processed_blocks)
    if pad_y > 0:
        new_image = new_image[:-pad_y, :, :]
        blur_strengths = blur_strengths[:-1, :]
    if pad_x > 0:
        new_image = new_image[:, :-pad_x, :]
        blur_strengths = blur_strengths[:, :-1]
    return (new_image, blur_strengths)

def filter_frame_noise(image: np.ndarray, frame_scores: np.ndarray, block_size: int, noise_variance: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """Apply adaptive Gaussian noise per block based on scores. Returns (image, noise_strengths)."""
    (h, w, c) = image.shape
    pad_y = (block_size - h % block_size) % block_size
    pad_x = (block_size - w % block_size) % block_size
    if pad_y > 0 or pad_x > 0:
        image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        frame_scores = np.pad(frame_scores, ((0, 1 if pad_y > 0 else 0), (0, 1 if pad_x > 0 else 0)), mode='constant', constant_values=0)
    blocks = split_image_into_blocks(image, block_size)
    noise_strengths = np.round(frame_scores * noise_variance).astype(np.float32)
    processed_blocks = blocks.copy()
    (num_blocks_y, num_blocks_x) = (blocks.shape[0], blocks.shape[1])
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            strength = noise_strengths[by, bx]
            if strength > 0:
                block = blocks[by, bx]
                noise = np.random.normal(0, strength, block.shape)
                noisy_block = np.clip(block.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                processed_blocks[by, bx] = noisy_block
    new_image = combine_blocks_into_image(processed_blocks)
    if pad_y > 0:
        new_image = new_image[:-pad_y, :, :]
        noise_strengths = noise_strengths[:-1, :]
    if pad_x > 0:
        new_image = new_image[:, :-pad_x, :]
        noise_strengths = noise_strengths[:, :-1]
    return (new_image, noise_strengths)

def filter_frame_qp(image: np.ndarray, frame_scores: np.ndarray, block_size: int, qp_range: int = 15, base_qp: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Apply DCT quantization per block based on scores to simulate QP degradation. Returns (image, qp_maps)."""
    (h, w, c) = image.shape
    pad_y = (block_size - h % block_size) % block_size
    pad_x = (block_size - w % block_size) % block_size
    if pad_y > 0 or pad_x > 0:
        image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        frame_scores = np.pad(frame_scores, ((0, 1 if pad_y > 0 else 0), (0, 1 if pad_x > 0 else 0)), mode='constant', constant_values=0)
    
    # Convert to YCrCb to apply quantization in luminance and chrominance
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    
    blocks = split_image_into_blocks(img_ycrcb, block_size)
    qp_maps = np.round(frame_scores * qp_range).astype(np.int32)
    processed_blocks = blocks.copy()
    
    (num_blocks_y, num_blocks_x) = (blocks.shape[0], blocks.shape[1])
    
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            delta_qp = qp_maps[by, bx]
            if delta_qp > 0:
                block = blocks[by, bx]
                qp = base_qp + delta_qp
                q_step = 2.0 ** (qp / 6.0)
                
                # Process each channel
                for ch in range(c):
                    # OpenCV dct requires float32
                    channel_data = block[:, :, ch] - 128.0 # Shift for DCT
                    dct_coeffs = cv2.dct(channel_data)
                    
                    # Quantize
                    dc = dct_coeffs[0, 0]
                    dct_quantized = np.round(dct_coeffs / q_step) * q_step
                    dct_quantized[0, 0] = dc # preserving DC avoids massive color shifts
                    
                    idct_coeffs = cv2.idct(dct_quantized)
                    processed_blocks[by, bx, :, :, ch] = idct_coeffs + 128.0

    new_img_ycrcb = combine_blocks_into_image(processed_blocks)
    new_img_ycrcb = np.clip(new_img_ycrcb, 0, 255).astype(np.uint8)
    new_image = cv2.cvtColor(new_img_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    if pad_y > 0:
        new_image = new_image[:-pad_y, :, :]
        qp_maps = qp_maps[:-1, :]
    if pad_x > 0:
        new_image = new_image[:, :-pad_x, :]
        qp_maps = qp_maps[:, :-1]
        
    return (new_image, qp_maps)
