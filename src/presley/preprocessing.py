import os
import cv2
import sys
import shutil
import subprocess
import numpy as np
from pathlib import Path

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalizes a NumPy array to the range [0, 1]."""
    min_val, max_val = arr.min(), arr.max()
    return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr

def get_reference_frames(video_name: str, width: int, height: int, dataset_dir: str, cache_dir: str):
    """
    Returns (raw_yuv_path, frames_list, framerate).
    Caches extracted frames and YUV at target resolution.
    """
    # Assuming 24fps default for DAVIS if not specified
    framerate = 24.0 
    
    key_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}")
    os.makedirs(key_dir, exist_ok=True)
    
    raw_yuv_path = os.path.join(key_dir, "reference_raw.yuv")
    ref_frames_dir = os.path.join(key_dir, "reference_frames")
    
    if os.path.exists(raw_yuv_path) and os.path.exists(ref_frames_dir):
        # Already cached
        frames = []
        for p in sorted(Path(ref_frames_dir).glob("*.png")):
            frames.append(cv2.imread(str(p), cv2.IMREAD_COLOR))
        if frames:
            return raw_yuv_path, frames, framerate

    # Not cached, build it from dataset
    source_dir = os.path.join(dataset_dir, video_name)
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source video frames not found at {source_dir}")
        
    os.makedirs(ref_frames_dir, exist_ok=True)
    frames = []
    
    src_paths = sorted(Path(source_dir).glob("*.jpg"))
    if not src_paths:
        src_paths = sorted(Path(source_dir).glob("*.png"))
        
    for i, p in enumerate(src_paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None: continue
        
        # Resize to target
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
        dst_path = os.path.join(ref_frames_dir, f"{i+1:05d}.png")
        cv2.imwrite(dst_path, img)
        frames.append(img)
        
    # Generate YUV (lossless equivalent of the resized frames for EVCA to use)
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(framerate),
        "-i", os.path.join(ref_frames_dir, "%05d.png"),
        "-pix_fmt", "yuv420p", raw_yuv_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
    return raw_yuv_path, frames, framerate


def get_evca_scores(video_name: str, width: int, height: int, block_size: int,
                    raw_yuv_path: str, reference_frames_dir: str, cache_dir: str):
    """
    Computes EVCA temporal and spatial complexity scores.
    Returns (temporal_3d, spatial_3d).
    """
    key_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}_bs{block_size}")
    os.makedirs(key_dir, exist_ok=True)
    
    evca_tc_dest = os.path.join(key_dir, "evca_TC_blocks.csv")
    evca_sc_dest = os.path.join(key_dir, "evca_SC_blocks.csv")
    
    frame_count = len(list(Path(reference_frames_dir).glob("*.png")))
    
    if not (os.path.exists(evca_tc_dest) and os.path.exists(evca_sc_dest)):
        try:
            import importlib
            evca_pkg = importlib.import_module('evca')
        except ImportError as exc:
            raise RuntimeError("The 'evca' package is not installed.") from exc
            
        evca_root = Path(evca_pkg.__file__).resolve().parent
        package_tc = evca_root / 'evca_TC_blocks.csv'
        package_sc = evca_root / 'evca_SC_blocks.csv'
        
        for p in (package_tc, package_sc, evca_root / 'evca.csv'):
            if p.exists():
                try: p.unlink()
                except: pass
                
        evca_cmd = [
            sys.executable, '-m', 'evca.main', 
            '-i', os.path.abspath(raw_yuv_path), 
            '-r', f'{width}x{height}', 
            '-b', str(block_size), 
            '-f', str(frame_count), 
            '-c', os.path.join(os.path.abspath(key_dir), 'evca.csv'), 
            '-bi', '1'
        ]
        result = subprocess.run(evca_cmd, capture_output=True, text=True, cwd=str(key_dir))
        if result.returncode != 0:
            raise RuntimeError(f"EVCA execution failed: {result.stderr}\n{result.stdout}")
            
        # EVCA writes directly to cwd, which is key_dir, so the files are already there
        
    temporal_array = np.loadtxt(evca_tc_dest, delimiter=',', skiprows=1)
    spatial_array = np.loadtxt(evca_sc_dest, delimiter=',', skiprows=1)
    
    num_blocks_x = width // block_size
    num_blocks_y = height // block_size
    num_frames = min(temporal_array.shape[1], spatial_array.shape[1])
    
    temporal_3d = temporal_array[:, :num_frames].T.reshape(num_frames, num_blocks_y, num_blocks_x)
    spatial_3d = spatial_array[:, :num_frames].T.reshape(num_frames, num_blocks_y, num_blocks_x)
    
    temporal_3d = normalize_array(temporal_3d)
    spatial_3d = normalize_array(spatial_3d)
    
    return temporal_3d, spatial_3d


def get_ufo_masks(video_name: str, width: int, height: int, block_size: int,
                  reference_frames_dir: str, cache_dir: str):
    """
    Returns UFO masks as (F, H, W) array.
    """
    key_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}")
    ufo_masks_dir = os.path.join(key_dir, "ufo_masks")
    
    frame_files = sorted(Path(reference_frames_dir).glob("*.png"))
    
    if not os.path.exists(ufo_masks_dir) or len(list(Path(ufo_masks_dir).glob("*.png"))) != len(frame_files):
        os.makedirs(ufo_masks_dir, exist_ok=True)
        import torch
        from ufo.test import segment_frames
        
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        frames_list = []
        for fpath in frame_files:
            frames_list.append(cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB))
            
        frames_arr = np.array(frames_list)
        
        try:
            import importlib
            ufo_pkg = importlib.import_module('ufo')
            model_path_for_ufo = str(Path(ufo_pkg.__file__).parent / 'weights' / 'video_best.pth')
        except Exception:
            model_path_for_ufo = 'weights/video_best.pth'
            
        if not os.path.exists(model_path_for_ufo):
            # Try downloader
            try:
                downloader = importlib.import_module('ufo.download_ufo_weights')
                model_path_for_ufo = str(downloader.main())
            except Exception:
                pass
                
        masks_arr = segment_frames(
            frames=frames_arr,
            device=device_str,
            model_path=model_path_for_ufo,
            group_size=5,
            img_size=224
        )
        
        for i, fname in enumerate(frame_files):
            mask_path = os.path.join(ufo_masks_dir, fname.name)
            mask_uint8 = (masks_arr[i] * 255.0).astype(np.uint8)
            cv2.imwrite(mask_path, mask_uint8)
            
    # Load masks
    masks = []
    for p in sorted(Path(ufo_masks_dir).glob("*.png")):
        masks.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        
    return np.array(masks)


def get_removability_scores(video_name: str, width: int, height: int, block_size: int,
                            alpha: float, beta: float, dataset_dir: str, cache_dir: str):
    """
    Returns combined removability scores (F, BY, BX). Caches to disk.
    """
    key_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}_bs{block_size}")
    os.makedirs(key_dir, exist_ok=True)
    
    score_path = os.path.join(key_dir, f"removability_a{alpha:.2f}_b{beta:.2f}.npy")
    
    if os.path.exists(score_path):
        return np.load(score_path)
        
    raw_yuv_path, frames, _ = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    
    temporal_3d, spatial_3d = get_evca_scores(video_name, width, height, block_size, raw_yuv_path, ref_frames_dir, cache_dir)
    ufo_masks = get_ufo_masks(video_name, width, height, block_size, ref_frames_dir, cache_dir)
    
    removability_scores = np.zeros_like(spatial_3d)
    removability_scores[:-1] = alpha * spatial_3d[:-1] + (1 - alpha) * temporal_3d[1:]
    removability_scores[-1] = spatial_3d[-1]
    
    num_blocks_x = width // block_size
    num_blocks_y = height // block_size
    num_frames = removability_scores.shape[0]
    
    for i in range(num_frames):
        mask = ufo_masks[i]
        resized_mask = cv2.resize(mask, (num_blocks_x, num_blocks_y), interpolation=cv2.INTER_NEAREST)
        background_blocks = resized_mask == 0
        removability_scores[i][background_blocks] *= 10.0
        
    if beta < 1.0 and num_frames >= 2:
        smoothed = np.zeros_like(removability_scores)
        smoothed[0] = removability_scores[0]
        smoothed[1:] = beta * removability_scores[1:] + (1 - beta) * removability_scores[:-1]
        removability_scores = smoothed
        
    removability_scores = normalize_array(removability_scores)
    np.save(score_path, removability_scores)
    
    return removability_scores
