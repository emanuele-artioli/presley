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

def load_frame(path: str) -> np.ndarray:
    """Load a single frame from disk as BGR numpy array."""
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise IOError(f'Failed to load frame: {path}')
    return frame
def save_frame(frame: np.ndarray, path: str) -> None:
    """Save a single frame to disk."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if not cv2.imwrite(path, frame):
        raise IOError(f'Failed to save frame: {path}')
def load_frames(directory: str, pattern: str='*.png') -> List[np.ndarray]:
    """Load all frames from a directory matching the pattern, sorted by name."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f'Directory does not exist: {directory}')
    paths = sorted(dir_path.glob(pattern))
    if not paths:
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            paths = sorted(dir_path.glob(ext))
            if paths:
                break
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    return frames
def save_frames(frames: Sequence[np.ndarray], directory: str, pattern: str='%05d.png', start_index: int=1) -> List[str]:
    """Save frames to directory with numbered filenames. Returns list of saved paths."""
    os.makedirs(directory, exist_ok=True)
    saved_paths = []
    for (idx, frame) in enumerate(frames):
        filename = pattern % (start_index + idx)
        path = os.path.join(directory, filename)
        if not cv2.imwrite(path, frame):
            raise IOError(f'Failed to save frame: {path}')
        saved_paths.append(path)
    return saved_paths
def load_masks(directory: str, width: int, height: int, expected_count: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load and resize masks, returning (fg_masks, bg_masks) as boolean arrays."""
    if expected_count <= 0:
        return ([], [])
    dir_path = Path(directory)
    mask_files = []
    if dir_path.is_dir():
        mask_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fg_masks: List[np.ndarray] = []
    bg_masks: List[np.ndarray] = []
    last_mask: Optional[np.ndarray] = None
    for frame_idx in range(expected_count):
        if frame_idx < len(mask_files):
            mask_path = os.path.join(directory, mask_files[frame_idx])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                if mask_img.shape[:2] != (height, width):
                    mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
                last_mask = mask_img
        if last_mask is not None:
            fg_mask = last_mask > 127
            bg_mask = ~fg_mask
        else:
            fg_mask = np.ones((height, width), dtype=bool)
            bg_mask = np.zeros((height, width), dtype=bool)
        fg_masks.append(fg_mask)
        bg_masks.append(bg_mask)
    return (fg_masks, bg_masks)
def clear_directory(directory: str, patterns: Sequence[str]=('*.png', '*.jpg', '*.jpeg')) -> None:
    """Remove files matching patterns from directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return
    for pattern in patterns:
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
def get_frame_paths(directory: str) -> List[Path]:
    """Get sorted list of frame file paths in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    valid_suffixes = ('.png', '.jpg', '.jpeg')
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in valid_suffixes])
def _load_resized_masks(masks_dir: str, width: int, height: int, expected_frames: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load, resize, and cache foreground/background masks as boolean arrays."""
    if expected_frames <= 0:
        return ([], [])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.isdir(masks_dir) else []
    fg_masks: List[np.ndarray] = []
    bg_masks: List[np.ndarray] = []
    last_mask: Optional[np.ndarray] = None
    for frame_idx in range(expected_frames):
        if frame_idx < len(mask_files):
            mask_path = os.path.join(masks_dir, mask_files[frame_idx])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                mask_bool = np.zeros((height, width), dtype=bool)
            else:
                mask_resized = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 128
        elif last_mask is not None:
            mask_bool = last_mask
        else:
            mask_bool = np.zeros((height, width), dtype=bool)
        fg_masks.append(mask_bool)
        bg_masks.append(~mask_bool)
        last_mask = mask_bool
    return (fg_masks, bg_masks)
