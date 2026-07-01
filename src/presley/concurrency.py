from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

@dataclass
class ChunkSpec:
    start: int
    end: int
    device: 'torch.device'
    chunk_id: int = 0

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



def chunk_for_devices(total: int, devices: List[torch.device], min_chunk_size: int=1) -> List[ChunkSpec]:
    """Split total items into chunks, one per device."""
    if not devices or total <= 0:
        return []
    num_devices = len(devices)
    base_size = total // num_devices
    remainder = total % num_devices
    chunks = []
    start = 0
    for (idx, device) in enumerate(devices):
        size = base_size + (1 if idx < remainder else 0)
        if size < min_chunk_size and idx > 0:
            continue
        end = start + size
        if end > start:
            chunks.append(ChunkSpec(start=start, end=end, device=device, chunk_id=idx))
        start = end
    return chunks
def parallel_process_frames(process_fn: Callable[[List[np.ndarray], torch.device], List[np.ndarray]], frames: List[np.ndarray], devices: List[torch.device], chunk_size: Optional[int]=None, max_workers: Optional[int]=None) -> List[np.ndarray]:
    """
    Process frames in parallel across devices using ThreadPoolExecutor.
    
    Args:
        process_fn: Function that takes (frames_chunk, device) and returns processed frames
        frames: List of frames to process
        devices: List of devices to use
        chunk_size: Optional fixed chunk size (otherwise auto-calculated)
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of processed frames in original order
    """
    if not frames:
        return []
    if not devices:
        devices = [torch.device('cpu')]
    num_frames = len(frames)
    if chunk_size is None:
        chunks = chunk_for_devices(num_frames, devices)
    else:
        chunks = []
        cursor = 0
        chunk_id = 0
        while cursor < num_frames:
            end = min(cursor + chunk_size, num_frames)
            device = devices[chunk_id % len(devices)]
            chunks.append(ChunkSpec(start=cursor, end=end, device=device, chunk_id=chunk_id))
            cursor = end
            chunk_id += 1
    if not chunks:
        return []
    if len(chunks) == 1:
        chunk = chunks[0]
        return process_fn(frames[chunk.start:chunk.end], chunk.device)
    results: Dict[int, List[np.ndarray]] = {}
    workers = max_workers or min(len(chunks), len(devices))

    def _process_chunk(chunk: ChunkSpec) -> Tuple[int, List[np.ndarray]]:
        chunk_frames = frames[chunk.start:chunk.end]
        processed = process_fn(chunk_frames, chunk.device)
        return (chunk.chunk_id, processed)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            (chunk_id, processed_frames) = future.result()
            results[chunk_id] = processed_frames
    output = []
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        output.extend(results[chunk.chunk_id])
    return output
def _resolve_device_list(devices: Optional[Sequence[Union[int, str, torch.device]]], *, prefer_cuda: bool=True, allow_cpu_fallback: bool=True) -> List[torch.device]:
    """Normalize user-provided device specifiers into unique torch.device entries."""
    available_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def _normalize_device(spec: Union[int, str, torch.device]) -> torch.device:
        if isinstance(spec, torch.device):
            device_obj = spec
        elif isinstance(spec, int):
            if not torch.cuda.is_available():
                raise ValueError('CUDA device indices were provided but no CUDA devices are available.')
            if spec < 0 or spec >= available_gpu_count:
                raise ValueError(f'Requested CUDA device index {spec} is out of range.')
            device_obj = torch.device(f'cuda:{spec}')
        else:
            spec_str = str(spec)
            if spec_str.startswith('cuda'):
                if not torch.cuda.is_available():
                    raise ValueError('CUDA devices were requested but CUDA is not available.')
                if spec_str == 'cuda' or spec_str == 'cuda:':
                    device_obj = torch.device('cuda')
                else:
                    try:
                        idx_part = spec_str.split(':', 1)[1]
                        idx_val = int(idx_part)
                    except (IndexError, ValueError):
                        raise ValueError(f"Invalid CUDA device string '{spec_str}'.") from None
                    if idx_val < 0 or idx_val >= available_gpu_count:
                        raise ValueError(f'Requested CUDA device {spec_str} exceeds detected count {available_gpu_count}.')
                    device_obj = torch.device(f'cuda:{idx_val}')
            else:
                device_obj = torch.device(spec_str)
        if device_obj.type == 'cuda':
            idx = device_obj.index if device_obj.index is not None else 0
            if idx < 0 or idx >= available_gpu_count:
                raise ValueError(f'Requested CUDA device {idx} is not available. Detected {available_gpu_count} device(s).')
        return device_obj
    if not devices:
        if prefer_cuda and available_gpu_count > 0:
            device_specs: Sequence[Union[int, str, torch.device]] = [f'cuda:{idx}' for idx in range(available_gpu_count)]
        elif allow_cpu_fallback:
            device_specs = ['cpu']
        else:
            raise ValueError('No CUDA devices available and CPU fallback disabled.')
    else:
        device_specs = devices
    resolved_devices: List[torch.device] = []
    seen_keys = set()
    for spec in device_specs:
        device_obj = _normalize_device(spec)
        key = str(device_obj)
        if device_obj.type == 'cuda':
            idx = device_obj.index if device_obj.index is not None else 0
            key = f'cuda:{idx}'
        if key in seen_keys:
            continue
        seen_keys.add(key)
        resolved_devices.append(device_obj)
    if not resolved_devices:
        if allow_cpu_fallback:
            resolved_devices.append(torch.device('cpu'))
        else:
            raise ValueError('No valid compute devices resolved from the provided specification.')
    return resolved_devices
