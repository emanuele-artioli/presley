import os
import math
import time
from typing import Dict, Any, List
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from presley.hnerv_arch import (
    HNeRVConfig,
    HNeRVModel,
    count_decoder_parameters,
    load_hnerv_checkpoint,
    save_hnerv_checkpoint,
)
from presley.encode_utils import save_frames_as_video

def _load_training_frames_from_pattern(frames_pattern: str, height: int, width: int) -> torch.Tensor:
    """Load frames from PNG sequence, resize if needed, return RGB float tensor.
    Returns Shape: [NumFrames, 3, Height, Width], values in [0, 1], RGB channel order.
    """
    import glob
    # frames_pattern is usually like cache_dir/.../%05d.png
    # Replace %05d with * for glob
    search_pattern = frames_pattern.replace("%05d", "*").replace("%04d", "*").replace("%06d", "*")
    files = sorted(glob.glob(search_pattern))
    if not files:
        raise ValueError(f"No frames found matching pattern {search_pattern}")

    frames_bgr = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        frames_bgr.append(img)

    if not frames_bgr:
        raise ValueError(f"No frames loaded from {search_pattern}")

    stacked = np.stack(frames_bgr, axis=0)  # Shape: [NumFrames, Height, Width, 3] BGR uint8
    stacked_rgb = stacked[..., ::-1]  # BGR -> RGB
    tensor = torch.from_numpy(np.ascontiguousarray(stacked_rgb)).permute(0, 3, 1, 2).contiguous()
    tensor = tensor.to(torch.float32) / 255.0  # Shape: [NumFrames, 3, Height, Width]
    return tensor

def _decode_hnerv_to_bgr_list(checkpoint_path: Path, device: torch.device) -> List[np.ndarray]:
    """Reload checkpoint, decode, and return list of BGR numpy arrays."""
    decoder, embeddings = load_hnerv_checkpoint(checkpoint_path, device=device)
    decoder = decoder.to(device).eval()
    with torch.no_grad():
        reconstruction = decoder(embeddings.to(device))  # Shape: [NumFrames, 3, Height, Width]
    reconstruction = reconstruction.cpu()
    
    frames_uint8 = (reconstruction.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
    frames_bgr = []
    for index in range(frames_uint8.shape[0]):
        frame_rgb = frames_uint8[index].transpose(1, 2, 0)  # HWC RGB
        frame_bgr = frame_rgb[..., ::-1].copy()  # BGR
        frames_bgr.append(frame_bgr)
    return frames_bgr

def encode_video_hnerv(
    ref_frames_pattern: str, 
    output_video: str, 
    framerate: float, 
    width: int, 
    height: int, 
    codec_params: Dict[str, Any], 
    checkpoint_path: str
) -> float:
    """
    Train HNeRV to overfit a video, save checkpoint, and decode back to a lossless MP4.
    Returns the training time in seconds.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load default params, overriding with codec_params
    epochs = int(codec_params.get("epochs", 4000))
    lr = float(codec_params.get("lr", 1e-3))
    
    embed_height = int(codec_params.get("embed_height", 9))
    embed_width = int(codec_params.get("embed_width", 16))
    embed_channels = int(codec_params.get("embed_channels", 64))
    
    # Strides and channels default for 360p (360=9*40, 640=16*40 -> 5*4*2=40)
    # The default resolution for HNeRV training is typically 640x360.
    target_height = int(codec_params.get("height", 360))
    target_width = int(codec_params.get("width", 640))
    
    strides_str = str(codec_params.get("strides", "5,4,2"))
    channels_str = str(codec_params.get("channels", "128,64,32"))
    
    strides = tuple(int(s.strip()) for s in strides_str.split(","))
    channels = tuple(int(c.strip()) for c in channels_str.split(","))
    
    config = HNeRVConfig(
        height=target_height,
        width=target_width,
        embed_height=embed_height,
        embed_width=embed_width,
        embed_channels=embed_channels,
        strides=strides,
        channels=channels,
    )
    
    frames = _load_training_frames_from_pattern(ref_frames_pattern, target_height, target_width)
    
    # Train
    torch.manual_seed(0)
    model = HNeRVModel(config).to(device)
    frames_dev = frames.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        reconstruction, _ = model(frames_dev)
        loss = F.l1_loss(reconstruction, frames_dev)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                mse = F.mse_loss(reconstruction, frames_dev).item()
                psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
            print(f"[hnerv] epoch {epoch}/{epochs} loss={loss.item():.5f} psnr={psnr:.2f}dB", flush=True)

    train_seconds = time.perf_counter() - started
    
    # Serialize to disk
    model.eval()
    with torch.no_grad():
        _, final_embedding = model(frames_dev)
    save_hnerv_checkpoint(checkpoint_path, model.decoder, final_embedding.cpu())

    # Decode and save to video
    decoded_bgr_list = _decode_hnerv_to_bgr_list(Path(checkpoint_path), device)
    
    # We output to mp4 losslessly using libx265 as expected by the pipeline.
    # The evaluation script handles resizing this back to native resolution for PSNR computation.
    save_frames_as_video(decoded_bgr_list, output_video, framerate, lossless=True, codec="libx265")
    
    return train_seconds
