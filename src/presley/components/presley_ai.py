import os
import time
import numpy as np
from typing import Dict, Any

from presley.preprocessing import get_reference_frames, get_removability_scores
from presley.encode_utils import save_frames_as_video, load_frames_from_video, encode_video_x265
from presley.degradation import filter_frame_downsample, filter_frame_gaussian

def run_presley_ai(experiment: Dict[str, Any], dataset_dir: str, results_dir: str, cache_dir: str) -> Dict[str, Any]:
    video_name = experiment['video']
    width = experiment['width']
    height = experiment['height']
    block_size = experiment['block_size']
    alpha = experiment['alpha']
    beta = experiment['beta']
    
    degradation = experiment['degradation'].lower()
    restorer = experiment['restorer'].lower()
    
    codec = experiment['codec'].lower()
    target_bitrate = experiment['target_bitrate']
    codec_params = experiment.get('codec_params', {})
    restorer_params = experiment.get('restorer_params', {})
    
    # 1. Load data
    raw_yuv_path, frames, framerate = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    removability_scores = get_removability_scores(video_name, width, height, block_size, alpha, beta, dataset_dir, cache_dir)
    
    start_time = time.time()
    
    # 2. Degrade
    degraded_frames_list = []
    strength_maps_list = []
    frames_arr = np.array(frames)
    
    for i in range(len(frames)):
        frame = frames_arr[i]
        score = removability_scores[i]
        
        if degradation == 'downsample':
            degraded, smap = filter_frame_downsample(frame, score, block_size)
        elif degradation == 'blur':
            degraded, smap = filter_frame_gaussian(frame, score, block_size)
        else:
            raise ValueError(f"Unknown degradation: {degradation}")
            
        degraded_frames_list.append(degraded)
        strength_maps_list.append(smap)
        
    temp_degraded_vid = os.path.join(results_dir, "temp_degraded_lossless.mkv")
    save_frames_as_video(degraded_frames_list, temp_degraded_vid, framerate, lossless=True, codec="libx265")
    
    # 3. Encode degraded frames
    transmitted_video = os.path.join(results_dir, "encoded_degraded.mp4")
    if codec == 'x265':
        encode_video_x265(temp_degraded_vid, transmitted_video, framerate, target_bitrate, preset=codec_params.get('preset', 'medium'))
    else:
        raise ValueError(f"Presley AI currently requires x265 for encoding, got {codec}")
        
    encoding_time = time.time() - start_time
    restoration_start = time.time()
    
    # Save strength maps (transmitted side information)
    strength_maps_path = os.path.join(results_dir, "strength_maps.npz")
    np.savez_compressed(strength_maps_path, strength_maps=np.array(strength_maps_list))
    
    # 4. Decode degraded frames
    decoded_degraded = load_frames_from_video(transmitted_video)
    
    # 5. Restore
    temp_frames_dir = os.path.join(results_dir, "temp_restoring")
    restored_frames_dir = os.path.join(results_dir, "restored_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)
    os.makedirs(restored_frames_dir, exist_ok=True)
    
    import cv2
    for i in range(len(decoded_degraded)):
        cv2.imwrite(os.path.join(temp_frames_dir, f"{i:05d}.png"), decoded_degraded[i])
        
    smap_arr = np.array(strength_maps_list)
    
    if restorer == 'realesrgan':
        if degradation != 'downsample':
            raise ValueError("RealESRGAN restorer expects 'downsample' degradation")
        from presley.restoration import restore_downsampled_with_realesrgan
        # tile>0 processes the frame in tiles → much lower peak VRAM (lets the job
        # fit alongside other GPU processes); tile=0 is full-frame (fastest, most VRAM).
        restore_downsampled_with_realesrgan(
            temp_frames_dir, restored_frames_dir, smap_arr, block_size,
            denoise_strength=restorer_params.get('denoise_strength', 1.0),
            tile=restorer_params.get('tile', 0),
            tile_pad=restorer_params.get('tile_pad', 10),
            fp32=restorer_params.get('fp32', False))

    elif restorer == 'instantir':
        if degradation != 'blur':
            raise ValueError("InstantIR restorer expects 'blur' degradation")
        from presley.restoration import restore_with_instantir_adaptive
        # batch_size is the main VRAM lever for InstantIR (SDXL-class); drop to 1–2
        # to run alongside other GPU jobs instead of needing a whole free GPU.
        restore_with_instantir_adaptive(
            temp_frames_dir, restored_frames_dir, smap_arr, block_size,
            cfg=restorer_params.get('cfg', 7.0),
            creative_start=restorer_params.get('creative_start', 1.0),
            preview_start=restorer_params.get('preview_start', 0.0),
            batch_size=restorer_params.get('batch_size', 4))
    else:
        raise ValueError(f"Unknown restorer: {restorer}")
        
    # Read restored frames
    import glob
    restored_frames = []
    restored_paths = sorted(glob.glob(os.path.join(restored_frames_dir, "*.png")))
    for f in restored_paths:
        restored_frames.append(cv2.imread(f))
        
    final_output = os.path.join(results_dir, "restored_lossless.mp4")
    save_frames_as_video(restored_frames, final_output, framerate, lossless=True, codec="libx265")
    
    restoration_time = time.time() - restoration_start
    
    import shutil
    shutil.rmtree(temp_frames_dir, ignore_errors=True)
    if os.path.exists(temp_degraded_vid):
        os.remove(temp_degraded_vid)
        
    vid_size = os.path.getsize(transmitted_video)
    maps_size = os.path.getsize(strength_maps_path)
    total_transmitted_bytes = vid_size + maps_size
    duration = len(frames) / framerate
    actual_bitrate = (total_transmitted_bytes * 8) / duration
    
    return {
        "video_frames": len(frames),
        "video_framerate": framerate,
        "output_video": final_output,
        "transmitted_video": transmitted_video,
        "actual_bitrate_bps": actual_bitrate,
        "file_size_bytes": os.path.getsize(final_output),
        "transmitted_size_bytes": total_transmitted_bytes,
        "encoding_time_seconds": encoding_time,
        "restoration_time_seconds": restoration_time,
        "total_time_seconds": encoding_time + restoration_time
    }
