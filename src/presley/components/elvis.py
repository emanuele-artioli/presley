import os
import time
import numpy as np
from typing import Dict, Any

from presley.preprocessing import get_reference_frames, get_removability_scores
from presley.encode_utils import save_frames_as_video, load_frames_from_video, encode_video_x265
from presley.degradation import apply_selective_removal
from presley.restoration import stretch_frame

def run_elvis(experiment: Dict[str, Any], dataset_dir: str, results_dir: str, cache_dir: str) -> Dict[str, Any]:
    video_name = experiment['video']
    width = experiment['width']
    height = experiment['height']
    block_size = experiment['block_size']
    alpha = experiment['alpha']
    beta = experiment['beta']
    shrink_amount = experiment['shrink_amount']
    inpainter = experiment['inpainter'].lower()
    
    codec = experiment['codec'].lower()
    target_bitrate = experiment['target_bitrate']
    codec_params = experiment.get('codec_params', {})
    
    # 1. Load data
    raw_yuv_path, frames, framerate = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    removability_scores = get_removability_scores(video_name, width, height, block_size, alpha, beta, dataset_dir, cache_dir)
    
    start_time = time.time()
    
    # 2. Shrink
    shrunk_frames_list = []
    masks_list = []
    frames_arr = np.array(frames)
    
    for i in range(len(frames)):
        frame = frames_arr[i]
        score = removability_scores[i]
        shrunk, binary_mask, _ = apply_selective_removal(frame, score, block_size, shrink_amount)
        shrunk_frames_list.append(shrunk)
        masks_list.append(binary_mask)
        
    # Save uncompressed shrunk frames temporarily for encoding
    temp_shrunk_vid = os.path.join(results_dir, "temp_shrunk_lossless.mkv")
    save_frames_as_video(shrunk_frames_list, temp_shrunk_vid, framerate, lossless=True, codec="libx265")
    
    # 3. Encode shrunk frames
    transmitted_video = os.path.join(results_dir, "encoded_shrunk.mp4")
    if codec == 'x265':
        encode_video_x265(temp_shrunk_vid, transmitted_video, framerate, target_bitrate, preset=codec_params.get('preset', 'medium'))
    else:
        raise ValueError(f"Elvis currently requires x265 for encoding, got {codec}")
        
    encoding_time = time.time() - start_time
    restoration_start = time.time()
    
    # Save masks (transmitted side information)
    masks_path = os.path.join(results_dir, "removal_masks.npz")
    np.savez_compressed(masks_path, masks=np.array(masks_list))
    
    # 4. Decode shrunk frames
    decoded_shrunk = load_frames_from_video(transmitted_video)
    
    # 5. Stretch
    stretched_frames_list = []
    for i in range(len(decoded_shrunk)):
        stretched = stretch_frame(decoded_shrunk[i], masks_list[i], block_size)
        stretched_frames_list.append(stretched)
        
    # Save stretched to disk as PNGs because propainter/e2fgvi expect directories of PNGs
    stretched_dir = os.path.join(results_dir, "temp_stretched")
    masks_dir = os.path.join(results_dir, "temp_masks")
    os.makedirs(stretched_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    import cv2
    for i in range(len(stretched_frames_list)):
        cv2.imwrite(os.path.join(stretched_dir, f"{i:05d}.png"), stretched_frames_list[i])
        # Inpainting models usually expect mask where 255 is the region to inpaint
        inpainting_mask = (masks_list[i] * 255).astype(np.uint8)
        # resize mask to full resolution
        inpainting_mask_full = cv2.resize(inpainting_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(masks_dir, f"{i:05d}.png"), inpainting_mask_full)
        
    # 6. Inpaint
    output_frames_dir = os.path.join(results_dir, "temp_inpainted")
    os.makedirs(output_frames_dir, exist_ok=True)
    
    if inpainter == 'propainter':
        from presley.restoration import inpaint_with_propainter
        inpaint_with_propainter(stretched_dir, masks_dir, output_frames_dir, width, height, framerate)
    elif inpainter == 'e2fgvi':
        from presley.restoration import inpaint_with_e2fgvi
        inpaint_with_e2fgvi(stretched_dir, masks_dir, output_frames_dir, width, height, framerate)
    else:
        raise ValueError(f"Unknown inpainter: {inpainter}")
        
    # Load inpainted frames and save as lossless video
    inpainted_frames = []
    for i in range(len(stretched_frames_list)):
        fpath = os.path.join(output_frames_dir, f"{i+1:05d}.png") # Propainter uses 1-based indexing for output? Or we can just sort
    
    # Just read them sorted
    import glob
    out_files = sorted(glob.glob(os.path.join(output_frames_dir, "*.png")))
    for f in out_files:
        inpainted_frames.append(cv2.imread(f))
        
    final_output = os.path.join(results_dir, "restored_lossless.mp4")
    save_frames_as_video(inpainted_frames, final_output, framerate, lossless=True, codec="libx265")
    
    restoration_time = time.time() - restoration_start
    
    # Clean up temp dirs
    import shutil
    shutil.rmtree(stretched_dir, ignore_errors=True)
    shutil.rmtree(masks_dir, ignore_errors=True)
    shutil.rmtree(output_frames_dir, ignore_errors=True)
    if os.path.exists(temp_shrunk_vid):
        os.remove(temp_shrunk_vid)
        
    # Bitrate calculation
    vid_size = os.path.getsize(transmitted_video)
    masks_size = os.path.getsize(masks_path)
    total_transmitted_bytes = vid_size + masks_size
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
