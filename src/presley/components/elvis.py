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
    # Speed/quality knobs for the in-painter (forwarded to the restoration fn).
    inpainter_params = experiment.get('inpainter_params', {})
    
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
        shrunk, binary_mask, _ = apply_selective_removal(frame, score, block_size, shrink_amount, cluster_blocks=True)
        shrunk_frames_list.append(shrunk)
        masks_list.append(binary_mask)
        
    # Save uncompressed shrunk frames temporarily for encoding
    temp_shrunk_vid = os.path.join(results_dir, "temp_shrunk_lossless.mkv")
    save_frames_as_video(shrunk_frames_list, temp_shrunk_vid, framerate, lossless=True, codec="libx265")
    
    # 3. Encode shrunk video
    encoded_shrunk = os.path.join(results_dir, "encoded_shrunk.mp4")
    # For now, we reuse x265 encoding from baselines for the shrunk video
    if codec == 'x265':
        encode_video_x265(temp_shrunk_vid, encoded_shrunk, framerate, target_bitrate, preset=codec_params.get('preset', 'medium'))
    else:
        raise ValueError(f"Elvis currently requires x265 for encoding, got {codec}")
        
    encoding_time = time.time() - start_time
    restoration_start = time.time()
    
    # Save masks (transmitted side information)
    masks_path = os.path.join(results_dir, "removal_masks.npz")
    np.savez_compressed(masks_path, masks=np.array(masks_list))
    
    # 4. Decode shrunk video
    decoded_shrunk = load_frames_from_video(encoded_shrunk)
    
    # 5. Stretch
    stretched_frames_list = []
    for i in range(len(decoded_shrunk)):
        stretched = stretch_frame(decoded_shrunk[i], masks_list[i], block_size)
        stretched_frames_list.append(stretched)
        
    # Save stretched to disk as PNGs because propainter/e2fgvi expect directories of PNGs
    stretched_dir = os.path.join(results_dir, "stretched_frames")
    masks_dir = os.path.join(results_dir, "masks")
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
    output_frames_dir = os.path.join(results_dir, "restored_frames")
    
    if inpainter == 'propainter':
        from presley.restoration import inpaint_with_propainter
        # Forward only the knobs inpaint_with_propainter accepts; defaults preserved.
        pp_keys = ('ref_stride', 'neighbor_length', 'subvideo_length', 'raft_iter', 'fp16', 'resize_ratio')
        pp_kwargs = {k: inpainter_params[k] for k in pp_keys if k in inpainter_params}
        inpaint_with_propainter(stretched_dir, masks_dir, output_frames_dir, width, height, framerate, mask_dilation=0, **pp_kwargs)
    elif inpainter == 'e2fgvi':
        from presley.restoration import inpaint_with_e2fgvi
        e2_keys = ('ref_stride', 'neighbor_stride', 'num_ref')
        e2_kwargs = {k: inpainter_params[k] for k in e2_keys if k in inpainter_params}
        inpaint_with_e2fgvi(stretched_dir, masks_dir, output_frames_dir, width, height, framerate, **e2_kwargs)
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
    vid_size = os.path.getsize(encoded_shrunk)
    masks_size = os.path.getsize(masks_path)
    total_transmitted_bytes = vid_size + masks_size
    duration = len(frames) / framerate
    actual_bitrate = (total_transmitted_bytes * 8) / duration
    
    return {
        "video_frames": len(frames),
        "video_framerate": framerate,
        "output_video": final_output,
        "transmitted_video": encoded_shrunk,
        "actual_bitrate_bps": actual_bitrate,
        "file_size_bytes": os.path.getsize(final_output),
        "transmitted_size_bytes": total_transmitted_bytes,
        "encoding_time_seconds": encoding_time,
        "restoration_time_seconds": restoration_time,
        "total_time_seconds": encoding_time + restoration_time
    }
