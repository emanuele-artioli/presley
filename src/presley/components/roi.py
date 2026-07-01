import os
import time
from typing import Dict, Any
from presley.preprocessing import get_reference_frames, get_removability_scores
from presley.encode_utils import (
    encode_with_roi_kvazaar, 
    # other roi encoding functions can be imported here once implemented
)

def run_roi(experiment: Dict[str, Any], dataset_dir: str, results_dir: str, cache_dir: str) -> Dict[str, Any]:
    video_name = experiment['video']
    width = experiment['width']
    height = experiment['height']
    block_size = experiment['block_size']
    alpha = experiment['alpha']
    beta = experiment['beta']
    
    roi_method = experiment['roi_method'].lower()
    target_bitrate = experiment['target_bitrate']
    codec_params = experiment.get('codec_params', {})
    
    # Get reference frames and removability scores
    raw_yuv_path, frames, framerate = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_pattern = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames", "%05d.png")
    
    removability_scores = get_removability_scores(video_name, width, height, block_size, alpha, beta, dataset_dir, cache_dir)
    
    output_video = os.path.join(results_dir, "encoded.mp4")
    
    start_time = time.time()
    
    if roi_method == 'kvazaar':
        qp_range = codec_params.get('qp_range', 15)
        encode_with_roi_kvazaar(ref_frames_pattern, output_video, removability_scores, block_size, framerate, width, height, target_bitrate, qp_range=qp_range)
    elif roi_method == 'x265_aq':
        # AQ doesn't use the removability scores directly, relies on encoder
        from presley.encode_utils import encode_video_x265
        aq_mode = codec_params.get('aq_mode', 1)
        aq_strength = codec_params.get('aq_strength', 1.0)
        preset = codec_params.get('preset', 'medium')
        # We need to pass AQ params in x265-params
        # Will add this capability to encode_video_x265 via a kwargs or modifying it later.
        raise NotImplementedError("x265_aq baseline not yet implemented")
    elif roi_method in ['svtav1', 'x264_addroi', 'x265_addroi']:
        raise NotImplementedError(f"{roi_method} ROI not yet fully implemented")
    else:
        raise ValueError(f"Unsupported ROI method: {roi_method}")
        
    encoding_time = time.time() - start_time
    
    file_size = os.path.getsize(output_video)
    duration = len(frames) / framerate
    actual_bitrate = (file_size * 8) / duration
    
    return {
        "video_frames": len(frames),
        "video_framerate": framerate,
        "output_video": output_video,
        "actual_bitrate_bps": actual_bitrate,
        "file_size_bytes": file_size,
        "transmitted_size_bytes": file_size,
        "encoding_time_seconds": encoding_time,
        "restoration_time_seconds": 0.0,
        "total_time_seconds": encoding_time
    }
