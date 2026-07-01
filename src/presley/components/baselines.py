import os
import time
from typing import Dict, Any
from presley.preprocessing import get_reference_frames
from presley.encode_utils import (
    encode_video_x264, encode_video_x265, 
    encode_video_svtav1, encode_video_kvazaar
)

def run_baseline(experiment: Dict[str, Any], dataset_dir: str, results_dir: str, cache_dir: str) -> Dict[str, Any]:
    video_name = experiment['video']
    width = experiment['width']
    height = experiment['height']
    codec = experiment['codec'].lower()
    target_bitrate = experiment['target_bitrate']
    codec_params = experiment.get('codec_params', {})
    
    # Get reference frames (caches YUV + PNGs)
    raw_yuv_path, frames, framerate = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_pattern = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames", "%05d.png")
    
    output_video = os.path.join(results_dir, "encoded.mp4")
    
    start_time = time.time()
    
    if codec == 'x264':
        encode_video_x264(ref_frames_pattern, output_video, framerate, target_bitrate, preset=codec_params.get('preset', 'medium'))
    elif codec == 'x265':
        encode_video_x265(ref_frames_pattern, output_video, framerate, target_bitrate, preset=codec_params.get('preset', 'medium'))
    elif codec == 'kvazaar':
        encode_video_kvazaar(ref_frames_pattern, output_video, framerate, target_bitrate, width, height)
    elif codec == 'svtav1':
        encode_video_svtav1(ref_frames_pattern, output_video, framerate, target_bitrate, preset=str(codec_params.get('preset', '8')))
    elif codec == 'hnerv':
        raise NotImplementedError("HNeRV baseline not yet implemented")
    elif codec == 'dcvc':
        raise NotImplementedError("DCVC baseline not yet implemented")
    else:
        raise ValueError(f"Unsupported baseline codec: {codec}")
        
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
