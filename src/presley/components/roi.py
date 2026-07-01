import os
import time
from typing import Dict, Any
import numpy as np
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
        # x265-params takes aq-mode and aq-strength
        # Actually, encode_video_x265 has an x265_params kwarg? Wait, encode_video_x265 signature doesn't take kwargs...
        # Let's import it to check, but since I'll use run_command later to fix encode_video_x265 if needed...
        # Wait, I will just call encode_video_x265 and it will need x265_params parameter!
        encode_video_x265(ref_frames_pattern, output_video, framerate, target_bitrate, preset=preset, x265_params=f"aq-mode={aq_mode}:aq-strength={aq_strength}")
        
    elif roi_method == 'presley_qp':
        encode_with_roi_kvazaar(ref_frames_pattern, output_video, removability_scores, block_size, framerate, width, height, target_bitrate, qp_range=codec_params.get('qp_range', 15))
        
    elif roi_method in ['presley_downsample', 'presley_blur', 'presley_noise']:
        # Apply degradation standalone, then encode lossy
        from presley.encode_utils import save_frames_as_video
        from presley.degradation import filter_frame_downsample, filter_frame_gaussian, filter_frame_noise
        
        frames_arr = np.array(frames)
        degraded_frames_list = []
        
        for i in range(len(frames)):
            frame = frames_arr[i]
            score = removability_scores[i]
            if roi_method == 'presley_downsample':
                degraded, _ = filter_frame_downsample(frame, score, block_size)
            elif roi_method == 'presley_blur':
                degraded, _ = filter_frame_gaussian(frame, score, block_size)
            else:
                degraded, _ = filter_frame_noise(frame, score, block_size)
            degraded_frames_list.append(degraded)
            
        temp_degraded_vid = os.path.join(results_dir, "temp_roi_presley.mkv")
        save_frames_as_video(degraded_frames_list, temp_degraded_vid, framerate, lossless=True, codec="libx265")
        
        # Now encode it using the target codec (default to x265 if not specified)
        presley_codec = experiment.get('codec', 'x265')
        if presley_codec == 'x265':
            from presley.encode_utils import encode_video_x265
            encode_video_x265(temp_degraded_vid, output_video, framerate, target_bitrate, preset=codec_params.get('preset', 'medium'))
        elif presley_codec == 'kvazaar':
            from presley.encode_utils import encode_video_kvazaar
            encode_video_kvazaar(temp_degraded_vid, output_video, framerate, target_bitrate, width, height)
        else:
            raise ValueError(f"Codec {presley_codec} not supported for standalone presley ROI")
            
        if os.path.exists(temp_degraded_vid):
            os.remove(temp_degraded_vid)
            
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
