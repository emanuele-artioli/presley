import os
import time
from typing import Dict, Any
from presley.preprocessing import get_reference_frames
from presley.stagetiming import StageTimer
from presley.encode_utils import (
    encode_video_x264, encode_video_x265,
    encode_video_svtav1, encode_video_kvazaar, derive_rate_control
)

def run_baseline(experiment: Dict[str, Any], dataset_dir: str, results_dir: str, cache_dir: str) -> Dict[str, Any]:
    video_name = experiment['video']
    width = experiment['width']
    height = experiment['height']
    codec = experiment['codec'].lower()
    target_bitrate = experiment['target_bitrate']
    codec_params = experiment.get('codec_params', {})
    
    stages = StageTimer()

    # Get reference frames (caches YUV + PNGs). Cached across experiments, so a
    # run hitting a warm cache reports ~0 here -- see stagetiming.CACHED_STAGES.
    with stages('preprocess'):
        raw_yuv_path, frames, framerate = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_pattern = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames", "%05d.png")

    output_video = os.path.join(results_dir, "encoded.mp4")

    start_time = time.time()
    encode_start = time.time()

    if codec == 'x264':
        encode_video_x264(ref_frames_pattern, output_video, framerate, target_bitrate, preset=codec_params.get('preset', 'medium'))
    elif codec == 'x265':
        if 'qp' in codec_params:
            # Fixed-QP mode (rate control off) — the operating mode where
            # FG-protecting transports win; compare on actual_bitrate_bps.
            from presley.encode_utils import encode_video_x265_qp
            encode_video_x265_qp(ref_frames_pattern, output_video, framerate, int(codec_params['qp']), preset=codec_params.get('preset', 'medium'))
        else:
            encode_video_x265(ref_frames_pattern, output_video, framerate, target_bitrate, preset=codec_params.get('preset', 'medium'))
    elif codec == 'kvazaar':
        if 'qp' in codec_params or codec_params.get('rate_control') == 'cqp':
            # Fixed-QP kvazaar. Without this branch the only kvazaar baseline
            # obtainable was --bitrate (VBR), which overshoots targets by
            # 30-45% -- so tab:roi's fixed-QP ROI arms had nothing like-for-like
            # to be compared against. rate_control='cqp' binary-searches the
            # base QP toward target_bitrate, matching encode_with_roi_kvazaar.
            from presley.encode_utils import encode_video_kvazaar_qp
            _qp = codec_params.get('qp')
            encode_video_kvazaar_qp(
                ref_frames_pattern, output_video, framerate, width, height,
                target_bitrate=None if _qp is not None else target_bitrate,
                qp=None if _qp is None else int(_qp),
                num_frames=len(frames))
        else:
            encode_video_kvazaar(ref_frames_pattern, output_video, framerate, target_bitrate, width, height)
    elif codec == 'svtav1':
        if 'qp' in codec_params:
            from presley.encode_utils import encode_video_svtav1_qp
            _tune = codec_params.get('tune')
            encode_video_svtav1_qp(ref_frames_pattern, output_video, framerate, int(codec_params['qp']), preset=str(codec_params.get('preset', '8')), tune=None if _tune is None else int(_tune))
        else:
            encode_video_svtav1(ref_frames_pattern, output_video, framerate, target_bitrate, preset=str(codec_params.get('preset', '8')))
    elif codec == 'hnerv':
        from presley.hnerv_utils import encode_video_hnerv
        checkpoint_path = os.path.join(results_dir, "hnerv_checkpoint.pt.gz")
        # Returns train_seconds but we keep track of total encoding_time using start_time anyway
        encode_video_hnerv(ref_frames_pattern, output_video, framerate, width, height, codec_params, checkpoint_path)
    elif codec == 'dcvc':
        raise NotImplementedError("DCVC baseline not yet implemented")
    else:
        raise ValueError(f"Unsupported baseline codec: {codec}")
        
    stages.add('encode', time.time() - encode_start)
    encoding_time = time.time() - start_time

    if codec == 'hnerv':
        file_size = os.path.getsize(os.path.join(results_dir, "hnerv_checkpoint.pt.gz"))
    else:
        file_size = os.path.getsize(output_video)
        
    duration = len(frames) / framerate
    actual_bitrate = (file_size * 8) / duration
    
    return {
        "video_frames": len(frames),
        "video_framerate": framerate,
        "output_video": output_video,
        "rate_control": derive_rate_control(codec, codec_params),
        "actual_bitrate_bps": actual_bitrate,
        "file_size_bytes": file_size,
        "transmitted_size_bytes": file_size,
        "encoding_time_seconds": encoding_time,
        "restoration_time_seconds": 0.0,
        "total_time_seconds": encoding_time,
        "stage_times_seconds": stages.as_dict(),
    }
