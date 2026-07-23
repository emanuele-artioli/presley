import os
import subprocess
import cv2
import math
import numpy as np
from typing import List, Optional, Dict, Any

def derive_rate_control(codec: str, codec_params: Optional[Dict[str, Any]] = None, roi_method: Optional[str] = None) -> str:
    """Classify the rate-control mode an encode_video_*/encode_with_roi_* call
    actually used, as a derived result field (not an experiments.yaml key --
    adding a key would perturb compute_experiment_hash and orphan every
    existing results/<hash>/ dir).

    'qp' in codec_params means different things per codec: x265/kvazaar honor
    it as constant-QP (rate control off); SVT-AV1's baseline/elvis/presley_ai
    path (encode_video_svtav1_qp) passes it as 'rc=0:q=<qp>', which with the
    default aq-mode=2 is equivalent to --crf (see SvtAv1EncApp --help) -- a
    different mechanism with the same "no bitrate target" property. The ROI
    component's own encode_with_roi_* helpers always binary-search a fixed
    QP/CRF regardless of codec_params, so roi_method is checked first.
    """
    codec_params = codec_params or {}
    codec = (codec or '').lower()
    if roi_method is not None:
        roi_method = roi_method.lower()
        if roi_method == 'kvazaar':
            return 'cqp'
        if roi_method == 'svtav1':
            return 'crf'
        if roi_method == 'x265_aq':
            return 'vbr_2pass'
        if roi_method in ('presley_downsample', 'presley_blur', 'presley_noise', 'presley_qp'):
            # These degrade pixels then always encode to a target_bitrate,
            # regardless of any codec_params.qp (that key is a degradation
            # knob here -- filter_frame_qp -- not an encoder rate-control flag).
            presley_codec = codec
            return 'vbr_1pass' if presley_codec == 'kvazaar' else 'vbr_2pass'
        return 'n/a'
    has_qp = 'qp' in codec_params
    if codec == 'x265':
        return 'cqp' if has_qp else 'vbr_2pass'
    if codec == 'x264':
        return 'vbr_2pass'
    if codec == 'kvazaar':
        return 'vbr_1pass'
    if codec == 'svtav1':
        return 'crf' if has_qp else 'vbr_1pass'
    return 'n/a'

def save_frames_as_video(frames: List[np.ndarray], output_path: str, framerate: float, lossless: bool = True, codec: str = "libx265") -> None:
    """Encode frames to video (lossless intermediate by default)."""
    if not frames:
        return
        
    height, width = frames[0].shape[:2]
    
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', str(framerate),
        '-i', '-'
    ]
    
    if lossless and codec == "libx265":
        # NOTE: "lossless" here is lossless only relative to a yuv420p-quantized
        # copy of the input -- chroma subsampling still discards information, so
        # a BGR frame does NOT round-trip bit-exact (verified: ~2.3 mean abs
        # diff/pixel). Fine for intermediates that get lossily re-encoded right
        # after anyway; NOT fine for outputs whose pixels are compared directly
        # (e.g. passthrough-composited results) -- use codec="ffv1" for those.
        cmd.extend(['-c:v', 'libx265', '-preset', 'medium', '-x265-params', 'lossless=1', '-pix_fmt', 'yuv420p'])
    elif lossless and codec == "ffv1":
        # bgr0 = native RGB, no YUV color-matrix conversion or chroma
        # subsampling -> verified bit-exact round-trip (0.0 diff). Requires an
        # .mkv/.avi-style container; ffv1 will not mux into .mp4.
        cmd.extend(['-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0'])
    else:
        cmd.extend(['-c:v', codec])
        
    cmd.append(output_path)
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            if frame is not None:
                process.stdin.write(frame.tobytes())
    finally:
        if process.stdin:
            process.stdin.close()
    
    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg encoding failed for {output_path}")

def load_frames_from_video(video_path: str) -> List[np.ndarray]:
    """Decode video into a list of BGR numpy arrays.

    Decodes straight to a bgr24 rawvideo pipe (same ffmpeg decoder as the old
    temp-PNG path, so frames are bit-identical — PNG is lossless — but without
    writing/reading 82 files, which is slow on NFS). Falls back to the temp-PNG
    path if the pipe decode yields nothing.
    """
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w > 0 and h > 0:
        proc = subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video_path,
             '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
            capture_output=True)
        frame_bytes = h * w * 3
        if proc.returncode == 0 and proc.stdout and len(proc.stdout) % frame_bytes == 0 and proc.stdout:
            arr = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, h, w, 3)
            return [f.copy() for f in arr]

    # Fallback: decode via temp PNGs (handles codecs cv2 can't probe)
    import tempfile
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        decode_video(video_path, tmpdir)
        for f in sorted(os.listdir(tmpdir)):
            img = cv2.imread(os.path.join(tmpdir, f), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
    return frames

def calculate_target_bitrate(width: int, height: int, framerate: float, quality_factor: float=1.0) -> int:
    pixels_per_second = width * height * framerate
    bits_per_pixel = 0.01 * quality_factor
    return int(pixels_per_second * bits_per_pixel)

def encode_video_x265_qp(input_video_or_pattern: str, output_video: str, framerate: float, qp: int, preset: str = "medium") -> None:
    """Encode with libx265 at a fixed QP (rate control off, single pass).

    Fixed-QP is the operating mode where FG-protecting transports actually win
    (VBR rate control partially absorbs their savings — measured repeatedly:
    kvazaar ROI, and the elvis blackout/freeze transports). target_bitrate is
    irrelevant here; compare results on actual_bitrate_bps.
    """
    input_args = ['-i', input_video_or_pattern] if '%' not in input_video_or_pattern else ['-framerate', str(framerate), '-i', input_video_or_pattern]
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *input_args,
                    '-c:v', 'libx265', '-preset', preset, '-x265-params', f'qp={qp}',
                    '-pix_fmt', 'yuv420p', output_video], check=True)

def encode_video_x265(input_video_or_pattern: str, output_video: str, framerate: float, target_bitrate: int, preset: str = "medium", passlog_file: str = "x265_passlog", x265_params: str = None) -> None:
    """Encode video using x265 2-pass."""
    input_args = ['-i', input_video_or_pattern] if '%' not in input_video_or_pattern else ['-framerate', str(framerate), '-i', input_video_or_pattern]
    
    cmd_pass1 = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *input_args,
        '-c:v', 'libx265', '-b:v', str(target_bitrate),
        '-preset', preset, '-x265-params', f'pass=1:stats={passlog_file}' + (f':{x265_params}' if x265_params else ''),
        '-pix_fmt', 'yuv420p',
        '-f', 'null', '/dev/null'
    ]
    subprocess.run(cmd_pass1, check=True)
    
    cmd_pass2 = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *input_args,
        '-c:v', 'libx265', '-b:v', str(target_bitrate),
        '-preset', preset, '-x265-params', f'pass=2:stats={passlog_file}' + (f':{x265_params}' if x265_params else ''),
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    subprocess.run(cmd_pass2, check=True)

def encode_video_x264(input_video_or_pattern: str, output_video: str, framerate: float, target_bitrate: int, preset: str = "medium", passlog_file: str = "x264_passlog") -> None:
    """Two-pass libx264."""
    base_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-r', str(framerate), '-i', input_video_or_pattern]
    
    pass1 = base_cmd + [
        '-c:v', 'libx264', '-b:v', str(target_bitrate),
        '-preset', preset, '-pass', '1', '-passlogfile', passlog_file,
        '-pix_fmt', 'yuv420p', '-f', 'mp4', os.devnull
    ]
    subprocess.run(pass1, check=True)
    
    pass2 = base_cmd + [
        '-c:v', 'libx264', '-b:v', str(target_bitrate),
        '-preset', preset, '-pass', '2', '-passlogfile', passlog_file,
        '-pix_fmt', 'yuv420p', output_video
    ]
    subprocess.run(pass2, check=True)

def encode_video_svtav1(input_video_or_pattern: str, output_video: str, framerate: float, target_bitrate: int, preset: str = "8") -> None:
    """SVT-AV1 encoding."""
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-r', str(framerate), '-i', input_video_or_pattern,
        '-c:v', 'libsvtav1', '-preset', str(preset), '-b:v', str(target_bitrate),
        '-pix_fmt', 'yuv420p', output_video
    ]
    subprocess.run(cmd, check=True)

def encode_video_kvazaar(input_video_or_pattern: str, output_video: str, framerate: float, target_bitrate: int, width: int = 0, height: int = 0) -> None:
    """Kvazaar pure encoding via FFmpeg pipe (assuming rawvideo or similar source, easiest is to pipe to kvazaar)."""
    # For simplicity, decode to yuv420p pipe and send to kvazaar
    # output_video needs to be .hevc, then muxed to mp4
    hevc_out = output_video + ".hevc"
    input_res = f"--input-res {width}x{height}" if width and height else ""
    cmd1 = f"ffmpeg -hide_banner -loglevel error -i {input_video_or_pattern} -f rawvideo -pix_fmt yuv420p - | " \
           f"kvazaar -i - {input_res} --input-fps {framerate} --bitrate {target_bitrate} -o {hevc_out}"
    subprocess.run(cmd1, shell=True, check=True)
    
    cmd2 = f"ffmpeg -hide_banner -loglevel error -y -i {hevc_out} -c copy {output_video}"
    subprocess.run(cmd2, shell=True, check=True)

def scores_to_qp_offsets(removability_scores: np.ndarray, qp_range: int) -> np.ndarray:
    """Map removability scores to bit-neutral per-block QP offsets in [-qp_range, +qp_range].

    Removability scores are heavily skewed low (median ~0.2 on DAVIS), so the naive
    `(score*2-1)*qp_range` mapping yields almost-uniformly negative offsets: the encoder
    is told to spend more bits nearly everywhere, which rate control either fights
    (kvazaar: massive bitrate overshoot) or cancels out (svtav1 CRF search: no FG/BG
    differentiation). Mean-centering each frame makes the offsets bit-neutral by
    construction — bits move from high-removability (background) to low-removability
    (foreground) blocks without changing the total budget.
    """
    centered = removability_scores.astype(np.float32) - removability_scores.mean(axis=(1, 2), keepdims=True).astype(np.float32)
    # Scale by the global (all-frames) max so offsets are comparable across frames;
    # individual frames therefore may not reach the full +/-qp_range.
    max_abs = float(np.abs(centered).max())
    if max_abs < 1e-6:
        return np.zeros_like(centered)
    return np.clip(centered / max_abs * qp_range, -qp_range, qp_range)

def encode_with_roi_kvazaar(input_video_or_pattern: str, output_video: str, removability_scores: np.ndarray, block_size: int, framerate: float, width: int, height: int, target_bitrate: int, qp_range: int = 15) -> None:
    """Encode using Kvazaar --roi (per-CTU delta QP file), fixed-QP mode.

    Kvazaar's VBR rate control (--bitrate) absorbs the ROI deltas: measured on
    bear @460kbps, --roi under VBR left FG-PSNR flat while fixed-QP + --roi
    gained +1.47 dB FG for -0.43 dB BG. So we encode with a fixed base QP
    (--qp, rate control off, ROI honored) and binary-search the base QP whose
    actual bitrate is closest to target_bitrate — same strategy as
    encode_with_roi_svtav1, whose CRF mode exists for the same reason.
    """
    # Background (high removability) -> positive delta QP (fewer bits), FG -> negative
    qp_maps = scores_to_qp_offsets(removability_scores, qp_range)
    
    temp_dir = os.path.dirname(output_video)
    roi_file = os.path.join(temp_dir, "kvazaar_roi.bin")
    
    num_frames, num_blocks_y, num_blocks_x = removability_scores.shape
    ctu_size = max(16, min(64, block_size))
    ctu_cols = math.ceil(width / ctu_size)
    ctu_rows = math.ceil(height / ctu_size)
    
    with open(roi_file, 'wb') as f:
        import struct
        for i in range(num_frames):
            map_frame = qp_maps[i]
            if map_frame.shape != (ctu_rows, ctu_cols):
                map_frame = cv2.resize(map_frame, (ctu_cols, ctu_rows), interpolation=cv2.INTER_AREA)
            
            # Write width and height as 32-bit integers (native byte order)
            f.write(struct.pack('ii', ctu_cols, ctu_rows))
            
            # Write delta QPs as signed 8-bit integers
            delta_qps = np.round(map_frame).astype(np.int8).flatten()
            f.write(delta_qps.tobytes())
                    
    input_args = ['-framerate', str(framerate), '-i', input_video_or_pattern] if '%' in input_video_or_pattern else ['-i', input_video_or_pattern]
    yuv_path = os.path.join(temp_dir, "kvazaar_roi_input.yuv")
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *input_args,
                    '-pix_fmt', 'yuv420p', '-f', 'rawvideo', yuv_path], check=True)

    duration = num_frames / framerate
    hevc_path = output_video + ".hevc"

    def _encode(qp: int) -> float:
        cmd = ['kvazaar', '--input', yuv_path, '--input-res', f'{width}x{height}',
               '--input-fps', str(framerate), '--qp', str(qp), '--roi', roi_file,
               '--output', hevc_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Kvazaar sometimes crashes on shutdown (double-free) after writing a valid
        # stream, so judge success by the output, not the exit code.
        if not os.path.exists(hevc_path) or os.path.getsize(hevc_path) < 1024:
            raise RuntimeError(f"Kvazaar ROI encoding failed (qp={qp}): {result.stderr}")
        return os.path.getsize(hevc_path) * 8 / duration

    try:
        # Binary-search the fixed base QP whose actual bitrate is closest to target
        lo, hi = 1, 51
        best_qp, best_diff, last_qp = None, None, None
        while lo <= hi:
            mid = (lo + hi) // 2
            bitrate = _encode(mid)
            last_qp = mid
            diff = abs(bitrate - target_bitrate)
            if best_diff is None or diff < best_diff:
                best_qp, best_diff = mid, diff
            if bitrate > target_bitrate:
                lo = mid + 1
            else:
                hi = mid - 1
        if last_qp != best_qp:
            _encode(best_qp)

        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'hevc',
                        '-framerate', str(framerate), '-i', hevc_path, '-c:v', 'copy', output_video], check=True)
    finally:
        for path in (hevc_path, yuv_path):
            if os.path.exists(path):
                os.remove(path)

def encode_with_roi_svtav1(input_video_or_pattern: str, output_video: str, removability_scores: np.ndarray, block_size: int, framerate: float, width: int, height: int, target_bitrate: int, qp_range: int = 15, preset: str = "8") -> None:
    """Encode using SVT-AV1 --roi-map-file (per-64x64-superblock QP offsets).

    SVT-AV1 (v1.8.0) silently ignores the ROI map under VBR (--rc 1) — verified
    empirically: identical output with/without the map. In CRF mode the map is
    honored, so we encode with CRF and binary-search the CRF value that best
    approximates target_bitrate (SVT-AV1 at this resolution encodes in seconds,
    so a handful of trial encodes is cheap).
    """
    temp_dir = os.path.dirname(output_video)

    # Decode input to a raw yuv420p file (SvtAv1EncApp input)
    yuv_path = os.path.join(temp_dir, "svtav1_roi_input.yuv")
    input_args = ['-framerate', str(framerate), '-i', input_video_or_pattern] if '%' in input_video_or_pattern else ['-i', input_video_or_pattern]
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *input_args,
                    '-pix_fmt', 'yuv420p', '-f', 'rawvideo', yuv_path], check=True)

    # ROI map: text file, one line per picture: "<picture_number> <offset per superblock...>"
    # in raster order at 64x64 superblock granularity. Background (score=1) -> +qp_range,
    # foreground (score=0) -> -qp_range (same convention as encode_with_roi_kvazaar).
    # AV1 segmentation allows at most 8 distinct segments, so SvtAv1EncApp rejects maps
    # with more than 8 distinct offset values ("Maximum number of segment supported by
    # AV1 spec is eight") — quantize the continuous offsets to a fixed 8-level palette.
    import cv2 as _cv2
    sb_size = 64
    sb_cols = math.ceil(width / sb_size)
    sb_rows = math.ceil(height / sb_size)
    num_frames = removability_scores.shape[0]
    # 7 evenly spaced levels (not 8) so the palette includes 0: with 8 levels a
    # near-zero offset gets pushed to the smallest nonzero level, uniformly
    # shifting flat regions instead of leaving them neutral.
    levels = np.unique(np.round(np.linspace(-qp_range, qp_range, 7)).astype(int))
    qp_maps = scores_to_qp_offsets(removability_scores, qp_range)
    roi_file = os.path.join(temp_dir, "svtav1_roi_map.txt")
    with open(roi_file, 'w') as f:
        for i in range(num_frames):
            raw = _cv2.resize(qp_maps[i], (sb_cols, sb_rows), interpolation=_cv2.INTER_AREA)
            offsets = levels[np.abs(raw.flatten()[:, None] - levels[None, :]).argmin(axis=1)]
            f.write(f"{i} " + " ".join(str(v) for v in offsets) + "\n")

    duration = num_frames / framerate
    fps_num, fps_denom = int(round(framerate * 1000)), 1000
    ivf_path = output_video + ".ivf"

    def _encode(crf: int) -> float:
        cmd = ['SvtAv1EncApp', '-i', yuv_path, '-w', str(width), '-h', str(height),
               '--fps-num', str(fps_num), '--fps-denom', str(fps_denom),
               '--preset', str(preset), '--crf', str(crf),
               '--roi-map-file', roi_file, '-b', ivf_path]
        for _ in range(2):  # rare flaky empty-output runs observed; retry once
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"SvtAv1EncApp failed (crf={crf}): {e.stderr}") from e
            if os.path.getsize(ivf_path) > 1024:
                return os.path.getsize(ivf_path) * 8 / duration
        raise RuntimeError(f"SvtAv1EncApp produced empty output twice (crf={crf})")

    try:
        # Binary-search CRF for the closest actual bitrate to target
        lo, hi = 1, 63
        best_crf, best_diff = None, None
        last_crf = None
        while lo <= hi:
            mid = (lo + hi) // 2
            bitrate = _encode(mid)
            last_crf = mid
            diff = abs(bitrate - target_bitrate)
            if best_diff is None or diff < best_diff:
                best_crf, best_diff = mid, diff
            if bitrate > target_bitrate:
                lo = mid + 1
            else:
                hi = mid - 1
        if last_crf != best_crf:
            _encode(best_crf)

        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', ivf_path,
                        '-c:v', 'copy', output_video], check=True)
    finally:
        # The yuv is large (hundreds of MB); the roi map is kept for diagnostics.
        for path in (ivf_path, yuv_path):
            if os.path.exists(path):
                os.remove(path)

def decode_video(video_path: str, output_dir: str, framerate: float = None, start_number: int = 1, quality: int = 2) -> bool:
    """Decode video to PNG frames. Returns True on success."""
    os.makedirs(output_dir, exist_ok=True)
    decode_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video_path, '-pix_fmt', 'rgb24', '-q:v', str(quality)]
    if framerate is not None:
        decode_cmd.extend(['-r', str(framerate)])
    decode_cmd.extend(['-f', 'image2', '-start_number', str(start_number), '-y', os.path.join(output_dir, '%05d.png')])
    result = subprocess.run(decode_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'Error decoding {video_path}: {result.stderr}')
        return False
    return True


def encode_video_svtav1_qp(input_video_or_pattern: str, output_video: str, framerate: float, qp: int, preset: str = "8") -> None:
    input_args = ['-i', input_video_or_pattern] if '%' not in input_video_or_pattern else ['-framerate', str(framerate), '-i', input_video_or_pattern]
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *input_args,
                    '-c:v', 'libsvtav1', '-preset', preset, '-svtav1-params', f'rc=0:q={qp}', output_video], check=True)

