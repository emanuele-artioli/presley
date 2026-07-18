import os
import time
import numpy as np
from typing import Dict, Any

from presley.preprocessing import get_reference_frames, get_removability_scores
from presley.encode_utils import save_frames_as_video, load_frames_from_video, encode_video_x265, encode_video_x265_qp, encode_video_svtav1_qp, encode_video_svtav1, derive_rate_control
from presley.degradation import (filter_frame_downsample, filter_frame_gaussian,
                                 filter_frame_noise,
                                 filter_frame_mean_fill, filter_frame_freeze,
                                 select_removal_mask_global)
from presley.sidechannel import save_binary_masks, composite_passthrough

# Degradations that punch holes to be filled by an in-painter (the ELVIS<->PRESLEY
# bridge), rather than blur/downsample restored by a super-resolver.
INPAINT_DEGRADATIONS = ('mean_fill', 'freeze')

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
    # Passthrough compositing (default on): emit the decoded transmitted pixels
    # everywhere and restored pixels only inside the degraded region, so the
    # untouched foreground is reproduced bit-exact instead of re-encoded through
    # the restorer. Set composite_output: false for the raw restorer frames.
    composite_output = experiment.get('composite_output', True)
    # Budgeted selection for the bridge degradations (mean_fill/freeze): when
    # shrink_amount is set, select blocks with elvis's global top-k (same
    # removal budget -> same starved operating point) instead of the
    # round(score)>0 threshold, which degrades too few blocks (9.4% on bear ->
    # 844 kbps, the comfortable regime where nothing can win). fg_protect adds
    # the hard UFO-mask exclusion, same as elvis.
    select_amount = experiment.get('shrink_amount')
    fg_protect = experiment.get('fg_protect', False)
    temporal_pool_masks = experiment.get('temporal_pool_masks', False)
    
    # 1. Load data
    raw_yuv_path, frames, framerate = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    removability_scores = get_removability_scores(video_name, width, height, block_size, alpha, beta, dataset_dir, cache_dir)
    
    start_time = time.time()
    
    # 2. Degrade
    degraded_frames_list = []
    strength_maps_list = []
    frames_arr = np.array(frames)
    
    # Block-level FG masks for hard protection (same recipe as elvis).
    fg_block_masks = None
    if fg_protect and degradation in INPAINT_DEGRADATIONS:
        import cv2 as _cv2
        from presley.preprocessing import get_ufo_masks
        ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
        ufo = get_ufo_masks(video_name, width, height, block_size, ref_frames_dir, cache_dir, temporal_pool=temporal_pool_masks)
        nby, nbx = height // block_size, width // block_size
        fg_block_masks = []
        for m in ufo:
            if m.shape != (height, width):
                m = _cv2.resize(m, (width, height), interpolation=_cv2.INTER_NEAREST)
            fg_block_masks.append(m[:nby * block_size, :nbx * block_size]
                                  .reshape(nby, block_size, nbx, block_size).max(axis=(1, 3)) > 127)

    prev_degraded = None
    for i in range(len(frames)):
        frame = frames_arr[i]
        score = removability_scores[i]

        sel = None
        if select_amount is not None and degradation in INPAINT_DEGRADATIONS:
            excl = fg_block_masks[i] if fg_block_masks is not None and i < len(fg_block_masks) else None
            sel = select_removal_mask_global(score, select_amount, cluster_blocks=True, exclude=excl) > 0

        if degradation == 'downsample':
            degraded, smap = filter_frame_downsample(frame, score, block_size)
        elif degradation == 'blur':
            degraded, smap = filter_frame_gaussian(frame, score, block_size)
        elif degradation == 'noise':
            degraded, smap = filter_frame_noise(frame, score, block_size)
        elif degradation == 'mean_fill':
            degraded, smap = filter_frame_mean_fill(frame, score, block_size, sel=sel)
        elif degradation == 'freeze':
            degraded, smap = filter_frame_freeze(frame, score, block_size, prev_degraded, sel=sel)
        else:
            raise ValueError(f"Unknown degradation: {degradation}")

        prev_degraded = degraded
        degraded_frames_list.append(degraded)
        strength_maps_list.append(smap)
        
    temp_degraded_vid = os.path.join(results_dir, "temp_degraded_lossless.mkv")
    save_frames_as_video(degraded_frames_list, temp_degraded_vid, framerate, lossless=True, codec="libx265")
    
    # 3. Encode degraded frames
    transmitted_video = os.path.join(results_dir, "encoded_degraded.mp4")
    if codec == 'x265':
        if 'qp' in codec_params:
            encode_video_x265_qp(temp_degraded_vid, transmitted_video, framerate, int(codec_params['qp']), preset=codec_params.get('preset', 'medium'))
        else:
            encode_video_x265(temp_degraded_vid, transmitted_video, framerate, target_bitrate, preset=codec_params.get('preset', 'medium'))
    elif codec == 'svtav1':
        if 'qp' in codec_params:
            encode_video_svtav1_qp(temp_degraded_vid, transmitted_video, framerate, int(codec_params['qp']), preset=codec_params.get('preset', '8'))
        else:
            encode_video_svtav1(temp_degraded_vid, transmitted_video, framerate, target_bitrate, preset=codec_params.get('preset', '8'))
    else:
        raise ValueError(f"Presley AI currently requires x265 or svtav1 for encoding, got {codec}")

    encoding_time = time.time() - start_time
    restoration_start = time.time()

    # Save strength maps (transmitted side information). Binary hole maps
    # (mean_fill/freeze) are bit-packed; multi-level maps stay savez'd.
    strength_maps_path = os.path.join(results_dir, "strength_maps.npz")
    if degradation in INPAINT_DEGRADATIONS:
        save_binary_masks(strength_maps_list, strength_maps_path)
    else:
        np.savez_compressed(strength_maps_path, strength_maps=np.array(strength_maps_list))
    
    # 4. Decode degraded frames
    decoded_degraded = load_frames_from_video(transmitted_video)

    import cv2

    # 5. Restore. restorer == 'none' is a Goal-1-only screen: skip the PNG
    # round-trip and any restorer entirely, so bitrate/FG-quality can be
    # measured with zero GPU cost before committing to a real restoration run.
    if restorer == 'none':
        restored_frames = list(decoded_degraded)
        restoration_time = 0.0
    else:
        temp_frames_dir = os.path.join(results_dir, "temp_restoring")
        restored_frames_dir = os.path.join(results_dir, "restored_frames")
        os.makedirs(temp_frames_dir, exist_ok=True)
        os.makedirs(restored_frames_dir, exist_ok=True)

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
        elif restorer in ('propainter', 'e2fgvi', 'telea'):
            # In-painting restorers for the hole-punching degradations (mean_fill /
            # freeze): fill the degraded region rather than super-resolve it.
            if degradation not in INPAINT_DEGRADATIONS:
                raise ValueError(f"{restorer} restorer expects a hole degradation {INPAINT_DEGRADATIONS}, got '{degradation}'")
            masks_dir = os.path.join(results_dir, "temp_masks")
            os.makedirs(masks_dir, exist_ok=True)
            for i in range(len(strength_maps_list)):
                m_full = cv2.resize((strength_maps_list[i] > 0).astype(np.uint8) * 255,
                                    (width, height), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(masks_dir, f"{i:05d}.png"), m_full)
            if restorer == 'propainter':
                from presley.restoration import inpaint_with_propainter
                pp_keys = ('ref_stride', 'neighbor_length', 'subvideo_length', 'raft_iter', 'fp16', 'resize_ratio')
                pp_kwargs = {k: restorer_params[k] for k in pp_keys if k in restorer_params}
                inpaint_with_propainter(temp_frames_dir, masks_dir, restored_frames_dir, width, height, framerate, mask_dilation=0, **pp_kwargs)
            elif restorer == 'e2fgvi':
                # Same in-painter set as elvis, so the fill x restorer grid is
                # symmetric across the two components (Goal-2 probe).
                from presley.restoration import inpaint_with_e2fgvi
                e2_keys = ('ref_stride', 'neighbor_stride', 'num_ref')
                e2_kwargs = {k: restorer_params[k] for k in e2_keys if k in restorer_params}
                inpaint_with_e2fgvi(temp_frames_dir, masks_dir, restored_frames_dir, width, height, framerate, **e2_kwargs)
            else:  # telea (classical CPU in-painting)
                radius = int(restorer_params.get('inpaint_radius', 3))
                for i in range(len(decoded_degraded)):
                    m = cv2.imread(os.path.join(masks_dir, f"{i:05d}.png"), 0)
                    inp = cv2.inpaint(decoded_degraded[i], (m > 127).astype(np.uint8), radius, cv2.INPAINT_TELEA)
                    cv2.imwrite(os.path.join(restored_frames_dir, f"{i:05d}.png"), inp)
            import shutil as _shutil
            _shutil.rmtree(masks_dir, ignore_errors=True)
        else:
            raise ValueError(f"Unknown restorer: {restorer}")

        # Read restored frames
        import glob
        restored_frames = []
        restored_paths = sorted(glob.glob(os.path.join(restored_frames_dir, "*.png")))
        for f in restored_paths:
            restored_frames.append(cv2.imread(f))

        restoration_time = time.time() - restoration_start
        import shutil
        shutil.rmtree(temp_frames_dir, ignore_errors=True)

    # Passthrough compositing: keep the decoded transmitted pixels (bit-exact FG)
    # and take restored pixels only where the frame was degraded (strength > 0).
    if composite_output:
        pix_masks = [cv2.resize((strength_maps_list[i] > 0).astype(np.uint8), (width, height),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                     for i in range(len(restored_frames))]
        restored_frames = composite_passthrough(decoded_degraded, restored_frames, pix_masks)

    # ffv1/bgr0 (verified bit-exact, unlike libx265's yuv420p "lossless" which
    # still chroma-subsamples): matters here because composited pixels are
    # compared directly against reference frames for the FG-quality claim.
    final_output = os.path.join(results_dir, "restored_lossless.mkv")
    save_frames_as_video(restored_frames, final_output, framerate, lossless=True, codec="ffv1")

    import shutil
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
        "rate_control": derive_rate_control(codec, codec_params),
        "actual_bitrate_bps": actual_bitrate,
        "file_size_bytes": os.path.getsize(final_output),
        "transmitted_size_bytes": total_transmitted_bytes,
        "encoding_time_seconds": encoding_time,
        "restoration_time_seconds": restoration_time,
        "total_time_seconds": encoding_time + restoration_time
    }
