import os
import time
import numpy as np
from typing import Dict, Any

from presley.preprocessing import get_reference_frames, get_removability_scores
from presley.encode_utils import save_frames_as_video, load_frames_from_video, encode_video_x265
from presley.degradation import apply_selective_removal, select_removal_mask_global
from presley.restoration import stretch_frame
from presley.sidechannel import save_binary_masks, composite_passthrough

def run_elvis(experiment: Dict[str, Any], dataset_dir: str, results_dir: str, cache_dir: str) -> Dict[str, Any]:
    video_name = experiment['video']
    width = experiment['width']
    height = experiment['height']
    block_size = experiment['block_size']
    alpha = experiment['alpha']
    beta = experiment['beta']
    shrink_amount = experiment['shrink_amount']
    inpainter = experiment['inpainter'].lower()
    # How removed blocks are represented in the transmitted video:
    #   shrink   - remove blocks and pack rows (original ELVIS; breaks temporal
    #              prediction: +193% bitrate overshoot at low targets)
    #   blackout - keep native resolution, set removed blocks to black (flat DC,
    #              motion vectors of surviving blocks stay valid)
    #   freeze   - keep native resolution; removed blocks copy the co-located
    #              region of the previous degraded frame (inter skip ~ 0 bits;
    #              frame 0 keeps the original content so the I-frame carries a
    #              real-texture prior for the in-painter)
    removal_mode = experiment.get('removal_mode', 'shrink').lower()
    # Passthrough compositing (default on): emit decoded transmitted pixels
    # everywhere and in-painted pixels only inside the holes, so foreground
    # quality is reproduced bit-exact instead of re-encoded through the
    # in-painter. Set composite_output: false to get the raw in-painter frames.
    composite_output = experiment.get('composite_output', True)
    # Hard foreground protection (blackout/freeze only): never remove a block
    # the UFO mask marks as foreground. The removability score protects FG only
    # softly (BG x10), which fails on high-motion foregrounds (bmx-trees: FG
    # outscored boosted BG and 12.8% of FG blocks were removed -> -3 dB FG).
    # Off by default so pre-existing results stay reproducible; new entries
    # should set fg_protect: true explicitly.
    fg_protect = experiment.get('fg_protect', False)

    codec = experiment['codec'].lower()
    target_bitrate = experiment['target_bitrate']
    codec_params = experiment.get('codec_params', {})
    # Speed/quality knobs for the in-painter (forwarded to the restoration fn).
    inpainter_params = experiment.get('inpainter_params', {})
    
    # 1. Load data
    raw_yuv_path, frames, framerate = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    removability_scores = get_removability_scores(video_name, width, height, block_size, alpha, beta, dataset_dir, cache_dir)
    
    start_time = time.time()

    # 2. Remove blocks (mode-dependent representation)
    import cv2
    shrunk_frames_list = []
    masks_list = []
    frames_arr = np.array(frames)

    # Block-level foreground masks for hard FG protection (cached UFO masks,
    # max-pooled to the block grid: a block is FG if it contains any FG pixel).
    fg_block_masks = None
    if fg_protect and removal_mode in ('blackout', 'freeze'):
        from presley.preprocessing import get_ufo_masks
        ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
        ufo = get_ufo_masks(video_name, width, height, block_size, ref_frames_dir, cache_dir)
        nby, nbx = height // block_size, width // block_size
        fg_block_masks = []
        for m in ufo:
            if m.shape != (height, width):
                m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
            fg_block_masks.append(m[:nby * block_size, :nbx * block_size]
                                  .reshape(nby, block_size, nbx, block_size).max(axis=(1, 3)) > 127)

    prev_degraded = None
    for i in range(len(frames)):
        frame = frames_arr[i]
        score = removability_scores[i]
        if removal_mode == 'shrink':
            # shrink packs surviving blocks into a rectangle, which requires an
            # equal removed-count per row -> per-row selection.
            shrunk, binary_mask, _ = apply_selective_removal(frame, score, block_size, shrink_amount, cluster_blocks=True)
            masks_list.append(binary_mask)
            shrunk_frames_list.append(shrunk)
        elif removal_mode in ('blackout', 'freeze'):
            # Native geometry removes the packing constraint: pick the globally
            # top-k most-removable blocks instead of an equal count per row.
            excl = fg_block_masks[i] if fg_block_masks is not None and i < len(fg_block_masks) else None
            binary_mask = select_removal_mask_global(score, shrink_amount, cluster_blocks=True, exclude=excl)
            masks_list.append(binary_mask)
            pix_mask = cv2.resize(binary_mask.astype(np.uint8), (width, height),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            degraded = frame.copy()
            if removal_mode == 'blackout':
                degraded[pix_mask] = 0
            else:  # freeze: hold previous degraded content; frame 0 keeps original
                if prev_degraded is not None:
                    degraded[pix_mask] = prev_degraded[pix_mask]
            prev_degraded = degraded
            shrunk_frames_list.append(degraded)
        else:
            raise ValueError(f"Unknown removal_mode: {removal_mode}")

    # Save uncompressed degraded frames temporarily for encoding
    temp_shrunk_vid = os.path.join(results_dir, "temp_shrunk_lossless.mkv")
    save_frames_as_video(shrunk_frames_list, temp_shrunk_vid, framerate, lossless=True, codec="libx265")
    
    # 3. Encode transmitted video
    encoded_shrunk = os.path.join(results_dir, "encoded_shrunk.mp4")
    if codec == 'x265':
        if 'qp' in codec_params:
            # Fixed-QP mode: where the blackout/freeze transports actually win
            # (+0.55-0.59 dB FG at matched bitrate; VBR partially absorbs it).
            from presley.encode_utils import encode_video_x265_qp
            encode_video_x265_qp(temp_shrunk_vid, encoded_shrunk, framerate, int(codec_params['qp']), preset=codec_params.get('preset', 'medium'))
        else:
            encode_video_x265(temp_shrunk_vid, encoded_shrunk, framerate, target_bitrate, preset=codec_params.get('preset', 'medium'))
    else:
        raise ValueError(f"Elvis currently requires x265 for encoding, got {codec}")
        
    encoding_time = time.time() - start_time
    restoration_start = time.time()
    
    # Save masks (transmitted side information), bit-packed to minimise the
    # fixed side-channel cost that dominates the starved-bitrate budget.
    masks_path = os.path.join(results_dir, "removal_masks.npz")
    save_binary_masks(masks_list, masks_path)
    
    # 4. Decode transmitted video
    decoded_shrunk = load_frames_from_video(encoded_shrunk)

    # 5. Restore native geometry. shrink mode needs unpacking (stretch);
    # blackout/freeze are already native-resolution — decode is the input.
    stretched_frames_list = []
    if removal_mode == 'shrink':
        for i in range(len(decoded_shrunk)):
            stretched = stretch_frame(decoded_shrunk[i], masks_list[i], block_size)
            stretched_frames_list.append(stretched)
    else:
        stretched_frames_list = decoded_shrunk

    # Save frames to disk as PNGs because propainter/e2fgvi expect directories of PNGs
    stretched_dir = os.path.join(results_dir, "stretched_frames")
    masks_dir = os.path.join(results_dir, "masks")
    os.makedirs(stretched_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

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
    elif inpainter == 'telea':
        # Classical per-frame in-painting (cv2, CPU) — the NOSSDAV paper's
        # "ELVIS CV2 benchmark": the published "2-3 VMAF avg" claim is
        # ProPainter/E2FGVI *over this*, so this branch is the replication
        # anchor. Also useful as a GPU-free end-to-end smoke path.
        os.makedirs(output_frames_dir, exist_ok=True)
        radius = int(inpainter_params.get('inpaint_radius', 3))
        for i in range(len(stretched_frames_list)):
            m = cv2.imread(os.path.join(masks_dir, f"{i:05d}.png"), 0)
            inp = cv2.inpaint(stretched_frames_list[i], (m > 127).astype(np.uint8), radius, cv2.INPAINT_TELEA)
            cv2.imwrite(os.path.join(output_frames_dir, f"{i:05d}.png"), inp)
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

    # Passthrough compositing: keep the decoded transmitted pixels (bit-exact FG)
    # and take in-painted pixels only inside the holes. `stretched_frames_list`
    # is the native-resolution transmitted frame (decode for blackout/freeze,
    # stretched decode for shrink); holes there are black/frozen/zero-filled.
    if composite_output:
        pix_masks = [cv2.resize(m.astype(np.uint8), (width, height),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                     for m in masks_list]
        inpainted_frames = composite_passthrough(stretched_frames_list, inpainted_frames, pix_masks)

    # ffv1/bgr0 (verified bit-exact, unlike libx265's yuv420p "lossless" which
    # still chroma-subsamples): matters here because composited pixels are
    # compared directly against reference frames for the FG-quality claim.
    final_output = os.path.join(results_dir, "restored_lossless.mkv")
    save_frames_as_video(inpainted_frames, final_output, framerate, lossless=True, codec="ffv1")
    
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
