import os
import time
import numpy as np
from typing import Dict, Any

from presley.preprocessing import get_reference_frames, get_removability_scores
from presley.encode_utils import save_frames_as_video, load_frames_from_video, encode_video_x265, encode_video_x265_qp, encode_video_svtav1_qp, encode_video_svtav1, derive_rate_control
from presley.degradation import (filter_frame_downsample, filter_frame_gaussian,
                                 filter_frame_noise, filter_frame_ac_truncate,
                                 filter_frame_mean_fill, filter_frame_freeze,
                                 select_removal_mask_global, upsample_block_mask)
from presley.sidechannel import save_level_masks, composite_passthrough

# Degradations that punch holes to be filled by an in-painter (the ELVIS<->PRESLEY
# bridge), rather than blur/downsample restored by a super-resolver.
INPAINT_DEGRADATIONS = ('mean_fill', 'freeze')

# Restorer -> which degradations it can consume.
#
# In-painters MASK OUT the degraded region before doing anything -- ProPainter's
# inference opens with `masked_frames = frames * (1 - masks_dilated)` -- so they
# discard whatever was transmitted inside the hole and can only synthesize from
# neighbouring frames. Pairing one with an information-preserving degradation
# would pay bits for a prior that is then thrown away, so they stay restricted
# to the hole degradations.
#
# CONDITIONED restorers take the degraded pixels as input and are degradation-
# agnostic in their internals: `restore_with_instantir_adaptive` restores the
# whole frame each round and copies back blocks whose map has hit 0, and
# `upscale_realesrgan_adaptive` re-injects the degraded input at every pyramid
# level. Neither ever sees a mask. They were previously pinned to
# downsample/blur by a hard assert, which is exactly the pairing that made
# Goal 2 untestable: the only degradations they accepted are the ones that free
# no bits under fixed QP, and the only degradations that free bits were paired
# with restorers that discard the prior. Unlocking the hole degradations for
# them is the point of this table.
#
# It is NOT a free-for-all, because each conditioned restorer INTERPRETS the
# strength map differently and a wrong pairing fails silently rather than
# loudly. `upscale_realesrgan_adaptive` does `np.power(2, map)` (downscale
# exponent), so handing it a noise map (values up to noise_variance) computes
# 2**50 and degenerates cv2.resize; `restore_blur_opencv_unsharp_mask` reads
# the map as sharpening rounds. So each restorer lists exactly the degradations
# whose map it can interpret, and `_restorer_strength_map` below converts +
# clamps. `None` = genuinely map-agnostic.
RESTORER_DEGRADATIONS = {
    'none':       None,
    'propainter': INPAINT_DEGRADATIONS,
    'e2fgvi':     INPAINT_DEGRADATIONS,
    'telea':      INPAINT_DEGRADATIONS,
    # map = log2 downscale factor
    'realesrgan': ('downsample',) + INPAINT_DEGRADATIONS,
    # map = log2 downscale factor (same units as realesrgan -- BSRGAN is the
    # same RRDBNet architecture with a different training-time degradation
    # model, so it consumes the strength map identically. Second conditioned
    # GAN/CNN super-resolver for the restoration-comparison ablation; see
    # `restore_downsampled_with_bsrgan` in presley.restoration.)
    'bsrgan':     ('downsample',) + INPAINT_DEGRADATIONS,
    # Recent SR GAN (Chen et al. / XPixelGroup HAT). Same log2-downscale map
    # as realesrgan/bsrgan; fp32 only — see restore_downsampled_with_real_hat_gan.
    'real_hat_gan': ('downsample',) + INPAINT_DEGRADATIONS,
    # 4× auto-regressive diffusion VSR (Shiu et al. / Stream-DiffVSR). Same
    # log2-downscale map units; temporal sequence path — see
    # restore_downsampled_with_stream_diffvsr. Isolated vendor env (do not
    # pip into pinned `presley`). fp16: try on CUDA; reject on Softmax/LN NaNs.
    'stream_diffvsr': ('downsample',) + INPAINT_DEGRADATIONS,
    # Diffusion VSR quality arm (Han et al. SIGGRAPH 2025 / DC-VSR). Same
    # log2-downscale map as the SR GANs. Dispatch is wired; inference is
    # blocked until upstream publishes a usable entrypoint — see
    # restore_downsampled_with_dc_vsr. fp32 only.
    'dc_vsr':      ('downsample',) + INPAINT_DEGRADATIONS,
    # map = number of restoration rounds
    'instantir':  ('blur',) + INPAINT_DEGRADATIONS,
    # CNN deblur gauge (Chen et al. 2022); same blur (+ hole) set as InstantIR.
    # Single full-frame forward — see restore_with_nafnet_adaptive.
    # `ac_truncate` is admitted here because NAFNet is a conditioned restorer
    # that takes the degraded pixels and does ONE full-frame forward, pasting
    # untouched blocks back -- it reads the map only as "was this block
    # degraded". AC truncation's map is binary for exactly the same reason
    # blur's is (one operator application per block), so the units match. This
    # pairing is what makes the O2 re-test possible: the same prior on both
    # operators, so the operator is the only variable.
    'nafnet':     ('blur', 'ac_truncate') + INPAINT_DEGRADATIONS,
    # map = sharpening rounds; downsample is also a low-pass, so unsharp is a
    # meaningful no-ML benchmark for it too. AC truncation is a low-pass in the
    # transform domain, so unsharp is its no-ML control as well.
    'unsharp':    ('blur', 'downsample', 'ac_truncate') + INPAINT_DEGRADATIONS,
}
# NOTE: no `lanczos` entry. `restore_downsample_opencv_lanczos` expects a frame
# whose blocks are still physically shrunk, but `filter_frame_downsample`
# downsamples AND re-upsamples within each block, so the transmitted frame is
# already full-resolution. Running it there re-downsamples and scores BELOW
# `restorer: none` (18.17 dB vs 18.43 dB on a bear frame) -- it is a second
# degradation, not a benchmark. `unsharp` is the correct no-ML control.

# Per-restorer clamps for the converted strength map: (max_value, dtype).
_STRENGTH_CLAMP = {
    'realesrgan': 3, 'bsrgan': 3, 'real_hat_gan': 3, 'stream_diffvsr': 3, 'dc_vsr': 3,
    'instantir': 8, 'nafnet': 8, 'unsharp': 8,
}


def _restorer_strength_map(smap_arr, restorer: str, degradation: str, restorer_params: Dict[str, Any]):
    """Translate the transmitted strength map into the restorer's own units."""
    if degradation in INPAINT_DEGRADATIONS:
        # Hole maps are binary; `rounds` lets the conditioned arm ask for more
        # than a single pass over the hole.
        level = int(restorer_params.get('rounds', 1))
        out = (np.asarray(smap_arr) > 0).astype(np.int32) * level
    else:
        out = np.rint(np.asarray(smap_arr)).astype(np.int32)
    cap = _STRENGTH_CLAMP.get(restorer)
    return np.clip(out, 0, cap) if cap is not None else out

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
    # Budgeted selection: when shrink_amount is set, select blocks with elvis's
    # global top-k (same removal budget -> same starved operating point) instead
    # of the round(score)>0 threshold, which degrades too few blocks (9.4% on
    # bear -> 844 kbps, the comfortable regime where nothing can win).
    # fg_protect adds the hard UFO-mask exclusion, same as elvis.
    #
    # This applies to EVERY degradation as of 2026-07-20. It used to be gated on
    # `degradation in INPAINT_DEGRADATIONS`, so downsample/blur/noise silently
    # fell back to the bare threshold -- meaning the screen that retired them
    # ("relocate no bits under fixed QP") compared 9-13% unclustered degraded
    # blocks against the bridge's 25% clustered, and was never budget-matched.
    #
    # ⚠ THE NAME IS A FOSSIL. `shrink_amount` does not shrink anything here.
    # It named a geometric operation in original ELVIS: remove blocks, repack
    # the survivors into a smaller rectangle (`removal_mode: 'shrink'`, retired
    # -- it breaks temporal prediction, +193% bitrate overshoot). No current
    # degradation shrinks the frame; freeze/blackout/downsample/blur/ac_truncate
    # all act on blocks in place and the frame stays full size.
    #
    # What it means NOW is the *fraction of blocks to degrade* -- a selection
    # budget. That is why it is read for every degradation rather than only the
    # in-painting ones. Two readers have already inferred from the name that the
    # parameter was dead legacy and could be dropped; dropping it silently
    # switches selection back to the unbudgeted `round(score)>0` threshold and
    # moves the operating point.
    #
    # NOT renamed, on purpose: it is part of compute_experiment_hash and of the
    # control-matching key the analysis tools use, so a rename orphans every
    # existing results/<hash>/ and unmatches arms from their `none` controls. An
    # alias would be worse -- two keys for one behaviour is one experiment with
    # two hashes. If it is ever renamed, migrate the corpus in the same commit.
    select_amount = experiment.get('shrink_amount')
    fg_protect = experiment.get('fg_protect', False)
    temporal_pool_masks = experiment.get('temporal_pool_masks', False)
    # Which foreground mask feeds removability scoring AND fg_protect below:
    # 'ufo' (default, existing behavior), 'gt' (ground-truth annotations), or
    # 'yolo' (open-vocab YOLOE). See preprocessing.resolve_masks.
    mask_source = experiment.get('mask_source', 'ufo').lower()
    # Orthogonal mask-noise axis (Q10): dilate|erode|jitter|none after the
    # base source. Radius is pixels on the full-res mask; seed is jitter-only.
    mask_morphology = experiment.get('mask_morphology', 'none')
    mask_morphology_radius = int(experiment.get('mask_morphology_radius', 0))
    mask_morphology_seed = int(experiment.get('mask_morphology_seed', 0))
    # S1: graded multi-level downscale (default 1 = the historical binary
    # behavior, byte-exact). Only 'downsample' consumes this -- the pyramid
    # restorers (Real-ESRGAN/BSRGAN/...) are the only restoration path built to
    # read a multi-level map (_adaptive_block_pyramid_upscale); blur/noise/
    # mean_fill/freeze have no graded restoration path, so this key is inert
    # for them and ignored on purpose rather than raising.
    downsample_levels = int(experiment.get('downsample_levels', 1))
    # S1b probe mode: force every block in the downsample_levels footprint to
    # one fixed level, so per-block damage at that level can be mined. Default
    # 0 = off, so no existing hash moves.
    downsample_uniform_level = int(experiment.get('downsample_uniform_level', 0))
    # S1b Arm B: replace the score-derived level assignment with a precomputed
    # per-frame level map (tools/build_oracle_levels.py), which keeps Arm A's
    # exact level histogram but reassigns which block sits at which level using
    # mined damage instead of the removability score. Default None = off, so no
    # existing hash moves. The map is authoritative -- it already encodes the
    # footprint, so both `sel` and the level quantizer are bypassed for it.
    downsample_level_map = experiment.get('downsample_level_map')
    oracle_levels = None
    if downsample_level_map:
        _p = os.path.join(downsample_level_map, f"{video_name.replace('/', '_')}.npz")
        if not os.path.exists(_p):
            raise FileNotFoundError(
                f"downsample_level_map is set but {_p} does not exist -- an Arm-B run "
                "without its oracle map would silently fall back to the naive graded "
                "assignment and be indistinguishable from Arm A")
        oracle_levels = np.load(_p)['levels']
    # Operator STRENGTH knobs. Both defaults reproduce the historical hardcoded
    # behavior byte-for-byte, so no existing experiment hash or output moves:
    # blur was pinned at cv2's k=15 here (`roi.py` always read it from the
    # config; `presley_ai` never did, which is why F5 could only test one blur
    # strength), and ac_truncate is new with F5's keep=2 as its default.
    blur_kernel = int(experiment.get('blur_kernel', 15))
    ac_keep = int(experiment.get('ac_keep', 2))

    # 1. Load data
    raw_yuv_path, frames, framerate = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    removability_scores = get_removability_scores(
        video_name, width, height, block_size, alpha, beta, dataset_dir, cache_dir,
        mask_source=mask_source,
        mask_morphology=mask_morphology,
        mask_morphology_radius=mask_morphology_radius,
        mask_morphology_seed=mask_morphology_seed,
    )
    
    start_time = time.time()
    
    # 2. Degrade
    degraded_frames_list = []
    strength_maps_list = []
    frames_arr = np.array(frames)
    
    # Block-level FG masks for hard protection (same recipe as elvis).
    fg_block_masks = None
    if fg_protect:
        import cv2 as _cv2
        from presley.preprocessing import resolve_masks
        ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
        ufo = resolve_masks(
            mask_source, video_name, width, height, block_size, ref_frames_dir, cache_dir,
            dataset_dir, temporal_pool=temporal_pool_masks,
            mask_morphology=mask_morphology,
            mask_morphology_radius=mask_morphology_radius,
            mask_morphology_seed=mask_morphology_seed,
        )
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
        if select_amount is not None:
            excl = fg_block_masks[i] if fg_block_masks is not None and i < len(fg_block_masks) else None
            sel = select_removal_mask_global(score, select_amount, cluster_blocks=True, exclude=excl) > 0

        if degradation == 'downsample':
            _lm = None
            if oracle_levels is not None:
                if i >= len(oracle_levels):
                    raise IndexError(
                        f"oracle level map has {len(oracle_levels)} frames but frame {i} was "
                        f"requested -- the map was built against a different frame count")
                _lm = oracle_levels[i]
            degraded, smap = filter_frame_downsample(frame, score, block_size, sel=sel, levels=downsample_levels,
                                                    uniform_level=downsample_uniform_level,
                                                    level_map=_lm)
        elif degradation == 'blur':
            degraded, smap = filter_frame_gaussian(frame, score, block_size, kernel_size=blur_kernel, sel=sel)
        elif degradation == 'ac_truncate':
            degraded, smap = filter_frame_ac_truncate(frame, score, block_size, keep=ac_keep, sel=sel)
        elif degradation == 'noise':
            degraded, smap = filter_frame_noise(frame, score, block_size, sel=sel)
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
            _tune = codec_params.get('tune')
            encode_video_svtav1_qp(temp_degraded_vid, transmitted_video, framerate, int(codec_params['qp']), preset=codec_params.get('preset', '8'), tune=None if _tune is None else int(_tune))
        else:
            encode_video_svtav1(temp_degraded_vid, transmitted_video, framerate, target_bitrate, preset=codec_params.get('preset', '8'))
    else:
        raise ValueError(f"Presley AI currently requires x265 or svtav1 for encoding, got {codec}")

    encoding_time = time.time() - start_time
    restoration_start = time.time()

    # Save strength maps (transmitted side information), bit-plane packed for
    # every degradation. Binary hole maps pack to a single plane, which is
    # byte-identical to the previous save_binary_masks output -- so no existing
    # experiment's transmitted_size_bytes moves. Multi-level maps used to be
    # written as raw savez'd int32, ~2.3x larger than necessary.
    strength_maps_path = os.path.join(results_dir, "strength_maps.npz")
    save_level_masks(strength_maps_list, strength_maps_path)
    
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

        # Capability check, single point of truth (see RESTORER_DEGRADATIONS).
        # Unknown names must fall through to the "Unknown restorer" branch
        # below rather than being reported as a capability conflict.
        if restorer in RESTORER_DEGRADATIONS:
            allowed = RESTORER_DEGRADATIONS[restorer]
            if allowed is not None and degradation not in allowed:
                raise ValueError(
                    f"restorer '{restorer}' cannot consume degradation '{degradation}': "
                    f"it either masks the degraded region out (in-painters discard the "
                    f"transmitted prior) or cannot interpret that strength map. "
                    f"Allowed: {allowed}")
        smap_r = _restorer_strength_map(smap_arr, restorer, degradation, restorer_params)

        if restorer == 'realesrgan':
            from presley.restoration import restore_downsampled_with_realesrgan
            # tile>0 processes the frame in tiles → much lower peak VRAM (lets the job
            # fit alongside other GPU processes); tile=0 is full-frame (fastest, most VRAM).
            restore_downsampled_with_realesrgan(
                temp_frames_dir, restored_frames_dir, smap_r, block_size,
                denoise_strength=restorer_params.get('denoise_strength', 1.0),
                tile=restorer_params.get('tile', 0),
                tile_pad=restorer_params.get('tile_pad', 10),
                fp32=restorer_params.get('fp32', False))

        elif restorer == 'bsrgan':
            from presley.restoration import restore_downsampled_with_bsrgan
            # Second conditioned GAN/CNN super-resolver (RRDBNet arch, same
            # as realesrgan above, different training-time degradation
            # model) -- the restoration-comparison ablation this branch
            # exists for. No denoise_strength: that's a Real-ESRGAN DNI
            # feature specific to the realesr-general-x4v3 checkpoint, not
            # applicable to the fixed BSRGAN model.
            restore_downsampled_with_bsrgan(
                temp_frames_dir, restored_frames_dir, smap_r, block_size,
                tile=restorer_params.get('tile', 0),
                tile_pad=restorer_params.get('tile_pad', 10),
                fp32=restorer_params.get('fp32', False))

        elif restorer == 'real_hat_gan':
            from presley.restoration import restore_downsampled_with_real_hat_gan
            # Recent HAT-based real-world SR GAN (Q4). Same adaptive pyramid
            # as realesrgan/bsrgan. Default fp32=True — half precision NaNs.
            restore_downsampled_with_real_hat_gan(
                temp_frames_dir, restored_frames_dir, smap_r, block_size,
                weights_path=restorer_params.get('weights_path'),
                tile=restorer_params.get('tile', 0),
                tile_pad=restorer_params.get('tile_pad', 32),
                fp32=bool(restorer_params.get('fp32', True)))

        elif restorer == 'stream_diffvsr':
            from presley.restoration import restore_downsampled_with_stream_diffvsr
            # Q7 diffusion VSR vs Real-ESRGAN. Isolated vendor env + HF
            # Jamichsu/Stream-DiffVSR. Default fp32=False (try fp16 on CUDA);
            # non-finite outputs / known Softmax NaNs demand fp32=True.
            restore_downsampled_with_stream_diffvsr(
                temp_frames_dir, restored_frames_dir, smap_r, block_size,
                model_id=restorer_params.get('model_id', 'Jamichsu/Stream-DiffVSR'),
                num_inference_steps=int(restorer_params.get('num_inference_steps', 4)),
                fp32=bool(restorer_params.get('fp32', False)),
                vendor_dir=restorer_params.get('vendor_dir'),
                python_executable=restorer_params.get('python'),
                cache_dir=restorer_params.get('cache_dir') or restorer_params.get('weights_path'))

        elif restorer == 'dc_vsr':
            from presley.restoration import restore_downsampled_with_dc_vsr
            # Q8 diffusion VSR quality arm. Dispatch + weight layout are real;
            # restore_downsampled_with_dc_vsr raises RuntimeError until HF
            # publishes inference (SAP/TAP/DSSAG + VAE/scheduler). fp32 only.
            restore_downsampled_with_dc_vsr(
                temp_frames_dir, restored_frames_dir, smap_r, block_size,
                weights_dir=restorer_params.get('weights_dir'),
                fp32=bool(restorer_params.get('fp32', True)),
                tile=restorer_params.get('tile', 0),
                num_frames=restorer_params.get('num_frames', 14))

        elif restorer == 'instantir':
            from presley.restoration import restore_with_instantir_adaptive
            # batch_size is the main VRAM lever for InstantIR (SDXL-class); drop to 1–2
            # to run alongside other GPU jobs instead of needing a whole free GPU.
            restore_with_instantir_adaptive(
                temp_frames_dir, restored_frames_dir, smap_r, block_size,
                cfg=restorer_params.get('cfg', 7.0),
                creative_start=restorer_params.get('creative_start', 1.0),
                preview_start=restorer_params.get('preview_start', 0.0),
                # Default 1 preserves historical CLAIM hashes; Q2 corrected
                # smokes pass 20–30 (docs/EXPERIMENTS_QUEUED.md InstantIR audit).
                num_inference_steps=int(restorer_params.get('num_inference_steps', 1)),
                batch_size=restorer_params.get('batch_size', 4))
        elif restorer == 'nafnet':
            from presley.restoration import restore_with_nafnet_adaptive
            # CNN deblur gauge for blur transport (Q5 / HOLE(tab:instantir-kill)).
            # Single full-frame pass; FG via composite_passthrough below.
            restore_with_nafnet_adaptive(
                temp_frames_dir, restored_frames_dir, smap_r, block_size,
                width=int(restorer_params.get('width', 64)),
                weights_path=restorer_params.get('weights_path'),
                # fp32 default True — fp16 overflows NAFNet (rainbow artifacts).
                fp32=bool(restorer_params.get('fp32', True)),
                local=bool(restorer_params.get('local', True)))
        elif restorer == 'unsharp':
            # No-ML conditioned benchmark: a generative restorer has to beat
            # this before its cost is justified. Not a formality -- on freeze
            # the classical Telea in-painter already beats both in-painters.
            from presley.restoration import restore_blur_opencv_unsharp_mask
            for i in range(len(decoded_degraded)):
                out = restore_blur_opencv_unsharp_mask(decoded_degraded[i], smap_r[i], block_size)
                cv2.imwrite(os.path.join(restored_frames_dir, f"{i:05d}.png"), out)
        elif restorer in ('propainter', 'e2fgvi', 'telea'):
            # In-painting restorers for the hole-punching degradations (mean_fill /
            # freeze): fill the degraded region rather than super-resolve it.
            masks_dir = os.path.join(results_dir, "temp_masks")
            os.makedirs(masks_dir, exist_ok=True)
            for i in range(len(strength_maps_list)):
                # exact block-grid upsample, not cv2.resize (misaligns block
                # boundaries when height/width isn't an exact multiple of
                # block_size -- see upsample_block_mask docstring)
                m_full = upsample_block_mask((strength_maps_list[i] > 0).astype(np.uint8),
                                             block_size, width, height) * 255
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
        pix_masks = [upsample_block_mask((strength_maps_list[i] > 0).astype(np.uint8),
                                         block_size, width, height).astype(bool)
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
