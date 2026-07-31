"""Claim (b): how much does damage-after-restoration vary between superblocks?

`NOTE(sec:implementation)` claim (b) says the selection objective's denominator
is where the variance lives: per-64x64-superblock damage spans a p90-p10 range of
6.2 dB (realesrgan) and 8.2 dB (propainter) *within a single run* -- one video,
one QP, one restorer, one budget -- against the 0.03-0.05 dB that alpha/beta move
FG-PSNR. That is the mechanism behind the alpha/beta null: the axis alpha and
beta parameterise is not the axis the variance is on.

The number is real but was mined by `tools/mine_block_damage.py` **outside the
runner**, so it has no `results/<hash>` and `NEXT(sec:implementation)` forbids it
from appearing in any reviewer-visible sentence. This component re-measures it
through the runner, exactly as `probe_oracle_bits` did for claim (a).

What it measures
----------------
One run, one config. The degrade->restore half is not reimplemented: it *is* a
`presley_ai` run, dispatched to `run_presley_ai`, so the probe cannot silently
measure a different pipeline than the one the paper describes. On top of that it
encodes the pristine baseline at the same QP and computes, per superblock and
frame,

    delta_psnr(s) = psnr(baseline_encode, s) - psnr(restored_output, s)

i.e. the damage attributable to degrade->restore over and above what the codec
already did at that QP. The headline is the **p90-p10 spread of delta_psnr over
the degraded superblocks of this single run** -- a within-run dispersion, which
is what claim (b) asserts. `delta_psnr` on *untouched* superblocks is reported
alongside: non-zero there is neighbour bleed through inter prediction, not
selection error, and a large value would mean the within-run spread is
contaminated rather than clean.

Why the baseline is encoded here rather than looked up
------------------------------------------------------
A matched pristine baseline usually exists in `results/`, but joining to it is
what made the original number unciteable in the first place -- the measurement
would again depend on a lookup outside its own hash. Encoding it inside the run
costs one extra x265/SVT-AV1 pass on an already GPU-bound job and makes the
result self-contained.

Citability
----------
Same trap and same answer as `probe_oracle_bits`:
`invariants._check_metrics_present` unconditionally requires FG/BG/overall
`psnr_mean`. This probe publishes the **restored video** as `output_video` (it
has a real one -- it ran the real pipeline), so the standard `evaluate_all` pass
scores real frames against the original and no hole is cut in the failsafe. The
saturation invariant applies too, since `config.restorer` is set: a probe run on
a diverging restorer is correctly not citable.
"""
import os
import time
from typing import Any, Dict

import numpy as np

from presley.blockdamage import pool_to_superblocks, psnr_from_mse, superblock_mse
from presley.components.presley_ai import run_presley_ai
from presley.encode_utils import load_frames_from_video
from presley.preprocessing import get_reference_frames
from presley.sidechannel import load_level_masks

# A superblock counts as degraded when most of it was: the strength map is on the
# run's own (smaller) block grid, so an SB straddling the selection boundary is
# genuinely part-degraded and belongs in neither group.
DEGRADED_FRAC = 0.5


def _percentiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"n": 0}
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {
        "n": int(values.size),
        "p10": float(p10),
        "median": float(p50),
        "p90": float(p90),
        "spread_p90_p10": float(p90 - p10),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def run_probe_block_damage(experiment: Dict[str, Any], dataset_dir: str,
                           results_dir: str, cache_dir: str) -> Dict[str, Any]:
    video_name = experiment["video"]
    width, height = experiment["width"], experiment["height"]
    codec_params = experiment.get("codec_params", {})
    codec = experiment.get("codec", "x265")

    # --- the real pipeline, not a re-implementation of it --------------------
    result = run_presley_ai(experiment, dataset_dir, results_dir, cache_dir)

    start = time.time()
    raw_yuv_path, frames, framerate = get_reference_frames(
        video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    ref_pattern = os.path.join(ref_frames_dir, "%05d.png")

    # --- the pristine baseline at the same QP --------------------------------
    baseline_video = os.path.join(results_dir, "_baseline_encode.mp4")
    _encode_pristine(ref_pattern, baseline_video, framerate, codec, codec_params)

    restored = load_frames_from_video(result["output_video"])
    baseline = load_frames_from_video(baseline_video)
    strength = load_level_masks(os.path.join(results_dir, "strength_maps.npz"))

    n_frames = min(len(frames), len(restored), len(baseline), strength.shape[0])
    if n_frames == 0:
        raise RuntimeError(
            f"{video_name}: nothing to compare (frames={len(frames)}, "
            f"restored={len(restored)}, baseline={len(baseline)}, maps={strength.shape[0]})")

    mse_restored = superblock_mse(frames[:n_frames], restored[:n_frames])
    mse_baseline = superblock_mse(frames[:n_frames], baseline[:n_frames])
    block_size = int(experiment.get("block_size", 8))
    frac = pool_to_superblocks(
        (strength[:n_frames] > 0).astype(np.float64), block_size, height, width)
    if frac.shape != mse_restored.shape:
        raise RuntimeError(
            f"{video_name}: strength grid pooled to {frac.shape}, superblock MSE is "
            f"{mse_restored.shape} -- these must be aligned index-for-index")

    delta = psnr_from_mse(mse_baseline) - psnr_from_mse(mse_restored)
    degraded = delta[frac > DEGRADED_FRAC]
    untouched = delta[frac == 0.0]

    probe = {
        "sb_grid": list(mse_restored.shape[1:]),
        "frames": n_frames,
        "degraded_frac_threshold": DEGRADED_FRAC,
        # The headline: dispersion of damage WITHIN this one run.
        "delta_psnr_degraded": _percentiles(degraded),
        # Contamination check, not a result.
        "delta_psnr_untouched": _percentiles(untouched),
        "degraded_sb_fraction": float((frac > DEGRADED_FRAC).mean()),
        "baseline_size_bytes": os.path.getsize(baseline_video),
    }

    os.remove(baseline_video)

    result["block_damage_probe"] = probe
    result["total_time_seconds"] = result.get("total_time_seconds", 0.0) + (time.time() - start)
    return result


def _encode_pristine(pattern: str, output: str, framerate: float,
                     codec: str, codec_params: Dict[str, Any]) -> None:
    """Fixed-QP pristine encode in the same codec the run used.

    Fixed QP only, and no bitrate target: the comparison is "what did the codec
    alone already cost at this QP", which under VBR would not be a fixed
    reference at all (hard rule 1).
    """
    qp = int(codec_params["qp"])
    if codec == "svtav1":
        from presley.encode_utils import encode_video_svtav1_qp
        encode_video_svtav1_qp(pattern, output, framerate, qp,
                               preset=str(codec_params.get("preset", "8")))
    elif codec == "x265":
        from presley.encode_utils import encode_video_x265_qp
        encode_video_x265_qp(pattern, output, framerate, qp,
                             preset=str(codec_params.get("preset", "medium")))
    else:
        raise ValueError(
            f"probe_block_damage has no fixed-QP encode for codec {codec!r}; "
            f"add one rather than falling back to a bitrate target")
