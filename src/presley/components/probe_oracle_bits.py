"""F1: how well does the EVCA cost proxy predict a block's true bit cost?

The paper's claim (a) in `NOTE(sec:implementation)` -- that selecting the top-25%
of superblocks by EVCA frees 93.0-99.4% of the bits a perfect bit oracle would
free -- came from an ad-hoc sweep with no `results/<hash>`, n=3 videos, 3-8
frames, and **all-intra coding**. All-intra is the fatal part: with no temporal
prediction a block's cost is essentially its own spatial detail, which is exactly
what EVCA SC measures, so the correlation is near-tautological and 93-99% is an
*upper bound* on the proxy's skill rather than a measurement of it.

This component re-measures it under inter coding, through the runner, so the
result carries a real hash. Design and pre-registered bounds:
`docs/F1_ORACLE_BITS.md`.

The measured quantity
---------------------
For each superblock position `i`, encode a variant with SB `i` mean-filled in
**every** frame, and take

    marginal_bits(i) = bits(reference) - bits(variant_i)

This is deliberately the *operational* quantity rather than a per-block bit tally
read out of the bitstream. Under inter coding, blanking one SB changes bits
inside it **and** in the neighbours that predicted from it, and what selection
actually needs to know is how many bits removing that block frees in total.

Two design points worth not re-litigating
-----------------------------------------
**The SB grid is EVCA's grid.** `get_evca_scores` tiles with `width //
block_size` (floor), so 640x360 at 64x64 gives 10x5 = 50 full superblocks and the
bottom 40-row strip is not covered. `filter_frame_mean_fill` derives its geometry
from the same `frame_scores.shape`, so the score vector and `marginal_bits` are
aligned index-for-index **by construction**. An off-by-one between those two is
the single most likely bug here and would silently produce a plausible but wrong
rho, so the alignment is structural rather than asserted.

**The blended score deliberately omits the x10 background boost.** The full
`get_removability_scores` multiplies background blocks by 10 using the UFO mask.
That is a *selection policy*, not a cost model, and folding it in would conflate
"does the complexity blend predict bits" with "does foreground protection change
what we pick". The question here is only the former, so the blend is computed
directly from EVCA's spatial/temporal outputs.

Citability
----------
`invariants._check_metrics_present` unconditionally requires
`metrics.{foreground,background,overall}.psnr_mean`, with no exemption for
components that produce no reconstructed video -- so a pure bit-accounting probe
would carry a permanent non-empty `invariant_failures` and never be citable,
which would defeat the entire point of re-running this through the runner.

The probe therefore publishes its **reference encode** as `output_video`. The
normal `evaluate_all` pass scores it against the original like any other run, so
the metrics are real measurements of real frames -- the pristine quality of that
clip at this QP -- rather than fabricated numbers or a hole cut in the failsafe.
"""
import os
import time
from typing import Any, Dict, List

import numpy as np

from presley.degradation import filter_frame_mean_fill
from presley.encode_utils import derive_rate_control, encode_video_svtav1_qp
from presley.preprocessing import get_evca_scores, get_reference_frames

TOP_FRACTION = 0.25          # the "top-25% of SBs" the original claim selected


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho without pulling in scipy (not a runtime dep of this repo)."""
    if a.size < 2:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not distort rho."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def _capture_ratio(score: np.ndarray, marginal: np.ndarray, k: int) -> float:
    """Bits freed by the top-k blocks *by score*, over the top-k *by oracle*.

    The oracle denominator is the most bits any k-block choice can free, so this
    is bounded above by 1.0 for a well-formed input. Negative marginal costs are
    kept rather than clipped -- they are real (mean-filling can cost bits) and
    hiding them would flatter the proxy.
    """
    if k <= 0:
        return float("nan")
    by_score = marginal[np.argsort(-score, kind="mergesort")[:k]].sum()
    by_oracle = marginal[np.argsort(-marginal, kind="mergesort")[:k]].sum()
    return float(by_score / by_oracle) if by_oracle > 0 else float("nan")


def run_probe_oracle_bits(experiment: Dict[str, Any], dataset_dir: str,
                          results_dir: str, cache_dir: str) -> Dict[str, Any]:
    video_name = experiment["video"]
    width, height = experiment["width"], experiment["height"]
    block_size = int(experiment.get("block_size", 64))
    codec_params = experiment.get("codec_params", {})
    qp = int(codec_params["qp"])
    preset = str(codec_params.get("preset", "8"))
    alpha = float(experiment.get("alpha", 0.5))
    beta = float(experiment.get("beta", 0.5))

    raw_yuv_path, frames, framerate = get_reference_frames(
        video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    ref_pattern = os.path.join(ref_frames_dir, "%05d.png")

    start = time.time()

    # --- the reference encode, which is also this run's output_video ---------
    reference_video = os.path.join(results_dir, "encoded.mp4")
    encode_video_svtav1_qp(ref_pattern, reference_video, framerate, qp, preset=preset)
    reference_bytes = os.path.getsize(reference_video)

    # --- scores, on the grid that also defines the superblocks ---------------
    temporal_3d, spatial_3d = get_evca_scores(
        video_name, width, height, block_size, raw_yuv_path, ref_frames_dir, cache_dir)
    # Same blend as Eq.(importance): alpha weights spatial now against temporal
    # next, with the final frame falling back to spatial alone. beta is the
    # across-frame smoothing that follows it.
    blended_3d = np.zeros_like(spatial_3d)
    blended_3d[:-1] = alpha * spatial_3d[:-1] + (1 - alpha) * temporal_3d[1:]
    blended_3d[-1] = spatial_3d[-1]
    if beta < 1.0 and blended_3d.shape[0] >= 2:
        smoothed = np.zeros_like(blended_3d)
        smoothed[0] = blended_3d[0]
        smoothed[1:] = beta * blended_3d[1:] + (1 - beta) * blended_3d[:-1]
        blended_3d = smoothed

    n_by, n_bx = spatial_3d.shape[1], spatial_3d.shape[2]
    n_sb = n_by * n_bx
    # One score per SB *position*, averaged over time: the sweep blanks a
    # position for the whole clip, so the comparand must be a whole-clip score.
    sc_vec = spatial_3d.mean(axis=0).ravel()
    blend_vec = blended_3d.mean(axis=0).ravel()

    # --- leave-one-superblock-out sweep --------------------------------------
    variant_dir = os.path.join(results_dir, "_variants")
    os.makedirs(variant_dir, exist_ok=True)
    marginal_bits: List[float] = []
    variant_bytes: List[int] = []
    duration = len(frames) / framerate

    for idx in range(n_sb):
        by, bx = divmod(idx, n_bx)
        sel = np.zeros((n_by, n_bx), dtype=bool)
        sel[by, bx] = True
        frames_dir = os.path.join(variant_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        import cv2
        for fi, frame in enumerate(frames):
            filled, _ = filter_frame_mean_fill(frame, spatial_3d[fi], block_size, sel=sel)
            cv2.imwrite(os.path.join(frames_dir, f"{fi + 1:05d}.png"), filled)
        variant_video = os.path.join(variant_dir, "variant.mp4")
        if os.path.exists(variant_video):
            os.remove(variant_video)
        encode_video_svtav1_qp(os.path.join(frames_dir, "%05d.png"), variant_video,
                               framerate, qp, preset=preset)
        vbytes = os.path.getsize(variant_video)
        variant_bytes.append(vbytes)
        marginal_bits.append((reference_bytes - vbytes) * 8.0)

    import shutil
    shutil.rmtree(variant_dir, ignore_errors=True)

    marginal = np.asarray(marginal_bits, dtype=float)
    k = max(1, int(round(TOP_FRACTION * n_sb)))

    probe = {
        "n_superblocks": n_sb,
        "sb_grid": [n_by, n_bx],
        "top_k": k,
        "reference_size_bytes": reference_bytes,
        "marginal_bits": marginal.tolist(),
        "evca_sc": sc_vec.tolist(),
        "blended_score": blend_vec.tolist(),
        "spearman_sc": _spearman(sc_vec, marginal),
        "spearman_blended": _spearman(blend_vec, marginal),
        "capture_ratio_sc": _capture_ratio(sc_vec, marginal, k),
        "capture_ratio_blended": _capture_ratio(blend_vec, marginal, k),
        "oracle_bits_top_k": float(np.sort(marginal)[::-1][:k].sum()),
        "reference_bits_total": reference_bytes * 8.0,
        "n_negative_marginal": int((marginal < 0).sum()),
    }

    encoding_time = time.time() - start
    return {
        "video_frames": len(frames),
        "video_framerate": framerate,
        "output_video": reference_video,
        "rate_control": derive_rate_control("svtav1", codec_params),
        "actual_bitrate_bps": (reference_bytes * 8) / duration,
        "file_size_bytes": reference_bytes,
        "transmitted_size_bytes": reference_bytes,
        "encoding_time_seconds": encoding_time,
        "restoration_time_seconds": 0.0,
        "total_time_seconds": encoding_time,
        "oracle_bits_probe": probe,
    }
