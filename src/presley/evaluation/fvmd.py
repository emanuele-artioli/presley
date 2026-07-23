"""Frechet Video Motion Distance and its cached feature rows.

FVMD has no established JND anywhere in the literature, so it is reported
for context and never gates a verdict."""

import os
import json
import numpy as np
from typing import Dict, Any, List
from presley.preprocessing import get_reference_frames, get_ufo_masks
from presley.encode_utils import load_frames_from_video
_REF_CACHE: Dict[Any, Any] = {}
_MASK_CACHE: Dict[Any, Any] = {}
_DISTS_CACHE: Dict[str, Any] = {}
from presley.evaluation.cache import _fvmd_feats_dir, _get_refs_cached, _savez_atomic


def _fvmd_hist_rows(refs: List[np.ndarray], decs: List[np.ndarray]):
    """Track a grid of keypoints across `refs` (ground truth) and `decs`
    (generated) and return `(gt_rows, gen_rows)`, each shape `[n_clips, D]` —
    the per-clip velocity+acceleration motion histograms that FVMD's Frechet
    distance is computed over. FVMD splits each video into fixed-length clips
    (the leading dim), so a single video already yields several feature rows.

    `gt_rows` depends only on the reference frames, so it can be reused as the
    reference distribution for any method's output on the same source video.
    Raises on failure; callers decide how to handle it.
    """
    from fvmd.datasets.video_datasets import VideoDatasetNP
    from fvmd.keypoint_tracking import track_keypoints
    from fvmd.extract_motion_features import calc_hist
    import tempfile

    refs_np = np.expand_dims(np.array(refs), axis=0)
    decs_np = np.expand_dims(np.array(decs), axis=0)
    gt_dataset = VideoDatasetNP(refs_np)
    gen_dataset = VideoDatasetNP(decs_np)
    with tempfile.TemporaryDirectory() as td:
        velo_gen, velo_gt, acc_gen, acc_gt = track_keypoints(
            log_dir=td, gen_dataset=gen_dataset, gt_dataset=gt_dataset, v_stride=1)
    B = velo_gen.shape[0]
    gt_rows = np.concatenate((calc_hist(velo_gt).reshape(B, -1),
                              calc_hist(acc_gt).reshape(B, -1)), axis=1)
    gen_rows = np.concatenate((calc_hist(velo_gen).reshape(B, -1),
                               calc_hist(acc_gen).reshape(B, -1)), axis=1)
    return gt_rows, gen_rows
def _fvmd_on_frames(refs: List[np.ndarray], decs: List[np.ndarray]) -> float:
    """Per-video FVMD: Frechet distance over one video's own clip features.
    NOTE: an intra-video adaptation — the leading dim is the clips of a single
    sequence, so scores are not comparable across videos of different motion
    content. Kept as an internal signal; the paper-grade metric is set-level
    (see `fvmd_set_level`)."""
    try:
        from fvmd.frechet_distance import calculate_fd_given_vectors
        gt_rows, gen_rows = _fvmd_hist_rows(refs, decs)
        return float(calculate_fd_given_vectors(gt_rows, gen_rows))
    except Exception as e:
        print(f"FVMD failed: {e}")
        return float('nan')
def _fvmd_rows_cached(experiment_hash, results_dir, cache_dir, dataset_dir):
    """Return `(gt_rows, gen_rows, key)` for one experiment, tracking keypoints
    only on a cache miss.

    Keypoint tracking is the whole cost of FVMD (~a minute of GPU per video);
    every statistic built on the rows afterwards is pure numpy. Persisting the
    768-dim rows to `cache/fvmd_feats/` makes the null control, the jackknife and
    any future analysis variant instant and re-runnable instead of costing a full
    GPU pass each.

    `key` is `(video, width, height, n_frames)`. The frame count is part of the
    key so a truncated decode can never read back another length's rows.

    A decode whose length differs from the reference is REJECTED rather than
    silently trimmed to `min(len(refs), len(decs))`. Two reasons, both
    correctness:
      * trimming makes `gt_rows` length-dependent, so the same source video
        would mint a second, near-duplicate reference block and get
        double-weighted in the pooled ground truth — breaking the
        "each video contributes exactly once" contract in `_collect_fvmd_rows`.
      * a set whose clips come from different-length decodes is not a
        like-for-like distribution comparison in the first place.
    A rejection is announced loudly; it must not pass as a quietly smaller set.

    Returns `(None, None, None)` if the experiment can't contribute.
    """
    result_path = os.path.join(results_dir, experiment_hash, "result.json")
    if not os.path.exists(result_path):
        print(f"  FVMD skip {experiment_hash}: no result.json"); return None, None, None
    data = json.load(open(result_path))
    cfg = data['config']
    video_name, width, height = cfg['video'], cfg['width'], cfg['height']
    output_video = data.get('output_video')
    if not output_video or not os.path.exists(output_video):
        print(f"  FVMD skip {experiment_hash}: output video missing"); return None, None, None

    feats_dir = _fvmd_feats_dir(cache_dir)
    gen_path = os.path.join(feats_dir, f"{experiment_hash}.npz")
    if os.path.exists(gen_path):
        z = np.load(gen_path)
        key = (video_name, width, height, int(z['n_frames']))
        ref_path = os.path.join(feats_dir, f"ref_{video_name}_{width}x{height}_{key[3]}.npz")
        if os.path.exists(ref_path):
            return np.load(ref_path)['rows'], z['rows'], key

    _, refs, _ = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    decs = load_frames_from_video(output_video)
    if not decs:
        print(f"  FVMD skip {experiment_hash}: no decodable frames"); return None, None, None
    if len(decs) != len(refs):
        print(f"  FVMD REJECT {experiment_hash} ({video_name}): decode has {len(decs)} frames "
              f"but reference has {len(refs)} — excluded from the set (see _fvmd_rows_cached)")
        return None, None, None
    n = len(refs)
    try:
        gt_rows, gen_rows = _fvmd_hist_rows(refs, decs)
    except Exception as e:
        print(f"  FVMD skip {experiment_hash}: {e}"); return None, None, None

    key = (video_name, width, height, n)
    _savez_atomic(gen_path, rows=gen_rows, n_frames=n)
    ref_path = os.path.join(feats_dir, f"ref_{video_name}_{width}x{height}_{n}.npz")
    if not os.path.exists(ref_path):
        _savez_atomic(ref_path, rows=gt_rows)
    return gt_rows, gen_rows, key
def fvmd_set_level(hashes, results_dir, cache_dir, dataset_dir, ref_cache=None):
    """Pool per-clip motion-histogram rows across a SET of experiments (which
    together span several source videos) and return one FVMD for the set's
    generated distribution vs the pooled clean-reference distribution.

    Returns `(fvmd_value, used_hashes)`. `ref_cache` (a dict keyed by
    `(video,width,height,n_frames)`) is populated with each source video's
    reference rows so the ground-truth distribution is tracked once and reused
    (and, crucially, each video contributes to the reference set exactly once)
    across every set that shares `ref_cache`.
    """
    gt_all, gen_all, used = _collect_fvmd_rows(hashes, results_dir, cache_dir,
                                               dataset_dir, ref_cache=ref_cache)
    if not gen_all:
        return float('nan'), []
    fd, _ = _fd_with_terms(np.concatenate(gt_all, axis=0), np.concatenate(gen_all, axis=0))
    return fd, [h for h, _ in used]
def _collect_fvmd_rows(hashes, results_dir, cache_dir, dataset_dir, ref_cache=None):
    """Gather rows for a set. Returns `(gt_blocks, gen_blocks, used)` where
    `gt_blocks` holds one reference row-block per *unique* source video in the
    set (deduped — a video that appears twice must not double-weight the ground
    truth) and `used` is a list of `(hash, key)`."""
    if ref_cache is None:
        ref_cache = {}
    gen_all, used = [], []
    for h in hashes:
        gt_rows, gen_rows, key = _fvmd_rows_cached(h, results_dir, cache_dir, dataset_dir)
        if gen_rows is None:
            continue
        ref_cache.setdefault(key, gt_rows)  # reference rows depend only on the clean video
        gen_all.append(gen_rows)
        used.append((h, key))
    if not gen_all:
        return [], [], []
    ref_keys = {k for _, k in used}
    return [ref_cache[k] for k in ref_keys], gen_all, used
def _fd_with_terms(feat1, feat2):
    """Frechet distance plus its decomposition into the mean and covariance
    terms: `FD = ||mu1-mu2||^2 + (tr(s1) + tr(s2) - 2*tr(covmean))`.

    At our sample size the rows are 768-dim but a 6-video set pools only ~402 of
    them, so the covariance estimate is rank-deficient and its term carries most
    of the estimator's bias. Splitting the terms out shows directly whether a
    score is driven by the mean difference (trustworthy at this N) or by the
    covariance term (not).

    Returns `(fd, {"mean_term":…, "cov_term":…})`. Raises on a numerically
    invalid sqrtm — callers must not turn that into a score.
    """
    from scipy import linalg
    from fvmd.frechet_distance import calculate_activation_statistics
    mu1, s1 = calculate_activation_statistics(feat1)
    mu2, s2 = calculate_activation_statistics(feat2)
    diff = mu1 - mu2
    mean_term = float(diff.dot(diff))
    eps = 1e-5  # parity with fvmd.frechet_distance.calculate_frechet_distance's default
    offset = np.eye(s1.shape[0]) * eps
    covmean, _ = linalg.sqrtm((s1 + offset).dot(s2 + offset), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        # Mirror the reference implementation EXACTLY: a materially complex
        # sqrtm means the result is numerically meaningless, and taking .real
        # unconditionally would launder that failure into a plausible finite
        # score. That matters most here -- a near-singular product is expected
        # when N < D, which is the regime this whole report exists to measure.
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f'Imaginary component {np.max(np.abs(covmean.imag))}')
        covmean = covmean.real
    cov_term = float(np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
    return mean_term + cov_term, {"mean_term": mean_term, "cov_term": cov_term}
