"""Validity diagnostics for FVMD and FID.

These answer 'is this metric measuring what we think on this data',
which is a question about the metric, not about a method."""

import os
import json
import numpy as np
import torch
from typing import Dict, Any, List
from presley.preprocessing import get_reference_frames, get_ufo_masks
from presley.encode_utils import load_frames_from_video
_REF_CACHE: Dict[Any, Any] = {}
_MASK_CACHE: Dict[Any, Any] = {}
_DISTS_CACHE: Dict[str, Any] = {}
from presley.evaluation.cache import _get_masks_cached, _get_refs_cached
from presley.evaluation.masked import _fg_tight_bbox
from presley.evaluation.perceptual import _inception_feats
from presley.evaluation.fvmd import _collect_fvmd_rows, _fd_with_terms, fvmd_set_level


def fvmd_setlevel_report(groups_path: str, results_dir: str, cache_dir: str,
                         dataset_dir: str, out_path: str) -> None:
    """Compute one set-level FVMD per group and write a standalone table.

    `groups_path` is a JSON file mapping set-name -> list of experiment hashes,
    e.g. {"baseline": ["ab..","cd..",...], "presley_ai": [...], ...}. All groups
    share a `ref_cache`, so the clean-reference distribution for each source
    video is computed once across the whole report.
    """
    with open(groups_path) as f:
        groups = json.load(f)
    ref_cache = {}
    rows = []
    for name, hashes in groups.items():
        print(f"[set-level FVMD] {name}: {len(hashes)} experiments")
        fd, used = fvmd_set_level(hashes, results_dir, cache_dir, dataset_dir, ref_cache=ref_cache)
        print(f"  -> FVMD={fd}  (used {len(used)}/{len(hashes)})")
        rows.append((name, fd, len(used), len(hashes)))
    with open(out_path, 'w') as f:
        f.write("set\tfvmd\tn_used\tn_total\n")
        for name, fd, nu, nt in rows:
            f.write(f"{name}\t{fd}\t{nu}\t{nt}\n")
    print(f"[set-level FVMD] wrote {out_path}")
def fvmd_validity_report(groups_path: str, results_dir: str, cache_dir: str,
                         dataset_dir: str, out_path: str, n_null: int = 10,
                         seed: int = 0) -> None:
    """Decide whether set-level FVMD can discriminate our methods AT ALL at this
    sample size, before any score is cited.

    FVMD rows here are 1024-dim while a 6-video set pools only ~372 of them, so
    the covariance is rank-deficient (N < D) and the Frechet distance is
    eps-regularised. That sounds fatal and is NOT, for one specific reason:

      **our comparison is PAIRED.** The decoded/restored video contains the same
      clips as its reference, so `gt_rows` and `gen_rows` describe the same
      underlying motion. The covariance-estimation error is then common to both
      sides and largely cancels inside `sqrtm(s1·s2)`. Measured: an *unpaired*
      split of identical data at N=186/side scores ~5.9e3 (real rows) and ~6.5e4
      (synthetic), while a *paired* comparison at N=372 with 1%/5%/20% added
      noise scores 4 / 103 / 1645 — small, and cleanly monotone in the
      perturbation. Rank-deficiency does not dominate a paired score.

    Consequences, learned the hard way (an earlier version of this report got
    both wrong and produced a "null floor" larger than the scores it was meant to
    bound -- an impossibility that revealed the error):

      * A split-half null of the reference is **UNPAIRED** and therefore does not
        bound our paired scores. It is reported below strictly as a diagnostic of
        the estimator's unpaired behaviour, and must never be read as a floor.
      * Subsampling `gen` to match a reference half likewise **breaks the
        pairing** and produces meaningless (huge) numbers. Not done.

    The real uncertainty is that we have only **6 source videos**, so the live
    question is video-sampling, not covariance rank. Hence:

      * chain scores -- one paired FVMD per group, with the mean/cov term split.
      * identity check -- FD(ref, ref) must be exactly 0 (instrument gate).
      * jackknife -- leave-one-video-out spread per group. **This is the gate:**
        if the between-group gaps are not large compared to the jackknife spread,
        the ordering is driven by which videos we happened to pick, and must not
        be cited.
    """
    with open(groups_path) as f:
        groups = json.load(f)
    ref_cache: Dict[Any, Any] = {}
    rows = []

    per_group = {}
    for name, hashes in groups.items():
        print(f"[FVMD validity] collecting {name}: {len(hashes)} experiments")
        gt_blocks, gen_blocks, used = _collect_fvmd_rows(hashes, results_dir, cache_dir,
                                                         dataset_dir, ref_cache=ref_cache)
        if not gen_blocks:
            print(f"  {name}: no usable experiments"); continue
        per_group[name] = (gt_blocks, gen_blocks, used)

    if not per_group:
        print("[FVMD validity] nothing to report"); return

    # --- chain scores + term decomposition -------------------------------
    for name, (gt_blocks, gen_blocks, used) in per_group.items():
        gt = np.concatenate(gt_blocks, axis=0)
        gen = np.concatenate(gen_blocks, axis=0)
        fd, terms = _fd_with_terms(gt, gen)
        print(f"[FVMD validity] {name}: FVMD={fd:.2f} "
              f"(mean_term={terms['mean_term']:.2f} cov_term={terms['cov_term']:.2f}) "
              f"N_gen={gen.shape[0]} N_gt={gt.shape[0]} D={gen.shape[1]}")
        rows.append((name, "score", fd, terms['mean_term'], terms['cov_term'],
                     gen.shape[0], gen.shape[1], len(used)))

    # Groups must span the same source videos, or their scores are not
    # comparable to each other (different ground-truth mixtures).
    vid_sets = {name: frozenset(k[0] for _, k in used) for name, (_, _, used) in per_group.items()}
    if len(set(vid_sets.values())) != 1:
        print("[FVMD validity] WARNING: groups do NOT span the same source videos — "
              "their scores are not directly comparable to each other.")
        for name, vs in vid_sets.items():
            print(f"    {name}: {sorted(vs)}")

    used_keys = sorted({k for _, (_, _, used) in per_group.items() for _, k in used})
    all_ref = np.concatenate([ref_cache[k] for k in used_keys], axis=0)

    # --- instrument gate: a paired comparison of identical rows must be 0 ---
    ident, _ = _fd_with_terms(all_ref, all_ref)
    print(f"[FVMD validity] identity check FD(ref,ref) = {ident:.6g} (must be ~0)")
    rows.append(("_identity_ref_vs_ref", "identity", ident, 0.0, 0.0,
                 all_ref.shape[0], all_ref.shape[1], 0))

    # --- diagnostic ONLY: unpaired split-half of the reference -------------
    # NOT a floor for the paired group scores above — see the docstring. Kept
    # because it quantifies how badly an UNPAIRED pooling behaves at this N/D,
    # which is the trap to avoid if anyone later compares across
    # non-corresponding clip sets.
    rng = np.random.default_rng(seed)
    half = all_ref.shape[0] // 2
    nulls = []
    for _ in range(n_null):
        idx = rng.permutation(all_ref.shape[0])
        fd, terms = _fd_with_terms(all_ref[idx[:half]], all_ref[idx[half:2 * half]])
        nulls.append(fd)
        rows.append(("_unpaired_ref_split", "diagnostic_unpaired", fd,
                     terms['mean_term'], terms['cov_term'], half, all_ref.shape[1], 0))
    print(f"[FVMD validity] [diagnostic, NOT a floor] unpaired ref split "
          f"(N={half}/side): mean={np.mean(nulls):.2f} std={np.std(nulls):.2f}")


    # --- jackknife: leave one source video out ---------------------------
    for name, (_, _, used) in per_group.items():
        vids = sorted({k[0] for _, k in used})
        jk = []
        for drop in vids:
            keep = [h for h, k in used if k[0] != drop]
            gt_b, gen_b, u2 = _collect_fvmd_rows(keep, results_dir, cache_dir,
                                                 dataset_dir, ref_cache=ref_cache)
            if not gen_b:
                continue
            fd, _ = _fd_with_terms(np.concatenate(gt_b, axis=0), np.concatenate(gen_b, axis=0))
            jk.append(fd)
            rows.append((name, f"jackknife_drop_{drop}", fd, float('nan'), float('nan'),
                         0, 0, len(u2)))
        if jk:
            print(f"[FVMD validity] {name}: jackknife mean={np.mean(jk):.2f} "
                  f"std={np.std(jk):.2f} range=[{np.min(jk):.2f}, {np.max(jk):.2f}]")

    with open(out_path, 'w') as f:
        f.write("group\tkind\tfvmd\tmean_term\tcov_term\tn_gen\td\tn_used\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"[FVMD validity] wrote {out_path}")
def fid_validity_report(experiment_hash: str, results_dir: str, cache_dir: str,
                        dataset_dir: str, out_path: str, seed: int = 0) -> None:
    """Decide whether a PER-EXPERIMENT FID is meaningful at our sample size, before any
    score is cited. Run this BEFORE backfilling the corpus -- if it fails, the
    per-experiment design is wrong and the backfill is wasted.

    The regime: one experiment gives N ~= 60-90 frames against D = 2048 Inception
    features, so the covariance is badly rank-deficient (N/D ~= 0.03-0.04).

    The obvious defence is the one that rescued set-level FVMD: **our comparison is
    PAIRED** (decoded frames are the same content as their references), so the
    covariance-estimation error is common to both sides and largely cancels inside
    `sqrtm(s1*s2)`. That argument is NOT inherited here, and must not be assumed: FVMD
    held at N=372, D=1024 (ratio 0.36) -- an order of magnitude better than FID's ratio
    here. So it is tested:

      1. identity     -- FD(ref, ref) must be ~0. Instrument gate; if it fails, stop.
                         NOTE this row is deterministic, not data-dependent: with the
                         eps offset, sqrtm((s+eI)(s+eI)) = s+eI, so the score collapses
                         to 2tr(s) - 2tr(s+eI) = -2*eps*D exactly. D=2048 -> -0.0410.
                         (It retro-explains FVMD's reported -0.02: D=1024 -> -0.0205.)
                         It validates the sqrtm path, not the sample size.
      2. estimator noise -- **THE GATE.** FD(ref, ref+noise) at 1%/5%/20%: a known,
                         purely-additive paired perturbation against a clean baseline.
                         Must rise monotonically from ~0. This isolates the estimator's
                         ability to resolve a small paired difference at this N, which
                         is the actual question, and it mirrors the FVMD precedent
                         (4 / 103 / 1645 from an identity of ~0).
      3. decoded+noise -- SECONDARY, and deliberately not a gate. FD(ref, decoded+noise)
                         conflates two effects and must not be read as an estimator
                         check: the decoded video is already far from the reference, and
                         at low bitrate it is *blurred*, so added noise injects
                         high-frequency energy that can move its Inception texture
                         statistics back TOWARD the detailed reference. Measured on
                         tennis fg_bbox: 386 (+1%) -> 369 (+5%) -> 381 (+20%),
                         non-monotone, with the mean term falling 300 -> 254 -> 247.
                         That is the metric conflating noise with texture -- the same
                         effect that makes FID prefer hallucinated detail to blur -- and
                         it is a property of FID, not evidence about N. Reported because
                         it is a real caveat on citing FID for generative restoration.
      4. unpaired split -- DIAGNOSTIC ONLY, reported as `_diagnostic_unpaired_split`.
                         An unpaired split is NOT a floor for the paired scores above.
                         The FVMD version of this row was misread as a floor once
                         already, producing a "null" larger than the scores it was meant
                         to bound -- an impossibility that is how the error was caught.
                         Expect paired scores to sit legitimately below it.

    Pre-registered fallback, recorded before the numbers are seen: if the identity check
    is non-zero, or the noise ladder is not monotone, or the 1% score is not small
    relative to the between-method gaps we intend to cite, then per-experiment FID does
    not survive at N ~= 60-90 and must be DROPPED in favour of set-level pooling (frames
    pooled across the 6 videos per method, as set-level FVMD does, giving N ~= 450-540
    and restoring the FVMD regime). In that case `fid_fg_bbox` becomes set-level only,
    and the existing per-experiment `overall.fid` carries the same warning.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    result_path = os.path.join(results_dir, experiment_hash, "result.json")
    with open(result_path) as f:
        data = json.load(f)
    cfg = data['config']
    video_name, width, height = cfg['video'], cfg['width'], cfg['height']
    _, refs, _ = _get_refs_cached(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")
    ufo_masks = _get_masks_cached(video_name, width, height, cfg.get('block_size', 8),
                                  ref_frames_dir, cache_dir)
    decs = load_frames_from_video(data['output_video'])
    n = min(len(refs), len(decs), len(ufo_masks))
    refs, decs = refs[:n], decs[:n]
    masks = [ufo_masks[i] > 127 for i in range(n)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FrechetInceptionDistance(feature=2048).to(device)
    rng = np.random.default_rng(seed)
    rows = []
    D = 2048

    print(f"[FID validity] {experiment_hash} ({video_name}, {cfg.get('component')}): "
          f"N={n} frames/side vs D={D} -> N/D={n/D:.3f}")
    fg_frac = float(np.mean([m.mean() for m in masks]))
    print(f"[FID validity] mean true FG fraction: {fg_frac:.3f}")

    def _crop(frames, use_bbox):
        """Whole-frame, or per-frame tight FG bbox (paired: same box on both sides)."""
        if not use_bbox:
            return frames
        out = []
        for f, m in zip(frames, masks):
            bb = _fg_tight_bbox(m, width, height)
            if bb is not None:
                y1, y2, x1, x2 = bb
                out.append(f[y1:y2, x1:x2])
        return out

    for kind, use_bbox in (("whole_frame", False), ("fg_bbox", True)):
        r_f = _crop(refs, use_bbox)
        d_f = _crop(decs, use_bbox)
        if len(r_f) < 2:
            print(f"[FID validity] {kind}: <2 usable frames, skipping")
            continue
        ref_feats = _inception_feats(r_f, device, model)
        n_side = len(ref_feats)

        # 1. identity gate
        ident, terms = _fd_with_terms(ref_feats, ref_feats)
        rows.append((kind, "identity_FD(ref,ref)", ident, terms['mean_term'], terms['cov_term'], n_side, D))
        print(f"[FID validity] {kind:11s} identity FD(ref,ref) = {ident:.4f} (must be ~0) "
              f"-> {'PASS' if abs(ident) < 1.0 else 'FAIL'}")

        # real decoded-vs-reference score, for scale
        dec_feats = _inception_feats(d_f, device, model)
        fd_real, terms = _fd_with_terms(ref_feats, dec_feats)
        rows.append((kind, "paired_decoded", fd_real, terms['mean_term'], terms['cov_term'], n_side, D))
        print(f"[FID validity] {kind:11s} paired decoded-vs-ref = {fd_real:.2f} "
              f"(mean {terms['mean_term']:.2f} / cov {terms['cov_term']:.2f})")

        def _noisy(frames, pct):
            return [np.clip(f.astype(np.float32) + rng.uniform(-pct*255, pct*255, f.shape),
                            0, 255).astype(np.uint8) for f in frames]

        # 2. THE GATE: estimator sensitivity, FD(ref, ref+noise), from a clean baseline.
        est_ladder = []
        for pct in (0.01, 0.05, 0.20):
            nf = _inception_feats(_noisy(r_f, pct), device, model)
            fd, t = _fd_with_terms(ref_feats, nf)
            est_ladder.append(fd)
            rows.append((kind, f"estimator_noise_{int(pct*100)}pct", fd, t['mean_term'], t['cov_term'], n_side, D))
            print(f"[FID validity] {kind:11s} GATE estimator ref-vs-ref+{int(pct*100):2d}% = {fd:.2f} "
                  f"(mean {t['mean_term']:.2f} / cov {t['cov_term']:.2f})")
        mono = all(est_ladder[i] < est_ladder[i+1] for i in range(len(est_ladder)-1))
        print(f"[FID validity] {kind:11s} GATE estimator ladder monotone from ~0? "
              f"{'PASS' if mono else 'FAIL'} ({' < '.join(f'{x:.1f}' for x in est_ladder)})")

        # 3. SECONDARY (not a gate): FD(ref, decoded+noise). Conflates estimator
        # behaviour with FID's noise-vs-texture confusion -- see the docstring.
        dec_ladder = []
        for pct in (0.01, 0.05, 0.20):
            nf = _inception_feats(_noisy(d_f, pct), device, model)
            fd, t = _fd_with_terms(ref_feats, nf)
            dec_ladder.append(fd)
            rows.append((kind, f"secondary_decoded_plus_noise_{int(pct*100)}pct", fd, t['mean_term'], t['cov_term'], n_side, D))
            print(f"[FID validity] {kind:11s} secondary decoded+{int(pct*100):2d}% = {fd:.2f} "
                  f"(mean {t['mean_term']:.2f} / cov {t['cov_term']:.2f})")
        dmono = all(dec_ladder[i] < dec_ladder[i+1] for i in range(len(dec_ladder)-1))
        print(f"[FID validity] {kind:11s} secondary decoded+noise monotone? {'yes' if dmono else 'NO'} "
              f"({' -> '.join(f'{x:.1f}' for x in dec_ladder)})"
              f"{'' if dmono else '  <- FID reading noise as texture, NOT an N failure'}")

        # 3. unpaired split -- DIAGNOSTIC ONLY, never a floor
        idx = rng.permutation(n_side)
        half = n_side // 2
        if half >= 2:
            fd, t = _fd_with_terms(ref_feats[idx[:half]], ref_feats[idx[half:2*half]])
            rows.append((kind, "_diagnostic_unpaired_split", fd, t['mean_term'], t['cov_term'], half, D))
            print(f"[FID validity] {kind:11s} _diagnostic_unpaired_split = {fd:.2f} at N={half}/side "
                  f"-- NOT a floor for the paired scores above")

    with open(out_path, 'w') as f:
        f.write("kind\trow\tfd\tmean_term\tcov_term\tn_per_side\td\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"[FID validity] wrote {out_path}")
