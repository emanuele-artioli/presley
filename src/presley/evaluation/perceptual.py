"""Learned perceptual metrics: LPIPS, DISTS, FID.

These carry the paper's quality argument, since PSNR rewards a flat
fill over hallucinated detail. Masked variants score true regions;
the bbox variants are crops and are named so their key says it."""

import numpy as np
import cv2
import torch
from typing import Dict, Any, List
from presley.preprocessing import get_reference_frames, get_ufo_masks
_REF_CACHE: Dict[Any, Any] = {}
_MASK_CACHE: Dict[Any, Any] = {}
_DISTS_CACHE: Dict[str, Any] = {}
from presley.evaluation.masked import _fg_tight_bbox


def _get_dists_model(device: str):
    if device not in _DISTS_CACHE:
        from DISTS_pytorch import DISTS
        _DISTS_CACHE[device] = DISTS().to(device).eval()
    return _DISTS_CACHE[device]
def calculate_lpips(refs: List[np.ndarray], decs: List[np.ndarray], device: str) -> List[float]:
    import lpips
    model = lpips.LPIPS(net='alex').to(device)
    scores = []
    with torch.no_grad():
        for r, d in zip(refs, decs):
            r_t = torch.from_numpy(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
            d_t = torch.from_numpy(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
            scores.append(model(r_t, d_t).item())
    return scores
def calculate_lpips_masked(refs: List[np.ndarray], decs: List[np.ndarray],
                           masks: List[np.ndarray], device: str) -> Dict[str, List[float]]:
    """Per-frame FG/BG/overall LPIPS using spatial-mode LPIPS.

    lpips(spatial=True) returns a per-pixel distance map at input resolution; we
    average it over the UFO mask (FG), its complement (BG), and the whole frame
    (overall). This is a true region-restricted perceptual metric — no bbox
    cropping or pixel-zeroing artifacts — and it's the FG number the paper argues.
    masks[i] is a >127 boolean foreground mask.

    Frames with an empty mask yield NaN, not 0.0: 0.0 is a *perfect* LPIPS score, and
    averaging fabricated zeros into foreground.lpips_mean would bias the paper's headline
    FG metric optimistically. Same convention as calculate_dists_masked and
    _fvmd_on_frames (see the fc203a9980dad7d3 fake-0.0 incident). Aggregate with nanmean.
    """
    import lpips
    model = lpips.LPIPS(net='alex', spatial=True).to(device)
    fg, bg, ov = [], [], []
    with torch.no_grad():
        for r, d, m in zip(refs, decs, masks):
            r_t = torch.from_numpy(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
            d_t = torch.from_numpy(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
            smap = model(r_t, d_t).squeeze().cpu().numpy()  # [H, W]
            if smap.shape != m.shape:
                smap = cv2.resize(smap, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
            ov.append(float(smap.mean()))
            fg.append(float(smap[m].mean()) if np.any(m) else float('nan'))
            bg.append(float(smap[~m].mean()) if np.any(~m) else float('nan'))
    return {"foreground": fg, "background": bg, "overall": ov}
def calculate_fid(refs, decs, device):
    # NOTE: FrechetInceptionDistance is a stateful accumulator -- it must be constructed
    # per call (or .reset()), never cached like _get_dists_model, or every experiment in
    # a backfill pass would be pooled into one distribution.
    from torchmetrics.image.fid import FrechetInceptionDistance
    import torch
    fid = FrechetInceptionDistance(feature=2048).to(device)
    batch_size = 16
    for i in range(0, len(refs), batch_size):
        r_batch = refs[i:i+batch_size]
        d_batch = decs[i:i+batch_size]
        r_t = torch.from_numpy(np.array([cv2.cvtColor(r, cv2.COLOR_BGR2RGB) for r in r_batch])).permute(0, 3, 1, 2).byte().to(device)
        d_t = torch.from_numpy(np.array([cv2.cvtColor(d, cv2.COLOR_BGR2RGB) for d in d_batch])).permute(0, 3, 1, 2).byte().to(device)
        fid.update(r_t, real=True)
        fid.update(d_t, real=False)
    return float(fid.compute().item())
def calculate_fid_bbox(refs: List[np.ndarray], decs: List[np.ndarray],
                       masks: List[np.ndarray], device: str) -> Dict[str, Any]:
    """Best-effort localised FID over per-frame tight FG bbox crops.

    THIS IS NOT A FOREGROUND METRIC, and its key (`fid_fg_bbox`) must always be written
    and cited by that full name. FID pools Inception down to a single 2048-d vector, so
    there is no spatial axis left to mask and no principled FG-FID exists -- unlike
    DISTS (see `calculate_dists_masked`) or LPIPS, whose spatial maps can be
    mask-weighted. The best available improvement is to replace the union bbox (100% of
    the frame on india, 58.6% on tennis vs 4.0% true FG) with a per-frame tight box,
    which is 1.3-3.8x tighter but still ~74% background on tennis. Cite `dists_fg` or
    FG-LPIPS for the foreground claim; cite this only as a corroborating signal.

    Crops vary in size per frame, so they cannot be batched. We therefore feed them one
    at a time at native size and let torchmetrics resize -- we deliberately do NOT resize
    to 299 ourselves. torch_fidelity's Inception extractor already resizes any input to
    299x299 internally (feature_extractor_inceptionv3.py:111, TF-compat bilinear), so an
    explicit resize would resample twice and would put this metric on a different
    preprocessing path than `overall.fid`, which is fed native frames. Do not "optimise"
    a resize back in.

    Because box size varies per frame while Inception's internal resize is anisotropic,
    this metric carries scale variance that whole-frame FID does not. The returned
    diagnostics quantify it. Note the box is derived from the *reference* mask and
    applied identically to reference and decoded frames, so the resize distortion is the
    same on both sides of every frame: it inflates within-distribution variance rather
    than biasing one side.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048).to(device)
    h, w = refs[0].shape[:2]
    n_used = n_skipped = 0
    areas, bg_fracs = [], []
    for r, d, m in zip(refs, decs, masks):
        bb = _fg_tight_bbox(m, w, h)
        if bb is None:
            # Skip on BOTH sides -- never asymmetrically, or the paired structure that
            # the small-sample validity argument rests on is broken.
            n_skipped += 1
            continue
        y1, y2, x1, x2 = bb
        r_c, d_c, m_c = r[y1:y2, x1:x2], d[y1:y2, x1:x2], m[y1:y2, x1:x2]
        box_px = (y2 - y1) * (x2 - x1)
        areas.append(box_px)
        bg_fracs.append(1.0 - float(m_c.sum()) / box_px)
        for arr, real in ((r_c, True), (d_c, False)):
            t = torch.from_numpy(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).byte().to(device)
            fid.update(t, real=real)
        n_used += 1

    if n_used < 2:
        # FID's covariance is undefined at N<2. Return NaN, never 0.0 (a perfect score).
        score = float('nan')
    else:
        score = float(fid.compute().item())
    side = np.sqrt(np.asarray(areas, dtype=np.float64)) if areas else np.array([0.0])
    return {
        "fid": score,
        "n_used": n_used,
        "n_skipped_empty": n_skipped,
        "area_frac_mean": float(np.mean(areas) / (w * h)) if areas else float('nan'),
        "area_frac_std": float(np.std(areas) / (w * h)) if areas else float('nan'),
        # dimensionless scale jitter across frames -- the artifact whole-frame FID lacks
        "scale_cv": float(side.std() / side.mean()) if areas and side.mean() > 0 else float('nan'),
        # how much background is still inside the box: this is the number that justifies
        # the key being named fid_fg_bbox and not fid_fg
        "bg_frac_mean": float(np.mean(bg_fracs)) if bg_fracs else float('nan'),
    }
def calculate_dists(refs: List[np.ndarray], decs: List[np.ndarray], device: str) -> List[float]:
    from DISTS_pytorch import DISTS
    model = DISTS().to(device)
    scores = []
    with torch.no_grad():
        for r, d in zip(refs, decs):
            r_t = torch.from_numpy(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
            d_t = torch.from_numpy(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
            scores.append(model(r_t, d_t).item())
    return scores
def _dists_layer_weights(mask_t, feats):
    """Area-downsample a [1,1,H,W] weight map onto each DISTS layer's feature grid.

    Area-averaging makes each entry the fraction of that feature location covered by
    the mask, which is the natural weight for the pooled statistics below. feats[0]
    is the input image at full resolution, so its map is the mask itself.

    Sizes are read from the tensors, never assumed: for a 360x640 input the stages
    are 360/180/90/45/22 rows (L2pooling is 3-tap, stride 2: 45 -> 22).
    """
    import torch.nn.functional as F
    out = []
    for f in feats:
        size = tuple(f.shape[-2:])
        w = mask_t if tuple(mask_t.shape[-2:]) == size else F.interpolate(mask_t, size=size, mode='area')
        out.append(w)
    return out
def calculate_dists_masked(refs: List[np.ndarray], decs: List[np.ndarray],
                           masks: List[np.ndarray], device: str) -> Dict[str, List[float]]:
    """Per-frame FG/BG/overall DISTS with MASK-WEIGHTED spatial pooling.

    Stock `DISTS.forward` pools every layer with `.mean([2,3])` -- a global spatial
    mean/var/cov per channel. This replaces that pooling with a mask-weighted one
    (weighted mean sum(wx)/sum(w), var sum(w(x-mu)^2)/sum(w), cov sum(wxy)/sum(w) - mu_x*mu_y),
    keeping the pretrained alpha/beta weights untouched. It is the exact analogue of
    `calculate_lpips_masked` and returns the same shape.

    Uniform weights reproduce stock DISTS to <1e-5 (float32 reduction order only) --
    that equivalence is the correctness gate for this function.

    This supersedes the old `_fg_union_bbox`-cropped "FG-DISTS", which was not a
    foreground metric: the union bbox is 100% of the frame on india (its FG-DISTS was
    bit-identical to overall-DISTS, verified across 16/16 experiments) and 58.6% on
    tennis against a 4.0% true FG. See TECHNICAL_REPORT_PIPELINE_INFRA.md 2026-07-16.

    Caveat, and it must be stated wherever this is reported: this is mask-*weighted*,
    not mask-*isolated*. Background locations get exactly zero weight, but VGG units at
    stages 4-5 have receptive fields spanning tens of pixels, so an in-mask feature
    still integrates some surrounding background. It measures the foreground in
    context. The same is true of the FG-LPIPS we already report, and it is categorically
    different from the union-bbox defect, where background *locations* were pooled in
    directly.

    masks[i] is a >127 boolean foreground mask. Frames with an empty mask yield NaN,
    not 0.0 -- 0.0 is a perfect DISTS score, and this repo has already been burned once
    by a fabricated 0.0 (fc203a9980dad7d3, a swallowed exception). Aggregate with nanmean.
    """
    model = _get_dists_model(device)
    c1 = c2 = 1e-6
    # Normalisation reproduced verbatim from DISTS.forward.
    w_sum = model.alpha.sum() + model.beta.sum()
    alpha = torch.split(model.alpha / w_sum, model.chns, dim=1)
    beta = torch.split(model.beta / w_sum, model.chns, dim=1)

    def _pooled_score(feats0, feats1, weights) -> float:
        dist1 = 0
        dist2 = 0
        for k in range(len(model.chns)):
            x, y, w = feats0[k], feats1[k], weights[k]
            W = w.sum([2, 3], keepdim=True).clamp_min(1e-8)
            x_mean = (w * x).sum([2, 3], keepdim=True) / W
            y_mean = (w * y).sum([2, 3], keepdim=True) / W
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = (w * (x - x_mean) ** 2).sum([2, 3], keepdim=True) / W
            y_var = (w * (y - y_mean) ** 2).sum([2, 3], keepdim=True) / W
            xy_cov = (w * x * y).sum([2, 3], keepdim=True) / W - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)
        return float((1 - (dist1 + dist2).squeeze()).item())

    fg, bg, ov = [], [], []
    with torch.no_grad():
        for r, d, m in zip(refs, decs, masks):
            r_t = torch.from_numpy(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
            d_t = torch.from_numpy(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
            feats0, feats1 = model.forward_once(r_t), model.forward_once(d_t)

            m_t = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            if tuple(m_t.shape[-2:]) != tuple(r_t.shape[-2:]):
                m_t = torch.nn.functional.interpolate(m_t, size=tuple(r_t.shape[-2:]), mode='area')
            fg_w = _dists_layer_weights(m_t, feats0)
            # Area-averaging is linear, so 1 - area_avg(mask) == area_avg(1 - mask) exactly.
            bg_w = [1.0 - w for w in fg_w]
            ov_w = [torch.ones_like(w) for w in fg_w]

            ov.append(_pooled_score(feats0, feats1, ov_w))
            fg.append(_pooled_score(feats0, feats1, fg_w) if m.any() else float('nan'))
            bg.append(_pooled_score(feats0, feats1, bg_w) if (~m).any() else float('nan'))
    return {"foreground": fg, "background": bg, "overall": ov}
def _inception_feats(frames: List[np.ndarray], device: str, fid_model=None) -> np.ndarray:
    """[N, 2048] Inception pool3 features, extracted through the SAME module torchmetrics
    FID scores with (`fid.inception`) -- so this measures the actual features, not a
    lookalike. Frames are fed one at a time at native size; the extractor resizes to
    299x299 itself (see calculate_fid_bbox)."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    model = fid_model if fid_model is not None else FrechetInceptionDistance(feature=2048).to(device)
    out = []
    with torch.no_grad():
        for f in frames:
            t = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).byte().to(device)
            out.append(model.inception(t).squeeze(0).cpu().numpy())
    return np.stack(out)
