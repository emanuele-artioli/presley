#!/usr/bin/env python3
"""M1 — is post-restoration damage predictable at transmit time? Runs ONCE.

Pre-registered in `docs/PREREG_M1_RESTORABILITY.md`. The mechanism argument for
looking here rather than at a better cost proxy:

  * the bits-cost NUMERATOR is nearly saturated -- the complexity score captures
    0.833 of a leave-one-superblock-out oracle's bits against a random null of
    0.402, bounding the remaining headroom at ~5% of bitrate;
  * the restorability DENOMINATOR is unmodelled and disperses 4.9 dB
    (Real-ESRGAN) and 8.4 dB (ProPainter) between its 10th and 90th percentile
    WITHIN a single run.

So the headroom is here or nowhere. "Transmit-time-computable" is the binding
constraint: a feature needing the restored output is useless for selection.

DEVIATION FROM THE PRE-REGISTRATION, declared rather than quietly taken. The
document states the corpus is "8 probe_block_damage runs". That is factually
wrong about which data exists: none of the eight probe runs appear in
`results/block_damage_s1b.npz`, which instead holds 143 ordinary restored runs
mined by `tools/mine_block_damage.py`. Restricting to the two restorers the
article actually reports gives **120 runs across 13 videos**.

The DESIGN is unchanged -- same five declared features, same within-run rank
statistic, same sign test over runs, same stopping rule. Only the run count
differs, and it differs upward. That cuts both ways and is worth saying: a
larger n can detect a smaller true effect, so a null here is a stronger null
than the registered n=8 would have produced, and cannot be excused as
underpowered.

Confounds are handled structurally, as in R1: within-run ranks mean video,
dataset provenance, duration, resolution, codec, QP and operating point are all
constant within the unit and difference out exactly.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from math import comb

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from presley.blockdamage import pool_to_superblocks  # noqa: E402

# The five features, declared in the pre-registration. Do NOT add a sixth after
# seeing these five -- that is candidate shopping, and the Holm family is sized
# for exactly this list, losers included.
FEATURES = ("sc_mean", "tc_mean", "sc_var", "tc_var", "frame_edge")
HOLM_K = len(FEATURES)
ALPHA = 0.05
RESTORERS = ("realesrgan", "propainter")

# Pre-registered bounds.
RHO_PLAUSIBLE = (0.15, 0.55)
RHO_LEAKAGE_ALARM = 0.80      # a transmit-time feature this good is likely leakage
TRIVIAL_RHO = 0.10


def sign_p(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) * 0.5 ** n for i in range(lo + 1)))


def spearman(a, b):
    """Rank correlation, average ranks for ties. No SciPy dependency."""
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average tied ranks
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def run_geometry(run_hash):
    """(width, height) from the run's own config, or None."""
    p = REPO_ROOT / "results" / run_hash / "result.json"
    if not p.is_file():
        return None
    try:
        cfg = json.loads(p.read_text()).get("config") or {}
    except (ValueError, OSError):
        return None
    w, h = cfg.get("width"), cfg.get("height")
    return (int(w), int(h)) if w and h else None


def evca_superblocks(video, width, height, block_size, n_frames):
    """SC/TC pooled onto the 64x64 superblock grid, or None if uncached."""
    base = REPO_ROOT / "cache" / f"{video}_{width}x{height}_bs{block_size}"
    if not base.is_dir():
        return None
    out = {}
    for name, key in (("evca_SC_blocks.csv", "sc"), ("evca_TC_blocks.csv", "tc")):
        p = base / name
        if not p.is_file():
            return None
        arr = np.loadtxt(p, delimiter=",", skiprows=1)   # (blocks, frames)
        if arr.ndim == 1:
            arr = arr[:, None]
        nby, nbx = height // block_size, width // block_size
        if arr.shape[0] < nby * nbx:
            return None
        f = min(n_frames, arr.shape[1])
        cube = arr[:nby * nbx, :f].T.reshape(f, nby, nbx)   # (F, BY, BX)
        out[key] = pool_to_superblocks(cube, block_size, height, width)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(REPO_ROOT / "results" / "block_damage_s1b.npz"))
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    run, vid, res = z["run"], z["video"], z["restorer"]
    sy, sx, frame, bs = z["sy"], z["sx"], z["frame"], z["block_size"]
    dmg = z["delta_psnr"]

    keep = np.isin(res, RESTORERS) & np.isfinite(dmg)
    idx_by_run = collections.defaultdict(list)
    for i in np.nonzero(keep)[0]:
        idx_by_run[str(run[i])].append(i)

    print("=" * 78)
    print("M1 -- does any transmit-time feature predict post-restoration damage?")
    print("Unit: the run. Statistic: within-run Spearman rho vs damage rank.")
    print("Restricted to superblocks that were ACTUALLY DEGRADED -- see the")
    print("leakage control in the source; over all superblocks the correlation")
    print("is inflated by SC also driving selection (rho 0.79).")
    print(f"Features (declared, k={HOLM_K}): {', '.join(FEATURES)}")
    print("=" * 78)

    per_feature = {f: [] for f in FEATURES}
    used, skipped = 0, collections.Counter()
    for r, idxs in sorted(idx_by_run.items()):
        idxs = np.array(idxs)
        v = str(vid[idxs[0]]); b = int(bs[idxs[0]])
        # Read the true geometry from the run itself. Inferring it from the
        # superblock grid is wrong: a 360-pixel height gives 6 superblock rows,
        # so the grid implies 384 and every cache lookup misses.
        geom = run_geometry(r)
        if geom is None:
            skipped[f"no result.json for {r[:8]}"] += 1
            continue
        width, height = geom
        ev = evca_superblocks(v, width, height, b, int(frame[idxs].max()) + 1)
        if ev is None:
            skipped[f"no EVCA cache for {v} {width}x{height} bs{b}"] += 1
            continue
        d = dmg[idxs]
        fr, yy, xx = frame[idxs], sy[idxs], sx[idxs]
        ok = (fr < ev["sc"].shape[0]) & (yy < ev["sc"].shape[1]) & (xx < ev["sc"].shape[2])
        if ok.sum() < 30:
            skipped[f"grid mismatch for {v} bs{b}"] += 1
            continue
        fr, yy, xx, d = fr[ok], yy[ok], xx[ok], d[ok]
        strength = z["strength_frac"][idxs][ok]
        # CONTROL, added after a pre-registered bound fired -- declared as an
        # addition, not back-dated into the design. Selection picks HIGH-SC
        # blocks, and only selected blocks carry damage, so SC correlates with
        # "was this degraded" before it correlates with anything about
        # restorability. Over all superblocks rho(SC, damage) is +0.68, outside
        # the registered 0.15-0.55 band; rho(SC, selection) is +0.79, which is
        # the leak. Restricting to blocks that were ACTUALLY degraded removes it.
        degraded = strength > 0
        if degraded.sum() < 30:
            skipped[f"too few degraded superblocks for {v}"] += 1
            continue
        fr, yy, xx, d = fr[degraded], yy[degraded], xx[degraded], d[degraded]
        sc, tc = ev["sc"][fr, yy, xx], ev["tc"][fr, yy, xx]
        # per-superblock variance across frames, broadcast back to each row
        scv = ev["sc"].var(axis=0)[yy, xx]
        tcv = ev["tc"].var(axis=0)[yy, xx]
        edge = ((yy == 0) | (xx == 0) | (yy == ev["sc"].shape[1] - 1)
                | (xx == ev["sc"].shape[2] - 1)).astype(float)
        feats = {"sc_mean": sc, "tc_mean": tc, "sc_var": scv,
                 "tc_var": tcv, "frame_edge": edge}
        for f in FEATURES:
            per_feature[f].append(spearman(feats[f], d))
        used += 1

    if skipped:
        print("\nruns skipped (reported, not silently dropped):")
        for k, c in skipped.most_common():
            print(f"   {c:4d}  {k}")
    print(f"\nruns used: {used}")
    if used < 6:
        print("too few runs for a verdict.")
        return 1

    print(f"\n{'feature':14}{'n':>5}{'median rho':>12}{'|rho|>0 on':>12}"
          f"{'sign p':>10}{'p_Holm':>10}")
    results = []
    for f in FEATURES:
        rs = per_feature[f]
        n = len(rs)
        pos = sum(1 for x in rs if x > 0)
        # two-tailed: is the SIGN consistent, in either direction?
        p = sign_p(pos, n)
        ph = min(1.0, p * HOLM_K)
        med = float(np.median(rs))
        results.append((f, n, med, pos, p, ph))
        print(f"{f:14}{n:>5}{med:>+12.4f}{f'{pos}/{n}':>12}{p:>10.4f}{ph:>10.4f}")

    best = max(results, key=lambda t: abs(t[2]))
    print(f"\nstrongest by |median rho|: {best[0]} at {best[2]:+.4f}")

    print("\nPre-registered bound status:")
    lo, hi = RHO_PLAUSIBLE
    print(f"  best |median rho| plausible {lo}-{hi}: got {abs(best[2]):.4f} -> "
          f"{'in band' if lo <= abs(best[2]) <= hi else 'OUTSIDE'}")
    if abs(best[2]) > RHO_LEAKAGE_ALARM:
        print(f"  *** ALARM: |rho| > {RHO_LEAKAGE_ALARM}. A transmit-time feature that")
        print("      nearly determines post-restoration damage is more likely leakage")
        print("      than a finding -- check the feature is not derived from output.")
    if abs(best[2]) < TRIVIAL_RHO:
        print(f"  best |median rho| < {TRIVIAL_RHO}: no feature carries usable signal")

    fires = best[5] <= ALPHA and abs(best[2]) >= TRIVIAL_RHO
    print("\n" + "=" * 78)
    if fires:
        print(f"M1 FIRES on {best[0]}: post-restoration damage is partly predictable")
        print("at transmit time. This names the term the selection objective is missing.")
        print("")
        print("And the sign is the uncomfortable part. The score already selects on")
        print("spatial complexity because complex blocks cost the most BITS. The same")
        print("quantity predicts how badly a block comes back, POSITIVELY. So the")
        print("current objective does not merely omit restorability -- it selects")
        print("preferentially the blocks that survive restoration worst, and the two")
        print("terms it does have are both proxies for the same thing. That is a")
        print("mechanism for why alpha and beta are inert rather than merely weak.")
    else:
        print("M1 DOES NOT FIRE.")
        print("This is the outcome the pre-registration expected, and it is the more")
        print("valuable one: it upgrades the selection null from 'alpha and beta are")
        print("inert' -- a parameter ablation -- to 'the missing term is not merely")
        print("unmodelled, it is not predictable from any transmit-time signal we can")
        print("compute', which is a mechanism claim with a stated feature family.")
        print("")
        print("Do NOT add a sixth feature now. That is candidate shopping, and the")
        print("Holm family was sized for exactly these five.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
