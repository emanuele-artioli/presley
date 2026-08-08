#!/usr/bin/env python3
"""W1g screen: would a restorability-aware objective select different blocks?

The article's diagnosis is that selection maximizes the wrong thing. It ranks
blocks by a removability score R, which estimates only how many BITS a block
costs; what it should maximize is a ratio -- bits freed per unit of damage that
survives restoration -- and the denominator was never modelled. M1 then showed
the denominator IS predictable at transmit time (sc_mean, within-run Spearman
+0.506, same sign on 120/120 runs), and predictable with the sign that makes
the current objective actively wrong rather than merely incomplete.

The obvious next step is to build the corrected rule and race it. Before
spending a GPU campaign on that, this screens the one way it can fail for free.

**Both terms are monotone in complexity.** R is high for complex blocks because
they cost the most bits; predicted damage is also high for complex blocks,
because that is what M1 measured. If the two rise together closely enough, the
ratio R/D is nearly constant in complexity, the corrected ranking is a
re-ordering of noise, and the corrected rule selects almost exactly the blocks
the current one already selects. In that case no campaign can show a
difference, and the honest result is this screen rather than a null from an
underpowered race.

So this reports, per run:

  * the skill of the damage predictor, fit on OTHER videos (held out by video,
    never by run -- runs of the same clip share content and would leak);
  * the rank correlation between the current ranking and the corrected one;
  * the overlap of the two rules' selected sets at the operating budget.

Read the overlap. Near 1.0 means the corrected objective is not a different
rule and the campaign is pointless. Well below 1.0 means it is a different rule
and the race is worth running.

This decides only whether to run the campaign. It is NOT evidence about which
rule is better -- that needs the runs.

Usage:
    python tools/analyze_corrected_objective.py
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# `results/` and `cache/` are gitignored and exist only in the main checkout,
# so a tool running from a worktree has to be told where they are. Set by
# --data-root; defaults to the tree the script lives in.
DATA_ROOT = REPO_ROOT

from presley.blockdamage import pool_to_superblocks  # noqa: E402

FEATURES = ("sc_mean", "tc_mean", "sc_var", "tc_var", "frame_edge")
RESTORERS = ("realesrgan", "propainter")
# The operating budget every reported run uses.
BUDGET = 0.25
# Ridge penalty: the features are collinear by construction (sc_mean and
# tc_mean both track complexity), so an unpenalized fit is unstable.
RIDGE = 1.0


def spearman(a, b):
    """Rank correlation, average ranks for ties. No SciPy dependency."""
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def run_geometry(run_hash):
    p = DATA_ROOT / "results" / run_hash / "result.json"
    if not p.is_file():
        return None
    try:
        cfg = json.loads(p.read_text()).get("config") or {}
    except (ValueError, OSError):
        return None
    w, h = cfg.get("width"), cfg.get("height")
    return (int(w), int(h)) if w and h else None


def evca_superblocks(video, width, height, block_size, n_frames):
    base = DATA_ROOT / "cache" / f"{video}_{width}x{height}_bs{block_size}"
    if not base.is_dir():
        return None
    out = {}
    for name, key in (("evca_SC_blocks.csv", "sc"), ("evca_TC_blocks.csv", "tc")):
        p = base / name
        if not p.is_file():
            return None
        arr = np.loadtxt(p, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[:, None]
        nby, nbx = height // block_size, width // block_size
        if arr.shape[0] < nby * nbx:
            return None
        f = min(n_frames, arr.shape[1])
        cube = arr[:nby * nbx, :f].T.reshape(f, nby, nbx)
        out[key] = pool_to_superblocks(cube, block_size, height, width)
    return out


def collect(npz_path):
    """Per-run feature matrices, damage, and the score the system ranks by."""
    z = np.load(npz_path, allow_pickle=True)
    run, vid, res = z["run"], z["video"], z["restorer"]
    sy, sx, frame, bs = z["sy"], z["sx"], z["frame"], z["block_size"]
    dmg, strength = z["delta_psnr"], z["strength_frac"]

    keep = np.isin(res, RESTORERS) & np.isfinite(dmg)
    idx_by_run = collections.defaultdict(list)
    for i in np.nonzero(keep)[0]:
        idx_by_run[str(run[i])].append(i)

    out, skipped = [], collections.Counter()
    for r, idxs in sorted(idx_by_run.items()):
        idxs = np.array(idxs)
        video = str(vid[idxs[0]])
        block_size = int(bs[idxs[0]])
        geom = run_geometry(r)
        if geom is None:
            skipped["no result.json"] += 1
            continue
        width, height = geom
        ev = evca_superblocks(video, width, height, block_size, int(frame[idxs].max()) + 1)
        if ev is None:
            skipped[f"no EVCA cache for {video} {width}x{height} bs{block_size}"] += 1
            continue
        fr, yy, xx = frame[idxs], sy[idxs], sx[idxs]
        ok = ((fr < ev["sc"].shape[0]) & (yy < ev["sc"].shape[1]) & (xx < ev["sc"].shape[2]))
        if ok.sum() < 30:
            skipped["grid mismatch"] += 1
            continue
        fr, yy, xx = fr[ok], yy[ok], xx[ok]
        d, s = dmg[idxs][ok], strength[idxs][ok]

        # Same leakage control as M1: only blocks that were actually degraded
        # carry damage, and selection itself is driven by complexity, so
        # including untouched blocks measures selection rather than
        # restorability.
        deg = s > 0
        if deg.sum() < 30:
            skipped["too few degraded superblocks"] += 1
            continue
        fr, yy, xx, d = fr[deg], yy[deg], xx[deg], d[deg]

        sc, tc = ev["sc"][fr, yy, xx], ev["tc"][fr, yy, xx]
        feats = np.column_stack([
            sc, tc,
            ev["sc"].var(axis=0)[yy, xx],
            ev["tc"].var(axis=0)[yy, xx],
            ((yy == 0) | (xx == 0) | (yy == ev["sc"].shape[1] - 1)
             | (xx == ev["sc"].shape[2] - 1)).astype(float),
        ])
        # The bits proxy the system actually ranks by is the removability
        # score, whose complexity term at this grid is the same convex blend of
        # sc and tc. alpha = 0.5 is the operating point of every reported run.
        bits_proxy = 0.5 * sc + 0.5 * tc
        out.append({"run": r, "video": video, "X": feats, "damage": d,
                    "bits": bits_proxy})
    return out, skipped


def fit_ridge(X, y, penalty=RIDGE):
    """Standardized ridge. Returns a predict(X) closure."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xs = np.column_stack([Xs, np.ones(len(Xs))])
    reg = penalty * np.eye(Xs.shape[1])
    reg[-1, -1] = 0.0                      # never penalize the intercept
    beta = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ y)

    def predict(Xn):
        Xn = (Xn - mu) / sd
        return np.column_stack([Xn, np.ones(len(Xn))]) @ beta
    return predict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT),
                    help="tree holding results/ and cache/ (the main checkout)")
    ap.add_argument("--npz")
    args = ap.parse_args()

    global DATA_ROOT
    DATA_ROOT = pathlib.Path(args.data_root).resolve()
    npz = args.npz or str(DATA_ROOT / "results" / "block_damage_s1b.npz")

    runs, skipped = collect(npz)
    if skipped:
        print("runs skipped (reported, not silently dropped):")
        for k, c in skipped.most_common():
            print(f"   {c:4d}  {k}")
    print(f"\nruns usable: {len(runs)}")
    if len(runs) < 6:
        print("too few runs for a screen.")
        return 1

    videos = sorted({r["video"] for r in runs})
    print(f"videos: {len(videos)} ({', '.join(videos)})")

    rows = []
    for held_out in videos:
        train = [r for r in runs if r["video"] != held_out]
        test = [r for r in runs if r["video"] == held_out]
        # Rank-normalize the target within each training run: damage scale
        # varies by run, and the rule only ever uses the ordering.
        Xtr = np.vstack([r["X"] for r in train])
        ytr = np.concatenate([
            (np.argsort(np.argsort(r["damage"])) / max(1, len(r["damage"]) - 1))
            for r in train])
        predict = fit_ridge(Xtr, ytr)

        for r in test:
            dhat = predict(r["X"])
            skill = spearman(dhat, r["damage"])

            # The corrected objective: bits per unit of predicted damage.
            # Shift predicted damage into a strictly positive range first --
            # a ridge prediction of a rank target can go slightly negative,
            # and a ratio through zero is meaningless.
            d_pos = dhat - dhat.min() + 0.05
            corrected = r["bits"] / d_pos
            agreement = spearman(r["bits"], corrected)

            # Overlap of the selected sets at the operating budget.
            k = max(1, int(round(BUDGET * len(r["bits"]))))
            cur = set(np.argsort(-r["bits"])[:k].tolist())
            new = set(np.argsort(-corrected)[:k].tolist())
            overlap = len(cur & new) / k
            rows.append((r["video"], skill, agreement, overlap))

    print(f"\n{'video':16}{'n runs':>8}{'pred skill':>12}{'rank agree':>12}{'set overlap':>13}")
    by_video = collections.defaultdict(list)
    for v, s, a, o in rows:
        by_video[v].append((s, a, o))
    for v in sorted(by_video):
        vals = by_video[v]
        print(f"{v:16}{len(vals):>8}{np.median([x[0] for x in vals]):>+12.3f}"
              f"{np.median([x[1] for x in vals]):>+12.3f}"
              f"{np.median([x[2] for x in vals]):>13.3f}")

    skills = [x[1] for x in rows]
    agrees = [x[2] for x in rows]
    overlaps = [x[3] for x in rows]
    print(f"\n{'OVERALL':16}{len(rows):>8}{np.median(skills):>+12.3f}"
          f"{np.median(agrees):>+12.3f}{np.median(overlaps):>13.3f}")

    print("\n" + "=" * 74)
    med_overlap = float(np.median(overlaps))
    med_skill = float(np.median(skills))
    print(f"held-out damage-predictor skill : rho {med_skill:+.3f} (median over runs)")
    print(f"selected-set overlap at budget  : {med_overlap:.1%}")
    if med_overlap > 0.90:
        print("\nVERDICT: the corrected objective is NOT a materially different rule.")
        print("It reselects the same blocks, so a race against the current rule")
        print("cannot separate them and should not be run. Report this screen.")
    elif med_skill < 0.10:
        print("\nVERDICT: the damage predictor has no held-out skill, so the")
        print("corrected objective has nothing to correct with. Report this screen.")
    else:
        print("\nVERDICT: the corrected objective selects a materially different")
        print("set with a predictor that generalizes across videos. The race is")
        print("worth running.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
