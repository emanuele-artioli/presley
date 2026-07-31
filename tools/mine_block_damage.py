#!/usr/bin/env python3
"""Mine per-block damage-after-restoration from experiments already on disk.

PRESLEY's block-selection score (alpha/beta over EVCA complexity) only ever
estimated how many BITS a block costs. The other half of the trade -- how well
a block comes back after restoration -- was never in the objective, which is
the structural reason the alpha/beta ablation found nothing. That denominator
was not unmeasurable; it was unqueried: every restored run already stores a
per-block strength map (`strength_maps.npz`, bit-plane packed) alongside
per-block MSE/PSNR/SSIM of the final result (`block_*.npz`).

This script joins each restored run to a matched pristine baseline (same video,
resolution, codec, QP) and emits, per 64x64 superblock:

    strength_frac  fraction of the SB that was degraded  [0, 1]
    mse_deg        mean MSE of the degraded+restored result
    mse_base       mean MSE of the pristine-encode baseline
    psnr_deg/base  the same expressed in dB
    delta_psnr     psnr_base - psnr_deg  == damage in dB attributable to the
                   degrade->restore pipeline, over and above what the codec
                   already did at that QP

MSE is what gets pooled, never PSNR -- PSNR is logarithmic and averaging it
across blocks is meaningless. Pooling is pixel-area weighted so that block
sizes which do not divide 64 (bs=24 appears in real runs) are handled exactly
rather than silently truncated, and partial edge superblocks are kept with
their true area, mirroring how AV1 pads to a superblock and crops on output.

Output is one row per (run, frame, superblock) in a compressed npz, plus a
summary printed to stdout.

Usage:
    python tools/mine_block_damage.py --results-dir results --db results/index.db
    python tools/mine_block_damage.py --restorer realesrgan --out scratch/damage.npz
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from presley.blockdamage import pool_to_superblocks, psnr_from_mse as _psnr  # noqa: E402
from presley.sidechannel import load_level_masks  # noqa: E402

# SB geometry, the PSNR convention and the pooling all live in
# presley.blockdamage, shared with the probe_block_damage component: the two
# must not disagree about where superblock boundaries fall.


def _load(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path) as z:
        return z["arr_0"]


def find_pairs(db: Path, restorer: str | None) -> list[tuple[dict, dict]]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    q = ("select * from results where restoration_time_seconds > 0 "
         "and config_restorer is not null and invariant_failures_n = 0")
    params: tuple = ()
    if restorer:
        q += " and config_restorer = ?"
        params = (restorer,)
    degraded = conn.execute(q, params).fetchall()

    baselines = conn.execute("select * from results where config_component = 'baselines'").fetchall()
    by_key: dict[tuple, list] = {}
    for b in baselines:
        key = (b["config_video"], b["config_width"], b["config_height"],
               b["config_codec"], b["config_codec_params_qp"])
        by_key.setdefault(key, []).append(b)

    pairs = []
    for d in degraded:
        key = (d["config_video"], d["config_width"], d["config_height"],
               d["config_codec"], d["config_codec_params_qp"])
        cands = by_key.get(key)
        if cands:
            pairs.append((dict(d), dict(cands[0])))
    conn.close()
    return pairs


def mine(results_dir: Path, db: Path, out: Path, restorer: str | None) -> int:
    pairs = find_pairs(db, restorer)
    if not pairs:
        print("error: no restored runs with a matched pristine baseline", file=sys.stderr)
        return 1

    cols: dict[str, list] = {k: [] for k in (
        "run", "video", "restorer", "degradation", "codec", "qp", "block_size",
        "frame", "sy", "sx", "strength_frac", "mse_deg", "mse_base",
        "psnr_deg", "psnr_base", "delta_psnr")}
    used = skipped = 0

    for deg, base in pairs:
        dh, bh = deg["experiment_hash"], base["experiment_hash"]
        bs = deg["config_block_size"]
        h, w = deg["config_height"], deg["config_width"]
        if not bs:
            skipped += 1
            continue

        mse_d = _load(results_dir / dh / "block_mse.npz")
        mse_b = _load(results_dir / bh / "block_mse.npz")
        smap_path = results_dir / dh / "strength_maps.npz"
        if mse_d is None or mse_b is None or not smap_path.is_file():
            skipped += 1
            continue
        try:
            strength = load_level_masks(str(smap_path))
        except (OSError, ValueError, KeyError) as e:
            print(f"warning: unreadable strength map for {dh}: {e}", file=sys.stderr)
            skipped += 1
            continue

        # Frame counts can differ by a frame between a run and its baseline;
        # compare only the overlap rather than silently misaligning them.
        nf = min(mse_d.shape[0], mse_b.shape[0], strength.shape[0])
        if nf == 0 or strength.shape[1:] != mse_d.shape[1:]:
            skipped += 1
            continue

        sb_d = pool_to_superblocks(mse_d[:nf], bs, h, w)
        sb_b = pool_to_superblocks(mse_b[:nf], base["config_block_size"] or bs, h, w)
        sb_s = pool_to_superblocks((strength[:nf] > 0).astype(np.float64), bs, h, w)
        if sb_d.shape != sb_b.shape:
            skipped += 1
            continue

        p_d, p_b = _psnr(sb_d), _psnr(sb_b)
        f_idx, y_idx, x_idx = np.meshgrid(
            np.arange(nf), np.arange(sb_d.shape[1]), np.arange(sb_d.shape[2]), indexing="ij")
        n = sb_d.size

        cols["run"].extend([dh] * n)
        cols["video"].extend([deg["config_video"]] * n)
        cols["restorer"].extend([deg["config_restorer"]] * n)
        cols["degradation"].extend([deg["config_degradation"] or ""] * n)
        cols["codec"].extend([deg["config_codec"] or ""] * n)
        cols["qp"].extend([deg["config_codec_params_qp"] or -1] * n)
        cols["block_size"].extend([bs] * n)
        cols["frame"].append(f_idx.ravel())
        cols["sy"].append(y_idx.ravel())
        cols["sx"].append(x_idx.ravel())
        cols["strength_frac"].append(sb_s.ravel())
        cols["mse_deg"].append(sb_d.ravel())
        cols["mse_base"].append(sb_b.ravel())
        cols["psnr_deg"].append(p_d.ravel())
        cols["psnr_base"].append(p_b.ravel())
        cols["delta_psnr"].append((p_b - p_d).ravel())
        used += 1

    if used == 0:
        print("error: no usable pairs (all skipped)", file=sys.stderr)
        return 1

    packed = {}
    for k, v in cols.items():
        packed[k] = np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **packed)

    print(f"mined {used} run/baseline pairs ({skipped} skipped) -> {out}")
    print(f"rows: {len(packed['delta_psnr']):,} superblock observations")
    _summarize(packed)
    return 0


def _summarize(d: dict) -> None:
    delta, frac, restorer = d["delta_psnr"], d["strength_frac"], d["restorer"]
    degraded = frac > 0.5
    clean = frac == 0.0

    print("\n-- damage attributable to degrade->restore, in dB (higher = worse) --")
    print(f"{'restorer':>16} {'n_deg':>8} {'delta(deg)':>11} {'delta(clean)':>13} {'spread':>8}")
    for r in sorted(set(restorer.tolist())):
        m = restorer == r
        dd, dc = delta[m & degraded], delta[m & clean]
        if dd.size == 0:
            continue
        print(f"{r:>16} {dd.size:>8,} {np.median(dd):>11.2f} "
              f"{(np.median(dc) if dc.size else float('nan')):>13.2f} "
              f"{np.percentile(dd, 90) - np.percentile(dd, 10):>8.2f}")

    print("\nThe 'spread' column is the p90-p10 range of damage across superblocks")
    print("within one restorer: the headroom a selection rule could exploit.")
    print("'delta(clean)' is damage on NON-degraded superblocks -- non-zero there")
    print("means neighbour bleed through inter prediction, not selection error.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--db", default=None, help="index db (default: <results-dir>/index.db)")
    ap.add_argument("--out", default=None, help="output npz (default: <results-dir>/block_damage.npz)")
    ap.add_argument("--restorer", default=None, help="restrict to one restorer")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    db = Path(args.db) if args.db else results_dir / "index.db"
    out = Path(args.out) if args.out else results_dir / "block_damage.npz"
    if not db.is_file():
        print(f"error: no index at {db} -- run tools/index_results.py first", file=sys.stderr)
        return 1
    return mine(results_dir, db, out, args.restorer)


if __name__ == "__main__":
    sys.exit(main())
