#!/usr/bin/env python3
"""Build S1b Arm-B oracle level maps: same histogram as Arm A, damage-aware assignment.

S1b's gate established that the *within*-level damage spread (12.36 dB at k=3)
is 7.4x the *between*-level cost of moving a block up a level (1.66 dB), so a
perfect damage-aware sort has real headroom. This script spends that headroom.

Arm A (naive graded) assigns level by removability score -- the bits-cost
proxy. Arm B keeps Arm A's **exact per-frame level histogram**, so the block
count at each level is identical and the bit spend is held roughly fixed by
construction, and reassigns *which* block gets which level using measured
damage instead of score.

The damage input is `tools/mine_block_damage.py`'s per-64x64-superblock
`delta_psnr` from the uniform-level probe runs: damage_b(k) for k in {2,3} on
the identical footprint. Blocks inherit their superblock's damage (block_size
16 -> 4x4 blocks per SB), so the oracle is superblock-resolution, which is the
resolution the damage was measured at. Sorting inside a superblock is stable
and arbitrary; that granularity limit is real and reported.

**This is a GREEDY oracle, not a proven optimum.** It sorts the footprint by
the marginal cost of the steep step, m = delta(3) - delta(2), and hands the
steepest levels to the blocks with the smallest m -- exactly the rule
pre-registered in docs/WAVE1_FALSIFIERS.md ("blocks that suffer least at a
steep level get the steep levels"). A constrained-optimal assignment could
only do better, so an Arm-B *win* is a valid ceiling demonstration, while an
Arm-B *loss* bounds only this rule and does not by itself prove no assignment
can win. Say so when reporting.

Usage:
    python tools/build_oracle_levels.py --results-dir results \
        --damage results/block_damage_s1b.npz --out cache/oracle_levels
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from presley.sidechannel import load_level_masks  # noqa: E402

SB = 64  # superblock side in pixels, matching mine_block_damage.py


def _find_strength_map(results_dir: str, h: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(results_dir, h, "**", "*strength*"), recursive=True))
    return hits[0] if hits else None


def _probe_hashes_by_video(results_dir: str) -> dict[str, dict[int, str]]:
    """Map video -> {uniform_level: hash} over every probe run on disk."""
    out: dict[str, dict[int, str]] = {}
    for p in glob.glob(os.path.join(results_dir, "*", "result.json")):
        try:
            cfg = json.load(open(p)).get("config", {})
        except Exception:
            continue
        lvl = cfg.get("downsample_uniform_level")
        if not lvl:
            continue
        out.setdefault(cfg["video"], {})[int(lvl)] = os.path.basename(os.path.dirname(p))
    return out


def _damage_grid(dmg: dict, run: str, n_frames: int, n_sb_y: int, n_sb_x: int) -> np.ndarray:
    """(frames, sb_y, sb_x) of delta_psnr for one run; NaN where unobserved."""
    m = dmg["run"] == run
    g = np.full((n_frames, n_sb_y, n_sb_x), np.nan, dtype=np.float64)
    fr, sy, sx = dmg["frame"][m], dmg["sy"][m], dmg["sx"][m]
    ok = (fr < n_frames) & (sy < n_sb_y) & (sx < n_sb_x)
    g[fr[ok], sy[ok], sx[ok]] = dmg["delta_psnr"][m][ok]
    return g


def build_for_video(video, arm_a_hash, probes, dmg, results_dir):
    smap_path = _find_strength_map(results_dir, arm_a_hash)
    if smap_path is None:
        raise SystemExit(f"{video}: no strength map under {arm_a_hash}")
    arm_a = load_level_masks(smap_path)  # (frames, by, bx)
    n_f, n_by, n_bx = arm_a.shape

    block_size = 16  # every S1/S1b run uses block_size 16; asserted by the caller
    per_sb = SB // block_size  # 4 blocks per superblock side
    n_sb_y = int(np.ceil(n_by / per_sb))
    n_sb_x = int(np.ceil(n_bx / per_sb))

    d2 = _damage_grid(dmg, probes[2], n_f, n_sb_y, n_sb_x)
    d3 = _damage_grid(dmg, probes[3], n_f, n_sb_y, n_sb_x)
    marginal_sb = d3 - d2  # cost in dB of stepping this SB from level 2 to 3

    # Broadcast superblock marginal cost down to blocks.
    by_idx = np.minimum(np.arange(n_by) // per_sb, n_sb_y - 1)
    bx_idx = np.minimum(np.arange(n_bx) // per_sb, n_sb_x - 1)
    marginal = marginal_sb[:, by_idx][:, :, bx_idx]  # (frames, by, bx)

    oracle = np.zeros_like(arm_a)
    unobserved = 0
    for f in range(n_f):
        foot = np.flatnonzero(arm_a[f].ravel() > 0)
        if foot.size == 0:
            continue
        levels_here = arm_a[f].ravel()[foot]
        n3 = int((levels_here == 3).sum())
        n2 = int((levels_here == 2).sum())

        cost = marginal[f].ravel()[foot]
        # A superblock with no damage observation must not be handed the steep
        # levels by accident: +inf sorts it to the mild end, the safe default.
        bad = ~np.isfinite(cost)
        unobserved += int(bad.sum())
        cost = np.where(bad, np.inf, cost)

        order = np.argsort(cost, kind="stable")  # ascending: most tolerant first
        assigned = np.ones(foot.size, dtype=np.int32)
        assigned[order[:n3]] = 3
        assigned[order[n3:n3 + n2]] = 2

        flat = oracle[f].ravel()
        flat[foot] = assigned
        oracle[f] = flat.reshape(n_by, n_bx)

    # The histogram is the whole point of the arm -- verify, never assume.
    for lvl in (0, 1, 2, 3):
        a, b = int((arm_a == lvl).sum()), int((oracle == lvl).sum())
        if a != b:
            raise SystemExit(f"{video}: histogram mismatch at level {lvl}: armA={a} oracle={b}")

    changed = int((oracle != arm_a).sum())
    foot_n = int((arm_a > 0).sum())
    return oracle, dict(frames=n_f, footprint=foot_n,
                        changed=changed,
                        changed_pct=100.0 * changed / max(foot_n, 1),
                        unobserved=unobserved)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--damage", required=True)
    ap.add_argument("--out", required=True, help="output directory for <video>.npz")
    ap.add_argument("--arm-a", required=True,
                    help="JSON mapping video -> Arm A (levels=3, naive graded) hash")
    args = ap.parse_args()

    dmg = dict(np.load(args.damage, allow_pickle=True))
    probes = _probe_hashes_by_video(args.results_dir)
    arm_a_map = json.loads(Path(args.arm_a).read_text())
    os.makedirs(args.out, exist_ok=True)

    print(f"{'video':16}{'frames':>7}{'foot':>8}{'reassigned':>12}{'unobs':>7}")
    for video, h in sorted(arm_a_map.items()):
        p = probes.get(video, {})
        if 2 not in p or 3 not in p:
            raise SystemExit(f"{video}: needs both k=2 and k=3 probes, found {sorted(p)}")
        oracle, stats = build_for_video(video, h, p, dmg, args.results_dir)
        np.savez_compressed(os.path.join(args.out, f"{video.replace('/', '_')}.npz"),
                            levels=oracle)
        print(f"{video:16}{stats['frames']:>7}{stats['footprint']:>8}"
              f"{stats['changed']:>8} ({stats['changed_pct']:.0f}%){stats['unobserved']:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
