#!/usr/bin/env python3
"""Backfill LPIPS + DISTS for a list of experiment hashes.

Exists because the `presley-evaluate` CLI takes exactly ONE `--backfill-*` flag
per invocation, so two metrics would otherwise mean two passes over the whole
results tree. Runs are PSNR/SSIM-only when they come off the runner, and the
perceptual metrics are the Goal-2 verdict, so skipping this leaves the verdict
missing rather than wrong -- which is worse.

chdir's to the repo root before calling in: result.json stores `output_video`
as a path relative to the repo root for every run made before the S1b batch, so
a caller sitting in a git worktree silently gets "output video missing" on all
of them.

Usage:
    python tools/s1b_backfill.py --repo-root /path/to/presley --hashes a,b,c
    python tools/s1b_backfill.py --repo-root /path/to/presley --hash-file hashes.txt
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--hashes", default="", help="comma-separated experiment hashes")
    ap.add_argument("--hash-file", default=None, help="file with one hash per line")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    hashes = [h.strip() for h in args.hashes.split(",") if h.strip()]
    if args.hash_file:
        with open(args.hash_file) as fh:
            hashes += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if not hashes:
        print("error: no hashes given", file=sys.stderr)
        return 1

    root = os.path.abspath(args.repo_root)
    os.chdir(root)
    sys.path.insert(0, os.path.join(root, "src"))
    from presley.evaluation.backfill import backfill_dists, backfill_lpips  # noqa: E402

    results_dir = os.path.join(root, "results")
    cache_dir = os.path.join(root, "cache")
    dataset_dir = os.path.join(root, "dataset")

    failures = 0
    for h in hashes:
        for fn in (backfill_lpips, backfill_dists):
            msg = fn(h, results_dir, cache_dir, dataset_dir, force=args.force)
            print(msg, flush=True)
            if "missing" in msg or "no result.json" in msg or "no metrics" in msg:
                failures += 1
    print(f"\n{len(hashes)} hashes, {failures} problem line(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
