"""Backfill LPIPS + DISTS for exactly the O2 re-test's runs, in one process.

The CLI's `--only` takes one hash and one `--backfill-*` flag per call, so the
96-run sweep would otherwise cost 192 process starts (and re-import torch each
time); `--backfill-lpips` without `--only` would instead walk the whole 700+ run
corpus while another agent is writing into it. This driver does neither: one
process, one model load, only our hashes.

Usage: python tools/o2_backfill.py <hashes.txt>
"""
import sys

from presley.evaluation.backfill import backfill_dists, backfill_lpips

RESULTS = "/home/itec/emanuele/presley/results"
CACHE = "/home/itec/emanuele/presley/cache"
DATASET = "/home/itec/emanuele/presley/dataset"


def main() -> None:
    hashes = [h.strip() for h in open(sys.argv[1]) if h.strip()]
    for name, fn in (("lpips", backfill_lpips), ("dists", backfill_dists)):
        for i, h in enumerate(hashes, 1):
            print(f"[{name} {i}/{len(hashes)}] {fn(h, RESULTS, CACHE, DATASET)}", flush=True)


if __name__ == "__main__":
    main()
