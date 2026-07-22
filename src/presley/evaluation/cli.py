"""Argparse entry point for `presley-evaluate`."""

from typing import Dict, Any, List
from presley.preprocessing import get_reference_frames, get_ufo_masks
_REF_CACHE: Dict[Any, Any] = {}
_MASK_CACHE: Dict[Any, Any] = {}
_DISTS_CACHE: Dict[str, Any] = {}
from presley.evaluation.run import evaluate_all
from presley.evaluation.backfill import backfill_dists, backfill_dists_all, backfill_fid, backfill_fid_all, backfill_fvmd, backfill_fvmd_all, backfill_lpips, backfill_lpips_all, backfill_transmitted_perceptual, backfill_transmitted_perceptual_all, backfill_vmaf, backfill_vmaf_all, drop_unionbbox_keys, drop_unionbbox_keys_all
from presley.evaluation.reports import fid_validity_report, fvmd_setlevel_report, fvmd_validity_report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str, default='results')
    parser.add_argument('--dataset-dir', type=str, default='dataset')
    parser.add_argument('--cache-dir', type=str, default='cache')
    parser.add_argument('--fast-metrics', action='store_true',
                        help='Only compute fast metrics (FG/BG/overall PSNR/SSIM/MSE); skip LPIPS/DISTS/VMAF/FVMD and block-level maps')
    parser.add_argument('--backfill-lpips', action='store_true',
                        help='Append FG/BG/overall masked LPIPS to existing result.json files without re-encoding or recomputing other metrics')
    parser.add_argument('--backfill-vmaf', action='store_true',
                        help='Append overall + FG-crop VMAF (default and NEG models) to existing result.json files without re-encoding')
    parser.add_argument('--backfill-dists', action='store_true',
                        help='Append overall DISTS + true mask-weighted FG/BG DISTS (dists_fg/dists_bg) to existing result.json files')
    parser.add_argument('--backfill-transmitted', action='store_true',
                        help='Append masked LPIPS+DISTS for the transmitted (decoded degraded) video to '
                             'metrics.transmitted -- makes the restoration perceptual gain measurable')
    parser.add_argument('--backfill-fid', action='store_true',
                        help='Append overall FID + per-frame tight-bbox fid_fg_bbox (best-effort locality, NOT a foreground metric) to existing result.json files')
    parser.add_argument('--backfill-fvmd', action='store_true',
                        help='Append overall + FG-crop per-video FVMD to existing result.json files (internal signal; not the paper metric)')
    parser.add_argument('--fvmd-setlevel', type=str, default=None, metavar='GROUPS_JSON',
                        help='Compute paper-grade set-level FVMD: JSON file mapping set-name -> list of experiment hashes')
    parser.add_argument('--fvmd-out', type=str, default='fvmd_setlevel.tsv',
                        help='Output table path for --fvmd-setlevel/--fvmd-validity')
    parser.add_argument('--fvmd-validity', type=str, default=None, metavar='GROUPS_JSON',
                        help='Set-level FVMD plus its validity gate (identity check, leave-one-video-out '
                             'jackknife = the real uncertainty at n=6, mean/cov term split) — run this '
                             'before citing any set-level score')

    parser.add_argument('--fid-validity', type=str, default=None, metavar='EXPERIMENT_HASH',
                        help='FID small-sample validity gate on one experiment (identity check, paired-noise '
                             'monotonicity, unpaired-split diagnostic) at N~60-90 vs D=2048 — run this before '
                             'citing any per-experiment FID')
    parser.add_argument('--fid-out', type=str, default='scratch/fid_validity.tsv',
                        help='Output table path for --fid-validity')
    parser.add_argument('--drop-unionbbox-keys', action='store_true',
                        help='One-shot: delete the superseded union-bbox foreground.dists_mean/foreground.fid '
                             'keys (not foreground metrics; see TECHNICAL_REPORT_PIPELINE_INFRA.md 2026-07-16). '
                             'No recomputation; idempotent')

    parser.add_argument('--force', action='store_true',
                        help='With a --backfill-* flag, recompute even if the metric is already present')
    parser.add_argument('--only', type=str, default=None, metavar='HASH',
                        help='With a --backfill-* / --drop-unionbbox-keys flag, act on a single experiment '
                             '(for verification runs)')
    parser.add_argument('--shard', type=str, default=None,
                        help='Shard the evaluation (e.g. 0/2)')
    args = parser.parse_args()
    if args.only:
        # Single-experiment verification path: run the one hash through the same
        # per-experiment function the *_all drivers call.
        fns = {
            'backfill_lpips': backfill_lpips, 'backfill_vmaf': backfill_vmaf,
            'backfill_dists': backfill_dists, 'backfill_fid': backfill_fid,
            'backfill_fvmd': backfill_fvmd,
            'backfill_transmitted': backfill_transmitted_perceptual,
        }
        selected = [f for f in fns if getattr(args, f)]
        if args.drop_unionbbox_keys:
            selected.append('drop_unionbbox_keys')
        if not selected:
            parser.error('--only requires a --backfill-* or --drop-unionbbox-keys flag')
        # Fail loudly rather than silently honouring only the first: the *_all path is an
        # elif chain, so passing two flags there already drops one, and --only is a
        # verification flag where a silently skipped metric is exactly the wrong outcome.
        if len(selected) > 1:
            parser.error(f'--only takes a single action, got: {", ".join(selected)}')
        if selected[0] == 'drop_unionbbox_keys':
            print(drop_unionbbox_keys(args.only, args.results_dir))
        else:
            print(fns[selected[0]](args.only, args.results_dir, args.cache_dir,
                                   args.dataset_dir, force=args.force))
        return
    if args.fid_validity:
        fid_validity_report(args.fid_validity, args.results_dir, args.cache_dir,
                            args.dataset_dir, args.fid_out)
    elif args.drop_unionbbox_keys:
        drop_unionbbox_keys_all(args.results_dir)
    elif args.backfill_lpips:
        backfill_lpips_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_vmaf:
        backfill_vmaf_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_dists:
        backfill_dists_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_transmitted:
        backfill_transmitted_perceptual_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_fid:
        backfill_fid_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force)
    elif args.backfill_fvmd:
        backfill_fvmd_all(args.results_dir, args.cache_dir, args.dataset_dir, force=args.force, shard=args.shard)
    elif args.fvmd_validity:
        fvmd_validity_report(args.fvmd_validity, args.results_dir, args.cache_dir, args.dataset_dir, args.fvmd_out)
    elif args.fvmd_setlevel:
        fvmd_setlevel_report(args.fvmd_setlevel, args.results_dir, args.cache_dir, args.dataset_dir, args.fvmd_out)
    else:
        evaluate_all(args.results_dir, args.cache_dir, args.dataset_dir, fast=args.fast_metrics)


if __name__ == "__main__":
    main()
