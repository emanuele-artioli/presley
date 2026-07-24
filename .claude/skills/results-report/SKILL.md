---
name: results-report
description: Summarize or compare PRESLEY experiment results under results/ (VMAF/LPIPS/DISTS/PSNR/SSIM/FVMD, bitrate, timing) for a video, component, or set of configs. Use when the user wants a table, comparison, or plot of experiment outcomes rather than raw JSON.
---

Read `/home/itec/emanuele/.agent-rules/skills/results-report/SKILL.md` and follow it.

## PRESLEY specifics

- **Result schema:** `results/<hash>/result.json` has `config` (what to
  group/filter by) and `metrics` (`foreground`/`background`/`overall`, each
  with `psnr_mean/std`, `ssim_mean/std`, `mse_mean/std`, plus
  `lpips_mean`/`dists_mean`/`vmaf_mean`/`fvmd` on `overall`). Block-level
  arrays live in sibling `.npz` files, referenced by path.
- **Citability:** a non-empty `invariant_failures` list makes a result
  uncitable (unsound run, not just odd numbers) — exclude and say which
  hashes were dropped and why. No `invariant_failures` key at all means
  unevaluated, not clean — backfill with `python -m presley.invariants
  results/` first.
- **FG-citable metrics only:** `foreground.lpips_mean` (spatial UFO-mask
  LPIPS) and `foreground.dists_fg` are true region metrics. NOT a FG metric:
  `foreground.fid_fg_bbox` (bbox crop — 100% of frame on india, 58.6% on
  tennis, vs a measured 4.0% true FG) — cite only as a corroborating signal,
  by its full name. **Banned:** `foreground.vmaf_fg_bbox`/`vmaf_neg_fg_bbox`,
  `foreground.fvmd` (all bbox-based). `overall.*` versions are legitimate
  whole-frame metrics.
- **Bitrate comparator:** always `actual_bitrate_bps`, never
  `file_size_bytes` (which is the lossless FFV1 decode-side artifact for
  elvis/presley_ai, unrelated to what was transmitted).
- `metrics.fast_only: true` means the entry lacks LPIPS/DISTS/VMAF/FVMD —
  run `presley-evaluate results/` to upgrade before reporting perceptual
  metrics.
- `rate_control` (`cqp`/`crf`/`vbr_1pass`/`vbr_2pass`/`n/a`) records the
  actual encoder mode; check it before any cross-codec rate-control claim
  (`qp` in `codec_params` means constant-QP for x265/kvazaar but CRF for
  SVT-AV1).
- **Expected chain:** `presley_ai > elvis > roi > baseline` on foreground
  quality at matched bitrate. Lead tables with `foreground` vs `background`,
  never `overall`-only. Use `presley-compare` (`src/presley/compare.py`,
  respects all rules above) for JND-gated comparisons and its group-scan mode
  (`--group-by component,video,codec_params.qp --baseline-component
  baselines`) for matched-QP sweeps, instead of hand-rolling deltas.
