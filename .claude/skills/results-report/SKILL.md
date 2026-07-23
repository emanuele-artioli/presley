---
name: results-report
description: Summarize or compare PRESLEY experiment results under results/ (VMAF/LPIPS/DISTS/PSNR/SSIM/FVMD, bitrate, timing) for a video, component, or set of configs. Use when the user wants a table, comparison, or plot of experiment outcomes rather than raw JSON.
---

# Summarizing PRESLEY results

Each `results/<hash>/result.json` has two parts:
- `config`: the original experiment dict (component, video, resolution,
  codec, degradation/restorer params, etc.) — this is what you group/filter by.
- `metrics` (present only after evaluation ran): `foreground`/`background`/
  `overall` blocks with `psnr_mean/std`, `ssim_mean/std`, `mse_mean/std`; plus
  `overall.lpips_mean`, `overall.dists_mean`, `overall.vmaf_mean`,
  `overall.fvmd`. Block-level PSNR/SSIM/MSE arrays are referenced by path to
  sibling `.npz` files (`block_psnr.npz` etc.), not inlined.

## Which FG perceptual metrics are citable — this is not cosmetic

Only **true masked** metrics measure the foreground. A metric computed on a
*bounding box* is not one, however it is named: the union bbox is **100% of the
frame on india** and 58.6% on tennis against a **4.0%** true FG (measured; see
`68e8b6bb11d0dd9e62a67aef/RESEARCH_LOG.md`, Hard rules).

- **Citable as FG:** `foreground.lpips_mean` (spatial-mode map over the true
  per-frame UFO mask) and `foreground.dists_fg` (mask-weighted DISTS pooling;
  paired with `background.dists_bg`). These are region metrics.
- **Not a FG metric:** `foreground.fid_fg_bbox` — a per-frame bbox crop, still
  ~74% background on tennis (see its `fid_fg_bbox_bg_frac_mean`). FID pools
  Inception to one 2048-d vector, so no principled FG-FID exists. Cite it only
  as a corroborating signal, always by its full name, never as "FG-FID".
- **Banned for the FG claim:** `foreground.vmaf_fg_bbox`/`vmaf_neg_fg_bbox`
  (renamed 2026-07-17 from `vmaf_mean`/`vmaf_neg_mean` so the on-disk key
  itself carries the caveat, matching `fid_fg_bbox`) and `foreground.fvmd`
  (union-bbox crops). `foreground.dists_mean`/`foreground.fid` were the same
  defect and have been deleted from `results/`; if you see them, or the old
  `vmaf_mean`/`vmaf_neg_mean` names, in an old copy, do not cite them.
- `overall.*` versions of all of these are legitimate whole-frame metrics.

Also present: `output_video`, `actual_bitrate_bps`, `file_size_bytes`,
`transmitted_size_bytes`, `encoding_time_seconds`, `restoration_time_seconds`,
`total_time_seconds`. **Always compare on `actual_bitrate_bps`, never
`file_size_bytes`** — every component (`elvis.py`, `presley_ai.py`, `roi.py`,
`baselines.py`) already computes `actual_bitrate_bps` from transmitted bytes
(video + any side-channel maps), so it's the uniform, correct comparator
everywhere, including `presley_ai`. `file_size_bytes` is misleading for
elvis/presley_ai specifically: it's the *lossless FFV1 restored output*
(tens of MB), a decode-side artifact unrelated to what was transmitted —
do not use it for a bitrate claim.

Entries with `metrics.fast_only: true` came from a `--fast-metrics` run and
lack LPIPS/DISTS/VMAF/FVMD and block-level maps; run `presley-evaluate
results/` to upgrade them before reporting perceptual metrics.

**Top-level `rate_control`** (`cqp`/`crf`/`vbr_1pass`/`vbr_2pass`/`n/a`) records
the actual encoder rate-control mode used, derived from `config.codec` +
`config.codec_params`/`roi_method` at result-write time (`derive_rate_control`
in `encode_utils.py`) — not a config field, so it doesn't affect
`compute_experiment_hash`. `qp` in `codec_params` means constant-QP for
x265/kvazaar but **CRF** for SVT-AV1 (`rc=0:q=N` with default `aq-mode=2` is
equivalent to `--crf N`); this field disambiguates that instead of leaving it
implicit. Within-codec comparisons are unaffected either way (both sides of a
comparison already share a mode); check this field before stating a
cross-codec rate-control claim. Results written before this field existed were
backfilled once from their `config` (no re-encoding, no metric change).

`presley-compare` (see `src/presley/compare.py`) already respects the FG
citability rules above when picking which key to read per region/metric —
prefer it over hand-rolling a comparison for same-quality/bitrate questions.

## Report foreground-first, against the right target

Follow the "Evaluation methodology" section of CLAUDE.md — the hypothesis
chain is **presley_ai > elvis > roi > baseline** on *foreground* quality at
matched bitrate:

- A table that only shows `overall` metrics buries the result. Lead with
  `foreground` vs `background`, per method.
- Codec ROI methods (`kvazaar`/`x265_aq`/`svtav1`) → compare to the **same
  codec's baseline**; the expected signature is FG↑/BG↓. Its absence is a
  setup bug until proven otherwise.
- `presley_*` ROI methods → compare to the codec ROI methods.
- `elvis` → compare to baselines (same FG↑/BG↓ analysis).
- `presley_ai` → compare to all of the above on `actual_bitrate_bps` (already
  includes the side-channel maps' bytes).

## Workflow

1. Scope the question: which video(s), which component(s), which metric.
2. Walk `results/*/result.json`, filter on `config`, skip entries missing
   `metrics` (not yet evaluated — flag these rather than silently omitting).
3. For a "is quality the same / who's cheaper at matched quality" question,
   use `presley-compare` instead of computing JND deltas by hand — it encodes
   the JND thresholds (VMAF/PSNR/DISTS/SSIM/LPIPS) and the FG-citability rules
   above in code, and its group-scan mode (`presley-compare results/
   --group-by component,video,codec_params.qp --baseline-component baselines`)
   does the matched-QP grouping, verdict, and bitrate-winner lookup in one
   call. For anything else (arbitrary filtering, non-JND summaries, prep for a
   plot), compute directly (pandas/jq) rather than re-deriving plotting code.
   For a reusable chart, prefer extending `plot_grid_search_results.ipynb`
   over writing new one-off scripts.
4. When comparing against a baseline, always match `video`/`width`/`height`
   and compare at similar **actual** bitrates (`actual_bitrate_bps`, not
   `target_bitrate` — rate control over/undershoots, and svtav1-ROI only
   approximates the target via CRF search). Bitrate-mismatched comparisons
   are the most common way to get misleading conclusions here. For paper-grade
   claims across bitrates, use BD-rate curves (multiple target bitrates per
   method); similar-bitrate spot checks are fine for preliminary conclusions.
5. If the user is feeding this into the paper (a table/figure for
   `main.tex`), hand off to the reviewer-response workflow so the relevant
   checklist item gets updated too.
