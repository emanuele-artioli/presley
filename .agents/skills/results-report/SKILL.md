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

Also present: `output_video`, `actual_bitrate_bps`, `file_size_bytes`,
`transmitted_size_bytes` (video + side-channel metadata, relevant for
`presley_ai` where strength maps are a separate cost), `encoding_time_seconds`,
`restoration_time_seconds`, `total_time_seconds`.

Entries with `metrics.fast_only: true` came from a `--fast-metrics` run and
lack LPIPS/DISTS/VMAF/FVMD and block-level maps; run `presley-evaluate
results/` to upgrade them before reporting perceptual metrics.

## Report foreground-first, against the right target

Follow the "Evaluation methodology" section of AGENTS.md — the hypothesis
chain is **presley_ai > elvis > roi > baseline** on *foreground* quality at
matched bitrate:

- A table that only shows `overall` metrics buries the result. Lead with
  `foreground` vs `background`, per method.
- Codec ROI methods (`kvazaar`/`x265_aq`/`svtav1`) → compare to the **same
  codec's baseline**; the expected signature is FG↑/BG↓. Its absence is a
  setup bug until proven otherwise.
- `presley_*` ROI methods → compare to the codec ROI methods.
- `elvis` → compare to baselines (same FG↑/BG↓ analysis).
- `presley_ai` → compare to all of the above, using `transmitted_size_bytes`
  for its bitrate (side-channel maps count).

## Workflow

1. Scope the question: which video(s), which component(s), which metric.
2. Walk `results/*/result.json`, filter on `config`, skip entries missing
   `metrics` (not yet evaluated — flag these rather than silently omitting).
3. For a quick answer, compute directly (pandas/jq) rather than re-deriving
   plotting code. For a reusable chart, prefer extending
   `plot_grid_search_results.ipynb` over writing new one-off scripts.
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
