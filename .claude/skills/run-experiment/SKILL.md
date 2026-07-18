---
name: run-experiment
description: Add or run PRESLEY experiments via experiments.yaml/presley-run. Use when the user wants to run a video-compression experiment, add a new experiment config, or check what results already exist for a given video/component.
---

# Running PRESLEY experiments

## Field cheat sheet by `component`

All experiments need: `component`, `video`, `width`, `height`.

- **`baselines`** (`src/presley/components/baselines.py`): `codec` (`x264`|`x265`|`kvazaar`|`svtav1`; `hnerv`/`dcvc` are stubs that raise `NotImplementedError` — see reviewer-response skill), `target_bitrate`, `codec_params` (e.g. `{preset: medium}`).
- **`roi`** (`components/roi.py`): `block_size`, `alpha`, `beta`, `roi_method` (`kvazaar`|`x265_aq`|`svtav1`|`presley_downsample`|`presley_blur`|`presley_qp`|`presley_noise`; `x264_addroi`/`x265_addroi` are stubs), `target_bitrate`, `codec_params`, and for `presley_*` methods also `degradation_params` (`downsample_scale`, `blur_kernel`, `noise_variance`) and `codec` (default `x265`). `kvazaar` and `svtav1` ROI both encode in fixed-QP/CRF mode with a binary search toward `target_bitrate` (bitrate-targeted rate control absorbs/ignores ROI deltas — see the fixed-QP hard rule in `68e8b6bb11d0dd9e62a67aef/RESEARCH_LOG.md`), so their `actual_bitrate_bps` is approximate — compare on actuals.
- **`elvis`** (`components/elvis.py`): `block_size`, `alpha`, `beta`, `shrink_amount`, `inpainter` (`propainter`|`e2fgvi`), `target_bitrate`, `codec`, `codec_params`.
- **`presley_ai`** (`components/presley_ai.py`): `block_size`, `alpha`, `beta`, `degradation` (`downsample`|`blur`), `restorer` (`realesrgan` requires `downsample`; `instantir` requires `blur`), `codec` (must be `x265`), `target_bitrate`, `codec_params`, `restorer_params` (`denoise_strength` for realesrgan; `cfg`, `creative_start`, `preview_start` for instantir).

This cheat sheet lags the code (e.g. `elvis` also takes `removal_mode:
blackout|freeze|shrink`, `fg_protect`, `composite_output`, `inpainter: telea`,
and `codec: svtav1`; `presley_ai` also takes `degradation: mean_fill|freeze`,
`restorer: propainter|none`, `codec: svtav1`, `shrink_amount`, `fg_protect`;
fixed-QP mode is `codec_params: {qp: N}` with no `target_bitrate`). The
components in `src/presley/components/*.py` are authoritative — check there,
and look at existing `experiments.yaml` entries (with their dated section
comments) for concrete examples before writing a new one.

## Workflow

0. **Check what the paper needs first.** Grep the paper's markers
   (`cd 68e8b6bb11d0dd9e62a67aef && grep -n '^% *\(GOAL\|HOLE\)(' main.tex
   sections/*.tex`) — if a `HOLE()` names this experiment, copy its exact
   config demands and note the anchor id so `/update-paper` can close it
   afterwards. Run only experiments the paper (or an explicit user request)
   needs.
1. Confirm the video is available: `ls dataset/` — if the target video isn't
   symlinked yet, ask the user for the DAVIS path rather than guessing one.
2. Add the experiment dict to `experiments.yaml`.
3. **Dry-run first**, scoped narrowly with `--filter`:
   ```
   presley-run experiments.yaml --filter video=<name> --dry-run
   ```
   Check the printed config looks right before running for real.
4. Run for real. These are GPU jobs that can take minutes (baselines/roi) to
   hours (elvis in-painting, presley_ai restoration) — prefer
   `run_in_background` for anything beyond a `baselines`/`roi` smoke test.
   For fast iteration add `--fast-metrics` (skips LPIPS/DISTS/VMAF/FVMD and
   block-level maps; keeps FG/BG/overall PSNR/SSIM/MSE, which is what the
   comparison methodology in CLAUDE.md needs). A later plain
   `presley-evaluate results/` upgrades fast-only results to full metrics.
5. Read back `results/<hash>/result.json`. The runner auto-calls
   `presley-evaluate` after a non-dry-run, which appends a `metrics` key
   (`foreground`/`background`/`overall` PSNR/SSIM/MSE, plus LPIPS/DISTS/VMAF/FVMD
   under `overall`, plus block-level `.npz` paths). If `metrics` is missing,
   evaluation didn't run yet — invoke `presley-evaluate results/` manually.
6. Analyze against the experiment's comparison target per the
   "Evaluation methodology" section in CLAUDE.md (foreground-first, matched
   actual bitrates) — an experiment without that comparison is not a result.

## Gotchas

- The result hash is a `sha256` of the sorted JSON of the experiment dict —
  key order doesn't matter, but changing *any* value (including nested
  `codec_params`) produces a new hash and a fresh run. To force a genuine
  re-run of an unchanged config, delete that specific `results/<hash>/`
  directory — never wipe all of `results/`.
- `--filter` only supports flat top-level key equality (`component=roi`,
  `video=bear`), not nested fields like `codec_params.preset`.
- Don't touch `src/presley/cli.py` / `pipeline_legacy.py` — see root CLAUDE.md.
