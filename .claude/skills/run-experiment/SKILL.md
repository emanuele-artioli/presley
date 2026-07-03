---
name: run-experiment
description: Add or run PRESLEY experiments via experiments.yaml/presley-run. Use when the user wants to run a video-compression experiment, add a new experiment config, or check what results already exist for a given video/component.
---

# Running PRESLEY experiments

## Field cheat sheet by `component`

All experiments need: `component`, `video`, `width`, `height`.

- **`baselines`** (`src/presley/components/baselines.py`): `codec` (`x264`|`x265`|`kvazaar`|`svtav1`; `hnerv`/`dcvc` are stubs that raise `NotImplementedError` — see reviewer-response skill), `target_bitrate`, `codec_params` (e.g. `{preset: medium}`).
- **`roi`** (`components/roi.py`): `block_size`, `alpha`, `beta`, `roi_method` (`kvazaar`|`x265_aq`|`presley_downsample`|`presley_blur`|`presley_qp`|`presley_noise`; `x264_addroi`/`x265_addroi` are stubs), `target_bitrate`, `codec_params`, and for `presley_*` methods also `degradation_params` (`downsample_scale`, `blur_kernel`, `noise_variance`) and `codec` (default `x265`).
- **`elvis`** (`components/elvis.py`): `block_size`, `alpha`, `beta`, `shrink_amount`, `inpainter` (`propainter`|`e2fgvi`), `target_bitrate`, `codec`, `codec_params`.
- **`presley_ai`** (`components/presley_ai.py`): `block_size`, `alpha`, `beta`, `degradation` (`downsample`|`blur`), `restorer` (`realesrgan` requires `downsample`; `instantir` requires `blur`), `codec` (must be `x265`), `target_bitrate`, `codec_params`, `restorer_params` (`denoise_strength` for realesrgan; `cfg`, `creative_start`, `preview_start` for instantir).

Look at existing entries in `experiments.yaml` for concrete examples before writing a new one.

## Workflow

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
5. Read back `results/<hash>/result.json`. The runner auto-calls
   `presley-evaluate` after a non-dry-run, which appends a `metrics` key
   (`foreground`/`background`/`overall` PSNR/SSIM/MSE, plus LPIPS/DISTS/VMAF/FVMD
   under `overall`, plus block-level `.npz` paths). If `metrics` is missing,
   evaluation didn't run yet — invoke `presley-evaluate results/` manually.

## Gotchas

- The result hash is a `sha256` of the sorted JSON of the experiment dict —
  key order doesn't matter, but changing *any* value (including nested
  `codec_params`) produces a new hash and a fresh run. To force a genuine
  re-run of an unchanged config, delete that specific `results/<hash>/`
  directory — never wipe all of `results/`.
- `--filter` only supports flat top-level key equality (`component=roi`,
  `video=bear`), not nested fields like `codec_params.preset`.
- Don't touch `src/presley/cli.py` / `pipeline_legacy.py` — see root CLAUDE.md.
