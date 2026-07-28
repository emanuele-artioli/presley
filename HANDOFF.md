# Handoff — NAFNet Q5 done (fp32); Real-HAT-GAN next (2026-07-28)

## What triggered this handoff

Session boundary after Wave 1 closed: NAFNet wired, fp16 crater diagnosed and
fixed, Q5 re-measured, paper retracted (`83ea603`). Next session = **Wave 2
Real-HAT-GAN (Q4)**. Paper Overleaf still **not** pushed unless the human asks.

## One-paragraph summary

PRESLEY degrades BG under fixed QP and restores client-side. InstantIR kill
stands (as-run and corrected). NAFNet GoPro-width64 is the CNN blur gauge:
must run **fp32 + Local/TLSC**; fp16 overflows LayerNorm2d/SCA (rainbow
artifacts — that was the false “12 dB crater”). Corrected NAFNet **ties
unsharp within JND**, beats InstantIR, ~zero Goal-2 gain vs transmitted.
Blur transport is not retired. Queue: [`docs/EXPERIMENTS_QUEUED.md`](docs/EXPERIMENTS_QUEUED.md).

## Current state (verified 2026-07-28)

### Code repo `presley/`

- Branch: `main` (pushed with this handoff’s commit).
- NAFNet: `src/presley/nafnet_arch.py`, restore/dispatch, tests, Q5 yaml.
- **Hard rule:** `get_nafnet` / deblur **reject** `fp32=False`. Default
  `local=True`. Do not re-introduce half precision “for speed.”
- Weights on disk: `weights/NAFNet-GoPro-width64.pth`, `weights/BSRGAN.pth`.
  **Missing:** Real-HAT-GAN `.pth`.

### Paper repo `68e8b6bb11d0dd9e62a67aef/`

- Local commits (ahead of Overleaf, **not pushed**):
  - `c67b3f6` — `tab:instantir-kill`
  - `78e7acb` — false fp16 NAFNet rows (superseded)
  - `83ea603` — retraction + corrected fp32+Local numbers
- InstantIR kill stands; NAFNet is a near-tie with unsharp CNN gauge.

### Q5 results (citable, CRF, `invariant_failures=[]`)

| Video | Hash | BG-LPIPS | vs unsharp | vs InstantIR-corr |
|---|---|---|---|---|
| bear | `93fcdf516cf7e363` | 0.356 | tie ≤0.08×JND | wins ~3.6×JND |
| camel | `8d2316a5e2d6128f` | 0.308 | tie | wins ~4.7×JND |

Also useful: InstantIR corrected `36fb007975de0593` / `afd6dc1e5fa2ebe1`;
unsharp `e26598d90f189991` / `6857ccfe20a8c701`; Real-ESRGAN conditioned
`e2cb6bed165d69b1` / `c29c94b5c290f208`.

### Running now

Nothing PRESLEY-owned. `nvidia-smi`; `ps -ef | rg 'presley-run' | rg -v rg`.

## Implementation plan — Wave 2 Real-HAT-GAN (Q4)

1. **Download weights** (no pip into `presley` env; unset proxy if HF 403):
   ```bash
   env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
     conda run -n presley python -c \
     "from huggingface_hub import hf_hub_download; print(hf_hub_download(
       'Acly/hat','Real_HAT_GAN_sharper.pth', local_dir='weights'))"
   ```
   (Confirm exact filename on the Hub if the name differs —
   `Real_HAT_GAN_SRx4_sharper` variants exist.)
2. **Integrate** `restorer: real_hat_gan` on **downsample** (mirror BSRGAN /
   Real-ESRGAN path). Prefer a small vendor or existing package that fits
   **torch 2.1.2** without upgrading diffusers.
3. **fp16 lesson:** before citing any cell, smoke one frame against a known
   author/reference path (or fp32 vs fp16 MAE). Ban half if unstable.
4. **Register** in `presley_ai.py` + dispatch/contract tests (BSRGAN pattern).
5. **Q4 experiments** (fixed-QP only, `codec=svtav1`): same recipe as
   `tab:conditioned` Real-ESRGAN cells; compare to
   `e2cb6bed165d69b1` / `c29c94b5c290f208`.
6. Dry-run → run bear then camel → `presley-compare` → `/update-paper`
   (no Overleaf unless asked).

### Deferred

- Q6 DiffBIR — not authorized by NAFNet result (ties unsharp; no new mechanism).
- Q7/Q8 Stream-DiffVSR / DC-VSR; Q9 noise; Q10 mask dilate.

## Ops gotchas

- Filter fixed-QP with **`codec=svtav1`**. Bare `restorer=` rematches VBR dirs.
- `fg_protect=True` (capital T).
- BG-PSNR never the Goal-2 verdict; use BG-LPIPS + `presley-compare`.
- NAFNet **fp32 only**. HF download may need proxy env cleared + full perms.

## Immediate next steps

1. Download Real-HAT-GAN weights.
2. Implement Wave 2 (arch → restore → register → tests → dry-run Q4).
3. Run Q4 bear/camel; compare to Real-ESRGAN; fold into paper.
4. Overleaf push of paper `c67b3f6`…`83ea603` — only if human asks.

## Prompt for the next session

Copy-paste:

```
Pick up HANDOFF.md in /home/itec/emanuele/presley.

Wave 1 (NAFNet Q5) is done and pushed in the code repo. InstantIR kill stands;
corrected NAFNet (fp32+Local) ties unsharp within JND and beats InstantIR —
do not re-litigate the false fp16 crater (paper 83ea603 already retracted it).
Paper Overleaf is still local-only (commits c67b3f6 → 78e7acb → 83ea603) —
do not push Overleaf unless I ask.

Your job is Wave 2: Real-HAT-GAN (Q4).
1. Download Real-HAT-GAN weights into weights/ (HF Acly/hat; unset proxy if needed).
2. Integrate restorer: real_hat_gan on downsample (mirror BSRGAN/Real-ESRGAN).
   Fit torch 2.1.2; do not upgrade diffusers. Validate fp16 vs fp32 before
   citing — NAFNet taught us half can silently destroy a run.
3. Register + tests; add fixed-QP Q4 yaml cells (codec=svtav1); dry-run then
   run bear then camel vs Real-ESRGAN hashes e2cb6bed165d69b1 / c29c94b5c290f208.
4. presley-compare; fold into the paper via /update-paper (no Overleaf push).

Filter tip: always codec=svtav1 (+ restorer/video). Read docs/EXPERIMENTS_QUEUED.md
and HANDOFF.md before running GPU jobs.
```

## Landmarks

| Path | Why |
|---|---|
| `docs/EXPERIMENTS_QUEUED.md` | Queue + restorer matrix |
| `HANDOFF.md` | This file + next-session prompt |
| `src/presley/nafnet_arch.py` / `restoration.py` | NAFNet (done; fp32+Local) |
| `src/presley/components/presley_ai.py` | Restorer dispatch |
| `tests/test_restoration_bsrgan.py` | Pattern for Real-HAT contract tests |
| `68e8b6bb11d0dd9e62a67aef/sections/evaluation.tex` | `tab:instantir-kill` |
| `.claude/skills/run-experiment/SKILL.md` | Add/run yaml |
| `.claude/skills/update-paper/SKILL.md` | Fold Q4 into paper |
