# Handoff — Q4 Real-HAT-GAN done; Q1–Q5 closed (2026-07-28)

## What triggered this handoff

Wave 2 Real-HAT-GAN (Q4) closed: restorer wired (fp32-only), fixed-QP bear/camel
ran, `presley-compare` vs Real-ESRGAN, paper folded locally (`0c30ad7`). Paper
Overleaf still **not** pushed unless the human asks. Code-repo Real-HAT changes
are uncommitted in this checkout — commit/push only if asked.

## One-paragraph summary

PRESLEY degrades BG under fixed QP and restores client-side. InstantIR kill
stands. NAFNet (fp32+Local) ties unsharp on blur. Real-HAT-GAN on downsample
**ties Real-ESRGAN within JND on BG-LPIPS** (bear 0.46×, camel 0.15×); keep
Real-ESRGAN as headline. BSRGAN is the light twin; Real-HAT the recent twin.
Both HAT and NAFNet **reject fp16** (Softmax NaNs / LayerNorm overflow). Queue:
[`docs/EXPERIMENTS_QUEUED.md`](docs/EXPERIMENTS_QUEUED.md) — Q1–Q5 done; Q6
DiffBIR still deferred.

## Current state (verified 2026-07-28)

### Code repo `presley/`

- Branch: `main` (Real-HAT integration **uncommitted** in this working tree:
  `hat_arch.py`, `restoration.py`, `presley_ai.py`, tests, `experiments.yaml`,
  queue/handoff docs).
- Restorers: NAFNet + Real-HAT-GAN + BSRGAN + Real-ESRGAN + InstantIR + unsharp.
- **Hard rules:** NAFNet and Real-HAT-GAN reject `fp32=False`.
- Weights on disk: `NAFNet-GoPro-width64.pth`, `BSRGAN.pth`,
  `Real_HAT_GAN_sharper.pth`.

### Paper repo `68e8b6bb11d0dd9e62a67aef/`

- Local commits (ahead of Overleaf, **not pushed**):
  - `c67b3f6` — `tab:instantir-kill`
  - `78e7acb` — false fp16 NAFNet rows (superseded)
  - `83ea603` — NAFNet retraction + corrected fp32+Local
  - `0c30ad7` — Real-HAT twin: `CLAIM(tab:conditioned-twins)`,
    `CLAIM(sec:restoration-comparison)`, `chen2023hat` bib

### Q4 results (citable, CRF, `invariant_failures=[]`)

| Video | Hash | BG-LPIPS | vs Real-ESRGAN BG-LPIPS |
|---|---|---|---|
| bear | `661d092654b4e18d` | 0.201 | within JND (0.46×; HAT numerically better) |
| camel | `4ec11482c4cfc93e` | 0.160 | within JND (0.15×; ESRGAN numerically better) |

Comparators: Real-ESRGAN `e2cb6bed165d69b1` / `c29c94b5c290f208`. FG
indistinguishable (passthrough). Do **not** cite camel BG-PSNR HAT gap as
Goal-2.

### Running now

Nothing PRESLEY-owned.

## Deferred

- Q6 DiffBIR — not authorized (NAFNet ties unsharp; no new mechanism).
- Q7/Q8 Stream-DiffVSR / DC-VSR; Q9 noise; Q10 mask dilate.
- Overleaf push of paper `c67b3f6`…`0c30ad7` — only if human asks.
- Code-repo commit of Real-HAT integration — only if human asks.

## Ops gotchas

- Filter fixed-QP with **`codec=svtav1`**. Bare `restorer=` rematches VBR dirs.
- `fg_protect=True` (capital T).
- BG-PSNR never the Goal-2 verdict; use BG-LPIPS + `presley-compare`.
- NAFNet **and** Real-HAT-GAN **fp32 only**.

## Immediate next steps

1. Commit/push code Real-HAT integration if human wants.
2. Overleaf push of paper local commits only if human asks.
3. Otherwise: write-up-only D6 tables, or Q7/Q8 when diffusion-SR story is needed.

## Prompt for the next session

```
Pick up HANDOFF.md in /home/itec/emanuele/presley.

Q1–Q5 are done. InstantIR kill stands; NAFNet (fp32+Local) ties unsharp;
Real-HAT-GAN ties Real-ESRGAN within JND — keep Real-ESRGAN. Paper Overleaf
is still local-only (c67b3f6 → … → 0c30ad7) — do not push unless I ask.
Code-repo Real-HAT changes may still be uncommitted — check git status.

Ask me what to do next (code commit, Overleaf push, Q7/Q8, or write-up).
Read docs/EXPERIMENTS_QUEUED.md before any GPU job. Filter tip: codec=svtav1.
```

## Landmarks

| Path | Why |
|---|---|
| `docs/EXPERIMENTS_QUEUED.md` | Queue — Q1–Q5 closed |
| `src/presley/hat_arch.py` / `restoration.py` | Real-HAT (fp32) |
| `src/presley/nafnet_arch.py` | NAFNet (fp32+Local) |
| `results/661d092654b4e18d` / `4ec11482c4cfc93e` | Q4 HAT hashes |
| `68e8b6bb11d0dd9e62a67aef/` commit `0c30ad7` | Paper twin CLAIMs |
