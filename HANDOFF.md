# Handoff — Q4 Real-HAT-GAN done; Q1–Q5 closed (2026-07-28)

## What triggered this handoff

Wave 2 Real-HAT-GAN (Q4) closed end-to-end: restorer wired (fp32-only),
fixed-QP bear/camel ran, `presley-compare` vs Real-ESRGAN, paper folded and
**pushed to Overleaf** (`0c30ad7`). Code Real-HAT integration committed and
pushed as `23aefab` on `presley` `main`.

## One-paragraph summary

PRESLEY degrades BG under fixed QP and restores client-side. InstantIR kill
stands. NAFNet (fp32+Local) ties unsharp on blur. Real-HAT-GAN on downsample
**ties Real-ESRGAN within JND on BG-LPIPS** (bear 0.46×, camel 0.15×); keep
Real-ESRGAN as headline. BSRGAN is the light twin; Real-HAT the recent twin.
Both HAT and NAFNet **reject fp16** (Softmax NaNs / LayerNorm overflow). Queue:
[`docs/EXPERIMENTS_QUEUED.md`](docs/EXPERIMENTS_QUEUED.md) — Q1–Q5 done; Q6
DiffBIR still deferred.

## Current state (verified 2026-07-28 evening)

### Code repo `presley/`

- Branch: `main` @ `23aefab` (pushed) — Real-HAT integration landed.
- Restorers: NAFNet + Real-HAT-GAN + BSRGAN + Real-ESRGAN + InstantIR + unsharp.
- **Hard rules:** NAFNet and Real-HAT-GAN reject `fp32=False`.
- Weights on disk: `NAFNet-GoPro-width64.pth`, `BSRGAN.pth`,
  `Real_HAT_GAN_sharper.pth`.
- Working tree may still have unrelated dirty files (`AGENTS.md`,
  `.cursor/rules/cursor-harness.mdc` host-rule sync) — not part of Q4.

### Paper repo `68e8b6bb11d0dd9e62a67aef/`

- Synced with Overleaf at `0c30ad7` (includes prior InstantIR/NAFNet commits
  `c67b3f6` → `78e7acb` → `83ea603` plus Real-HAT twin CLAIMs).

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
- Write-up-only D6 tables (mask-sensitivity, non-DAVIS breadth, etc.).

## Ops gotchas

- Filter fixed-QP with **`codec=svtav1`**. Bare `restorer=` rematches VBR dirs.
- `fg_protect=True` (capital T).
- BG-PSNR never the Goal-2 verdict; use BG-LPIPS + `presley-compare`.
- NAFNet **and** Real-HAT-GAN **fp32 only**.

## Immediate next steps

1. Write-up-only D6 tables / remaining paper holes (no new encode if data exists).
2. Q7/Q8 (Stream-DiffVSR / DC-VSR) when a downsample *diffusion* SR story is needed.
3. Q9/Q10 as bandwidth allows; Q6 only with a new mechanism argument.
4. Optionally clear leftover dirty host-rule files (`AGENTS.md`, cursor-harness)
   in a separate commit — not research work.

## Prompt for the next session

```
Pick up HANDOFF.md in /home/itec/emanuele/presley.

Q1–Q5 are done and pushed (code 23aefab; paper Overleaf 0c30ad7). InstantIR
kill stands; NAFNet (fp32+Local) ties unsharp; Real-HAT-GAN ties Real-ESRGAN
within JND — keep Real-ESRGAN. DiffBIR (Q6) still deferred.

Ask me what to do next (D6 write-up, Q7/Q8 diffusion SR, Q9/Q10, or other).
Read docs/EXPERIMENTS_QUEUED.md before any GPU job. Filter tip: codec=svtav1.
```

## Landmarks

| Path | Why |
|---|---|
| `docs/EXPERIMENTS_QUEUED.md` | Queue — Q1–Q5 closed |
| `src/presley/hat_arch.py` / `restoration.py` | Real-HAT (fp32) |
| `src/presley/nafnet_arch.py` | NAFNet (fp32+Local) |
| `results/661d092654b4e18d` / `4ec11482c4cfc93e` | Q4 HAT hashes |
| code `23aefab` / paper `0c30ad7` | Pushed landings |
