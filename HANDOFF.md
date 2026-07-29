# Handoff — Q7/Q9/Q10 closed; Q6 gated + Q8 blocked (2026-07-29)

## What triggered this handoff

Natural session boundary after Wave 1–3 for Q7/Q9/Q10: code merges + GPU
results + paper folds are committed and pushed. Next session owns the
remaining open items (Q6 DiffBIR gate, Q8 DC-VSR when upstream lands, plus
standing paper HOLEs).

## One-paragraph summary

PRESLEY perceptual compression queue Q1–Q5 and Q7/Q9/Q10 are closed.
Real-ESRGAN stays the headline conditioned SR (Stream-DiffVSR catalogued, no
Goal-2 win). Matched-budget noise still costs bits (+77…+83%). Mask morphology
at radius 4 is within JND vs UFO and does not demo fg_protect defeat — YOLO
under-cover remains the failure mode. Still open: **Q6 DiffBIR** (needs a
written mechanism argument before any wire), **Q8 DC-VSR** (HF weights-only;
no public inference), and paper HOLEs av1/goal2/priced-trade n>2.

## Current state (verified)

### Code repo `presley/`

- `main` @ *(this handoff commit)* — Wave 1 merges (Q7/Q8/Q9/Q10) + queue/
  HANDOFF refresh; `vendor/` gitignored.
- Restorers wired: realesrgan, bsrgan, real_hat_gan, stream_diffvsr, dc_vsr
  (stub), instantir, nafnet, unsharp, inpainters.
- **Hard rules:** NAFNet / Real-HAT-GAN reject `fp32=False`. Stream-DiffVSR
  uses isolated env (do not pip into `presley`).

### Paper repo `68e8b6bb11d0dd9e62a67aef/`

- `26cc066` — Q7 Stream-DiffVSR catalogue (`CLAIM(tab:conditioned-stream-diffvsr)`)
- `9d6de4d` — Q9 noise retire + Q10 `tab:mask-morph` (`HOLE(sec:evaluation)` filled)

### Running now

Nothing PRESLEY-owned. (`nvidia-smi` may show other users' jobs — check PIDs
before assuming a free GPU.)

## Result hashes (citable; `invariant_failures=[]`)

| ID | Hashes / notes |
|---|---|
| Q7 | Stream-DiffVSR `85d92dabda4b9907` (bear) / `0bcc99560e456dba` (camel) vs RE `e2cb6bed165d69b1` / `c29c94b5c290f208` |
| Q9 | noise `f27b0f17` / `4e2d1e22` / `d1e32185` / `65c5aae4`; ds/blur comparators in yaml Q9 section; baselines `21e85db5` / `2214e6bf` / `83cb00c9` / `a2cd5ebc` |
| Q10 | morph `08cf9e58` / `e802ae0a` / `61ef4d76` / `754b7d6e` / `9d995b6c` / `bee9c70a`; fair baseline UFO none `1dca3441` (bear) / `40e78d87` (camel) — **not** gt |

## Open questions / decisions

1. **Q6 DiffBIR:** Do you have a mechanism argument why DiffBIR’s prior would
   beat NAFNet/unsharp on spatial-Gaussian+CRF blur? NAFNet already ties
   unsharp with no Goal-2 gain. **If no → leave deferred; do not commission.**
2. **Q8 DC-VSR:** Re-check HF `Janghyeok/dc-vsr` / author code for a runnable
   pipeline (VAE + scheduler + SAP/TAP/DSSAG). Until that exists, the stub’s
   `RuntimeError` is correct — no GPU cells.
3. **Standing HOLEs** (optional if paper needs them next): av1 n>2,
   goal2/conditioned n>2, priced-trade shrink_amount.

## Immediate next steps (for the next session)

1. Read this file + `docs/EXPERIMENTS_QUEUED.md`. Confirm `git status` clean
   on both repos (or only expected dirt).
2. **Ask before Q6.** No mechanism → skip DiffBIR entirely.
3. **Probe Q8 readiness:** `hf` / WebFetch `Janghyeok/dc-vsr` file list; if
   inference code appeared, wire for real in a worktree `feat/q8-dc-vsr-infer`
   then fixed-QP cells; else leave stub.
4. If paper HOLEs are the priority instead, grep
   `HOLE(tab:av1|goal2|priced-trade)` and plan cells under fixed QP only.
5. After any new GPU results: `presley-compare`, bound-before-believing,
   `/update-paper`, refresh this HANDOFF.

## Ops gotchas

- Filter fixed-QP with **`codec=svtav1`**. Bare `restorer=` rematches VBR dirs.
- `fg_protect=True`. BG-PSNR never Goal-2 verdict.
- Q7 lean env: `STREAM_DIFFVSR_ROOT=…/vendor/stream-diffvsr`,
  `STREAM_DIFFVSR_PYTHON=…/envs/stream-diffvsr/bin/python`. Keep
  `transformers>=4.45,<5` + `xformers` in that env only.
- Q10 morph fair A/B is **UFO none**, not gt (camel UFO FG ~1 dB above gt).
- Substantive code → worktree + branch. Run-only stays on this checkout.
- Never `rm` wholesale `results/` / `dataset/` / `cache/`.

## Prompt for the next session

```
Pick up HANDOFF.md in /home/itec/emanuele/presley.

Q1–Q5 and Q7/Q9/Q10 are closed and pushed (code Wave-1 merges + paper
9d6de4d/26cc066). Real-ESRGAN stays headline; Stream-DiffVSR catalogued (no
Goal-2 win). Matched-budget noise +77…+83% bits. Mask morph r=4 within JND.

Still open:
1. Q6 DiffBIR — ASK for a mechanism argument before any wire; else skip.
2. Q8 DC-VSR — only if upstream published inference (not weights-only);
   otherwise leave the RuntimeError stub.
3. Optional paper HOLEs: av1 / goal2-conditioned n>2 / priced-trade shrink.

Read docs/EXPERIMENTS_QUEUED.md. Filter tip: codec=svtav1. fp32 for NAFNet/HAT.
```

## Landmarks

| Path | Why |
|---|---|
| `docs/EXPERIMENTS_QUEUED.md` | Queue status |
| `src/presley/stream_diffvsr.py` / `dc_vsr.py` | Q7/Q8 glue |
| `src/presley/preprocessing.py` | Q10 mask morph |
| Paper `CLAIM(tab:conditioned-stream-diffvsr)` / `tab:mask-morph` / `CLAIM(sec:noise-retire)` | Landed claims |
| `vendor/stream-diffvsr` + conda `stream-diffvsr` | Q7 runtime (gitignored vendor) |
| Real-ESRGAN hashes `e2cb6bed…` / `c29c94b5…` | Q7 comparators |
