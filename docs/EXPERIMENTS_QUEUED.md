# Experiment queue — restorers, InstantIR audit, D6 upgrades

**Status:** Q1–Q5 / Q7 / Q9 / Q10 done (2026-07-28…29). InstantIR kill stands.
Real-HAT-GAN ties Real-ESRGAN within JND — keep Real-ESRGAN as headline.
Stream-DiffVSR catalogued (no Goal-2 win). Filter fixed-QP cells with
`codec=svtav1`. All degradations here are **fixed-QP/CRF only** (AGENTS.md
hard rule). Never commission VBR for these cells.

**Comparison recipe** (unless a row says otherwise): match
`CLAIM(tab:conditioned)` — bear + camel, `block_size: 8`, `shrink_amount` /
`sa=0.25`, `fg_protect: true`, starved SVT-AV1 or x265 fixed QP; judge with
`presley-compare` and FG/BG LPIPS·DISTS (BG-PSNR is never the Goal-2 verdict).

---

## Restorer matrix

| Role | Model | Status in repo | Notes |
|---|---|---|---|
| Headline conditioned SR | **Real-ESRGAN** | Wired; results exist | Keep. Downsample transport. |
| Light GAN twin | **BSRGAN** | Wired; weights downloadable | Few selected fixed-QP cells only (same era as Real-ESRGAN). |
| Recent SR GAN | **Real-HAT-GAN** (`Real_HAT_GAN_SRx4_sharper`) | Wired (`restorer: real_hat_gan`); Q4 done | Within JND of Real-ESRGAN on BG-LPIPS; keep Real-ESRGAN. **fp32 only** (fp16 NaNs). Weights: `weights/Real_HAT_GAN_sharper.pth` (HF `Acly/hat`). |
| Blur diffusion (current) | **InstantIR** | Wired; **kill stands** | Corrected settings still lose to unsharp. |
| Blur non-diffusion gauge | **NAFNet** | Wired (`restorer: nafnet`); weights on disk | CNN deblur. **fp32 only**. Q5: ties unsharp within JND. |
| Second blur diffusion (conditional) | **DiffBIR** | Not integrated | Only with a new mechanism argument (Q6). |
| Downsample diffusion (speed) | **Stream-DiffVSR** | Wired (`restorer: stream_diffvsr`); **Q7 DONE** | No Goal-2 win vs Real-ESRGAN; ~2× slower. Keep RE. Lean env `stream-diffvsr` + `vendor/stream-diffvsr` (gitignored). Pins: `transformers>=4.45,<5`, `xformers`. |
| Downsample diffusion (quality) | **DC-VSR** | Wired stub (`restorer: dc_vsr`); **inference blocked** | HF `Janghyeok/dc-vsr` is UNet-EMA weights only. Stub raises `RuntimeError`. **fp32 only**. |
| Cheap blur control | **unsharp** | Wired; already beats InstantIR as-run | Keep as baseline for blur Goal-2. |

**Goal-1 transport fact (do not retire blur):** after the S1 budget fix,
blur frees bits under fixed QP similarly to downsample. Matched-budget noise
still costs bits (+77…+83% — Q9). InstantIR Goal-2 failure ≠ blur method failure.

```text
downsample → Real-ESRGAN (keep) | BSRGAN (few) | Real-HAT-GAN (done, twin) | Stream-DiffVSR (done, no win) | DC-VSR (stub; inference blocked)
blur       → InstantIR (kill stands) | NAFNet (gauge, ties unsharp) | DiffBIR (deferred) | unsharp (control)
```

---

## Queued experiments (fixed-QP only — do not run from this doc alone)

| ID | Purpose | Sketch | Paper target |
|---|---|---|---|
| **Q1** | LPIPS upgrade | **DONE** | D6 citability |
| **Q2** | InstantIR corrected settings | **DONE** — kill stands | Validate InstantIR kill |
| **Q3** | BSRGAN few cells | **DONE** | Light GAN twin |
| **Q4** | Real-HAT-GAN vs Real-ESRGAN | **DONE** | Recent SR GAN |
| **Q5** | NAFNet vs InstantIR/unsharp | **DONE** | Blur method vs model |
| **Q6** | DiffBIR (conditional) | Only with new mechanism argument | Second diffusion deblur |
| **Q7** | Stream-DiffVSR vs Real-ESRGAN | **DONE** — BG-LPIPS worse than RE (bear 0.251 vs 0.223; camel 0.190 vs 0.152; LPIPS within JND); ~2× slower; keep RE. Hashes `85d92dab` / `0bcc9956` | Report §II catalogue |
| **Q8** | DC-VSR quality arm | Wired stub — GPU blocked until upstream publishes inference (VAE/scheduler/SAP/TAP/DSSAG) | Report §II quality ceiling |
| **Q9** | Noise threshold rematch | **DONE** — matched-budget noise +77…+83% bits (vs unbudgeted +213…+334%); ds/blur −10…−27%; FG within JND | Cleaner dead-end number |
| **Q10** | Mask dilate/erode/jitter | **DONE** (r=4) — vs UFO none within JND; does not demo fg_protect defeat. YOLO under-cover remains the failure mode. Paper `tab:mask-morph` | Referee mask half |

### Write-up status (2026-07-29)

**Landed in manuscript:**
- D6 tables (`tab:breadth`, `tab:breadth-ext`, `tab:mask-sens`, `tab:transport`, `tab:inpainters`, …)
- Q9/Q10 fold — paper `9d6de4d` (`CLAIM(sec:noise-retire)`, `tab:mask-morph`)
- Q7 fold — paper `26cc066` (`CLAIM(tab:conditioned-stream-diffvsr)`)

**Still open (need data or a gate):**
- `HOLE(tab:av1)` n>2; `HOLE(tab:goal2/conditioned)` n>2; `HOLE(tab:priced-trade)` shrink_amount
- **Q8** upstream DC-VSR inference — **re-probed 2026-07-29: still unavailable.**
  HF `Janghyeok/dc-vsr` holds only `unet_ema/` + `.gitattributes` (no
  `model_index.json`, no VAE, no scheduler, no pipeline code); the project page
  (`daramgc.github.io/docs/Publications/dc-vsr`) links arXiv/YouTube only, no
  code repo; no official GitHub release found. Stub `RuntimeError` stands.
- **Q6** DiffBIR — human-gated mechanism argument only

See `HANDOFF.md` for the next-session pickup prompt.
