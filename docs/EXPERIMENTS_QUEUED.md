# Experiment queue — restorers, InstantIR audit, D6 upgrades

**Status:** Q1–Q3 and Q5 done 2026-07-28. InstantIR kill stands (corrected
settings + NAFNet negative gauge). Next: Wave 2 Real-HAT-GAN (Q4). Filter
fixed-QP cells with `codec=svtav1`. All degradations here are **fixed-QP/CRF
only** (AGENTS.md hard rule). Never commission VBR for these cells.

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
| Recent SR GAN | **Real-HAT-GAN** (`Real_HAT_GAN_SRx4_sharper`) | Not integrated | XPixelGroup/HAT; HF mirror `Acly/hat`. Prefer over expanding BSRGAN. |
| Blur diffusion (current) | **InstantIR** | Wired; **likely misconfigured** | See audit below before citing the kill. |
| Blur non-diffusion gauge | **NAFNet** | Wired (`restorer: nafnet`); weights on disk | CNN deblur. Vendored `nafnet_arch.py` + Local TLSC convert; **must run fp32** (fp16 overflows). Q5: ties unsharp within JND; beats InstantIR; no Goal-2 gain vs transmitted. |
| Second blur diffusion (conditional) | **DiffBIR** | Not integrated | Only if corrected InstantIR still loses to NAFNet/unsharp. |
| Downsample diffusion (speed) | **Stream-DiffVSR** | Not integrated | Report §II; HF `Jamichsu/Stream-DiffVSR`. Catalog. |
| Downsample diffusion (quality) | **DC-VSR** | Not integrated | Report §II; HF `Janghyeok/dc-vsr`. Catalog. |
| Cheap blur control | **unsharp** | Wired; already beats InstantIR as-run | Keep as baseline for blur Goal-2. |

**Goal-1 transport fact (do not retire blur):** after the S1 budget fix,
blur frees bits under fixed QP similarly to downsample (RESEARCH_LOG: bear
blur qp32 −26.9%, qp37 −15.4%; camel blur qp32 −10.0%; boundary camel qp37
+8.4%). Noise costs bits. InstantIR Goal-2 failure ≠ blur method failure.

```text
downsample → Real-ESRGAN (keep) | BSRGAN (few) | Real-HAT-GAN (queue) | Stream-DiffVSR / DC-VSR (catalog)
blur       → InstantIR (audit) | NAFNet (gauge) | DiffBIR (if InstantIR still bad) | unsharp (control)
```

---

## InstantIR misuse audit

Upstream tips ([instantX-research/InstantIR](https://github.com/instantX-research/InstantIR))
vs defaults in `src/presley/restoration.py` / `presley_ai.py`:

| Knob | Our default | Upstream guidance | Risk |
|---|---|---|---|
| `num_inference_steps` | **1** | paper tests ~**30-step DDIM** | Severely under-sampled |
| `creative_start` | **1.0** | **0.6–0.8** for fidelity (*smaller* = more creative) | Creative from step 0 |
| `preview_start` | **0.0** | **0.1–0.4** to preserve LQ fidelity | Previewer never for fidelity |
| `cfg` | 7.0 | paper default 7.0; tips 3–5 if over-smooth | OK as start |

Existing blur+instantir `CLAIM(tab:conditioned)` hashes remain “as-run / may be
misconfigured” until Q2 lands. Do not silently delete them.

**Corrected smoke settings (Q2):** `num_inference_steps: 20–30`,
`creative_start: 0.7`, `preview_start: 0.2`, `cfg: 7.0` (or 5.0 if over-smooth),
same blur transport + starved QP as the original cell, bear first.

---

## Queued experiments (fixed-QP only — do not run from this doc alone)

| ID | Purpose | Sketch | Paper target |
|---|---|---|---|
| **Q1** | LPIPS upgrade | **DONE 2026-07-27** — backfilled FG/BG/OV LPIPS on all 9 hashes (still tagged `fast_only` for other deferred metrics). Hashes: yolo `bb665875e6e1552c`, `03f9308cf6a2c628`, `11756585ed472839`, `859af0848b792c98`, `c00f85574d84b59e`, `faf4decac7e0927f`, `e82612a5be69eeed`; fii86rku elvis `1e0586074c131092`, `b2d4411804b29e69` | D6 citability |
| **Q2** | InstantIR corrected settings | **DONE bear+camel.** Hashes `36fb007975de0593` / `afd6dc1e5fa2ebe1` (steps=20, creative=0.7, preview=0.2). BG-LPIPS still loses to unsharp (bear 0.387 vs 0.352; camel 0.328 vs 0.304) — within LPIPS JND but worse; BG-PSNR crater ~2 dB (3.6–4.8×JND). Corrected ≈ as-run on LPIPS. **InstantIR kill stands**; proceed to NAFNet (Q5). Filter tip: `codec=svtav1`. | Validate InstantIR kill |
| **Q3** | BSRGAN few cells | **DONE** bear `432569abedc41fca` / camel `e437776f348cadbe`. vs Real-ESRGAN at matched bitrate: FG/BG indistinguishable (BG-LPIPS within JND); Real-ESRGAN still slightly better numerically. Both beat `none` on BG-LPIPS (bear BSRGAN 0.243 vs none 0.294, ~1.0×JND). Keep Real-ESRGAN; BSRGAN = light twin only. | Light GAN twin |
| **Q4** | Real-HAT-GAN vs Real-ESRGAN | Integrate `restorer: real_hat_gan`; same recipe as `tab:conditioned` | Recent SR GAN |
| **Q5** | NAFNet vs InstantIR/unsharp | **DONE (fp32+Local; retract fp16 crater).** Same hashes `93fcdf516cf7e363` / `8d2316a5e2d6128f` re-ran after banning fp16. Bear BG-LPIPS 0.356 ≈ unsharp 0.352 (within JND); camel 0.308 ≈ 0.304. Beats InstantIR-corrected; ~zero gain vs transmitted. InstantIR kill stands. **Root cause of first crater:** CUDA half() overflow in LayerNorm2d/SCA — not model trash. | Blur method vs model |
| **Q6** | DiffBIR (conditional) | Only if Q2+Q5 still damn InstantIR; blur+diffbir | Second diffusion deblur |
| **Q7** | Stream-DiffVSR vs Real-ESRGAN | downsample + `stream_diffvsr` | Report §II speed diffusion SR |
| **Q8** | DC-VSR quality arm | downsample + `dc_vsr` once integrated | Report §II quality ceiling |
| **Q9** | Noise threshold rematch | Matched-budget noise vs blur/downsample after `filter_frame_noise` `round(score)>0` fix (already in code) | Cleaner dead-end number |
| **Q10** | Mask dilate/erode/jitter | UFO mask noise arm, 2 DAVIS — still missing from yaml (`HOLE(sec:evaluation)`) | Referee mask half |

### Write-up only (data largely exists — no new encode)

- D6.1 mask-sensitivity (gt vs yolo) and D6.2 non-DAVIS breadth tables
- `tab:goal2`, `tab:conditioned`, `fig:priced-trade`, DAVIS `tab:breadth`

### BSRGAN weights

Official: `https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth`  
Placed under repo `weights/BSRGAN.pth` (also searched via realesrgan package
weights dirs by `restoration._instantiate_bsrgan_upsampler`).

---

## Integration order (next session)

1. Q1 (CPU/eval, cheap)  
2. Q2 InstantIR corrected smoke  
3. Q3 BSRGAN few cells (weights already on disk)  
4. Integrate NAFNet → Q5  
5. Integrate Real-HAT-GAN → Q4  
6. Q6 only if InstantIR still bad  
7. Stream-DiffVSR / DC-VSR (Q7/Q8) when downsample diffusion story is needed  
8. Q9 / Q10 as bandwidth allows  
