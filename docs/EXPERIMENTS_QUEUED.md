# Experiment queue — restorers, InstantIR audit, D6 upgrades

**Status:** catalog only. Do **not** treat this file as authorization to start
GPU runs — launch from a fresh session after reading HANDOFF.md and grepping
paper `HOLE()` markers. All degradations here are **fixed-QP/CRF only**
(AGENTS.md hard rule). Never commission VBR for these cells.

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
| Blur non-diffusion gauge | **NAFNet** | Not integrated | CNN deblur (not a GAN — modern deblur SOTA rarely is). HF: `opencv/deblurring_nafnet`, `mikestealth/nafnet-models`. |
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
| **Q1** | LPIPS upgrade | `presley-evaluate results/ --backfill-lpips` on 9 `fast_only` D6 hashes: yolo `bb665875e6e1552c`, `03f9308cf6a2c628`, `11756585ed472839`, `859af0848b792c98`, `c00f85574d84b59e`, `faf4decac7e0927f`, `e82612a5be69eeed`; fii86rku elvis `1e0586074c131092`, `b2d4411804b29e69` | D6 citability |
| **Q2** | InstantIR corrected settings | `degradation: blur`, `restorer: instantir`, hyperparams above; bear (+ camel) starved fixed QP, sa=0.25, fg_protect | Validate InstantIR kill |
| **Q3** | BSRGAN few cells | `degradation: downsample`, `restorer: bsrgan`, fixed-QP; bear+camel @ 1–2 starved QPs. **Ignore** existing VBR BSRGAN yaml entries | Light GAN twin |
| **Q4** | Real-HAT-GAN vs Real-ESRGAN | Integrate `restorer: real_hat_gan`; same recipe as `tab:conditioned` | Recent SR GAN |
| **Q5** | NAFNet vs InstantIR/unsharp | Integrate `restorer: nafnet` on blur; compare to existing blur+instantir / blur+unsharp | Blur method vs model |
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
