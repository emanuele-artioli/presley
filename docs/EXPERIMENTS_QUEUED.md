# Experiment queue — restorers, InstantIR audit, D6 upgrades

**Status:** Q1–Q5 done 2026-07-28. InstantIR kill stands (corrected
settings + NAFNet negative gauge). Real-HAT-GAN (Q4) ties Real-ESRGAN within
JND — keep Real-ESRGAN as headline. Filter fixed-QP cells with `codec=svtav1`.
All degradations here are **fixed-QP/CRF only** (AGENTS.md hard rule). Never
commission VBR for these cells.

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
| Second blur diffusion (conditional) | **DiffBIR** | Not integrated | Only with a new mechanism argument. |
| Downsample diffusion (speed) | **Stream-DiffVSR** | Not integrated | Report §II; HF `Jamichsu/Stream-DiffVSR`. Catalog. |
| Downsample diffusion (quality) | **DC-VSR** | Wired on `feat/q8-dc-vsr` (`restorer: dc_vsr`); **inference blocked** | HF `Janghyeok/dc-vsr` is UNet-EMA weights only (no pipeline / SAP/TAP/DSSAG code). Stub raises `RuntimeError`. **fp32 only**. Weights: `hf download Janghyeok/dc-vsr --local-dir weights/dc-vsr` (isolate future deps — do not upgrade pinned `presley` env). |
| Cheap blur control | **unsharp** | Wired; already beats InstantIR as-run | Keep as baseline for blur Goal-2. |

**Goal-1 transport fact (do not retire blur):** after the S1 budget fix,
blur frees bits under fixed QP similarly to downsample (RESEARCH_LOG: bear
blur qp32 −26.9%, qp37 −15.4%; camel blur qp32 −10.0%; boundary camel qp37
+8.4%). Noise costs bits. InstantIR Goal-2 failure ≠ blur method failure.

```text
downsample → Real-ESRGAN (keep) | BSRGAN (few) | Real-HAT-GAN (done, twin) | Stream-DiffVSR (catalog) | DC-VSR (wired on branch; inference blocked)
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
| **Q7** | Stream-DiffVSR vs Real-ESRGAN | downsample + `stream_diffvsr` | Report §II speed diffusion SR |
| **Q8** | DC-VSR quality arm | **Wired on branch** (`feat/q8-dc-vsr`) — yaml dry-run cells; GPU Wave 2 blocked until upstream inference | Report §II quality ceiling |
| **Q9** | Noise threshold rematch | Matched-budget noise vs blur/downsample | Cleaner dead-end number |
| **Q10** | Mask dilate/erode/jitter | UFO mask noise arm, 2 DAVIS — still missing from yaml (`HOLE(sec:evaluation)`) | Referee mask half |

### Write-up status (2026-07-28)

**Landed in manuscript** (no new encode):
- D6.1 `tab:mask-sens` (gt vs YOLOE)
- D6.2 `tab:breadth-ext` (MOSEv2 / YouTube-VOS)
- DAVIS `tab:breadth`
- `tab:transport`, `tab:inpainters`
- (already landed earlier: `tab:goal2`, `tab:conditioned`, `tab:priced-trade`)

**Still open HOLEs (need data, not write-up):** av1 n>2, goal2/conditioned n>2,
priced-trade shrink_amount arm, Q10 dilate/erode.

---

## Integration order (next session)

See `HANDOFF.md` wave plan for Q6–Q10. Summary:
1. **Wave 1 (parallel):** Q7 / Q8 / Q10 integrate; Q9 yaml (code fix already landed)
2. **Wave 2:** fixed-QP GPU cells via `experiment-runner` (codec=`svtav1`)
3. **Wave 3:** `presley-compare` + paper fold
4. **Q6 DiffBIR:** human-gated — only with a new mechanism argument (NAFNet already ties unsharp)
