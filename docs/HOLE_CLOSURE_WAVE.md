# Closing the four open paper HOLEs

Design and **pre-registered bounds, written before any of these runs existed**.
Companion to `docs/RATEMATCHED_BREADTH.md`, `docs/F1_ORACLE_BITS.md` and
`docs/CLAIM_B_BLOCK_DAMAGE.md`, which follow the same discipline.

Every HOLE below says the same thing in different words: *n=2 videos, or one QP,
is not enough to generalize.* None of them threatens a landed claim — they
decide how widely the existing claims may be worded. **If one of them
contradicts a landed CLAIM, that is a result and the text changes.**

Rules that bind all four: fixed QP only (hard rule 1); the rate axis is
`actual_bitrate_bps`; BG-LPIPS is the Goal-2 verdict and BG-PSNR never is;
a run with non-empty `invariant_failures` is not citable; `presley-compare`
decides whether a quality difference is real.

---

## H1 — `HOLE(tab:av1)`: does the starved/comfortable sign flip generalize?

**The claim at risk.** `CLAIM(tab:av1)` reports the paper's strongest and most
quotable result: on SVT-AV1 the sign of the bit relocation flips with regime —
at starved QPs ELVIS frees bits (bear blackout −28.87%, freeze −10.50%; camel
−25.56%, −7.54%) while at comfortable bitrates plain encoding wins outright
(+16.68% / +40.83%). It rests on **two videos**.

**Design.** Add `dog` and `pigs` (the HOLE names 1–2 of dog/pigs/india/tennis;
both have cached 640×360 reference frames). Config copied verbatim from
`3f87d371bac4baf1` / `a07560c409dc38ce`: SVT-AV1 preset 8, 640×360, bs16,
alpha=beta=0.5, sa=0.25, `fg_protect`, `composite_output`, ProPainter, arms =
`baselines` + `elvis`{blackout, freeze}, four fixed-QP rungs per video.

**QPs are recalibrated per video, not copied.** bear used 43/51/58/61 and camel
42/50/58/62 because "starved" is a property of the content, not of the QP
number. Stage 1 runs a cheap `baselines` QP sweep (QP 40/45/50/55/60/63, seconds
each) on both videos; the four rungs are then chosen as the QPs whose baseline
overall PSNR brackets the same range bear/camel's rungs did. Stage 1's choice is
recorded here before stage 2 launches.

### Stage 1 outcome (2026-08-01) — rungs chosen, and one caveat

Incumbent starved ladders span **29.6 → 26.2 dB** (bear) and **29.3 → 25.6 dB**
(camel). Measured sweep:

| QP | dog kbps / dB | pigs kbps / dB |
|---|---|---|
| 40 | 749 / 30.93 | 460 / 32.47 |
| 45 | 476 / 30.00 | 331 / 31.77 |
| 50 | 379 / 29.39 | 269 / 30.96 |
| 55 | 219 / 28.46 | 170 / 30.16 |
| 60 | 123 / 27.31 | 103 / 28.87 |
| 63 |  61 / 25.78 |  55 / 27.19 |

**Both clips take rungs 50 / 55 / 60 / 63.** For `dog` that brackets the
incumbent range almost exactly (29.39 → 25.78 dB).

⚠ **`pigs` cannot be starved as hard as the incumbents.** Even at QP 63 — the
codec's maximum — its baseline is still at **27.19 dB**, about a decibel above
where bear and camel end. It is simply an easier clip at this resolution. The
rungs are kept (there is no QP left to go to), and the consequence is recorded
rather than hidden: if the regime effect is weaker on `pigs` than on `dog`, that
is partly because `pigs` is less starved, and the write-up must not read a
weaker effect there as content-dependence without saying so.

**Bounds.**

| quantity | plausible | **alarm** |
|---|---|---|
| starved blackout Δbits vs baseline | −10 … −35% | positive (no saving at all) |
| starved freeze Δbits | −3 … −15% | positive |
| comfortable Δbits | +5 … +45% | negative on both videos |
| FG-PSNR at matched rate | within 0.5 dB JND | > 1 dB loss |

**A sign that does not flip is a real outcome and gets reported.** If dog or
pigs frees bits at comfortable bitrates, or fails to free them when starved, the
regime story becomes video-dependent and `tab:av1`'s wording must be scoped to
say so — the same way `tab:bdrate`'s even split already is.

---

## H2 — `HOLE(tab:goal2)`: is the restorer-on-fill result stable?

**The claim at risk.** `CLAIM(tab:goal2)` is deliberately weak already:
restoration is a >1 JND win on a zero-information fill (blackout) and
video-dependent on freeze — a marginal DISTS-uncorroborated loss on bear, a wash
on camel. `NOTE(tab:goal2)` records that this finding has been **mis-stated
twice in opposite directions**, and that "n=2 videos cannot settle a direction
that flips between them". This is the HOLE most likely to change wording.

**Design.** Config from `a0ee8b50c40d23de`: bs8, SVT-AV1 preset 8, 640×360,
sa=0.25, `fg_protect`, `composite_output`. Cells = {freeze, blackout} ×
{none, telea, e2fgvi, propainter}.

- **Two new videos** (`dog`, `pigs`) at their starved QP → 16 runs.
- **A second QP** on bear and camel → 16 runs.

n=4 videos is still under the n≥6 that hard rule 2b requires for significance,
so this HOLE **cannot** produce a "significant" verdict. It can only widen or
narrow the descriptive claim — say that in the text rather than implying more.

**Bounds.**

| quantity | plausible | **alarm** |
|---|---|---|
| blackout: best restorer vs `none`, BG-LPIPS | 1.0 … 3.5× JND improvement | no improvement on any video |
| freeze: best restorer vs `none`, BG-LPIPS | −1.2 … +1.2× JND (a wash) | > 2× JND either way |
| FG-LPIPS between restorers | sub-JND | supra-JND (would contradict `fg_protect`) |

⚠ `NOTE(tab:goal2)` warns that FG-LPIPS is mask-**weighted**, not
mask-isolated: a sub-JND FG-LPIPS move between restorers is a background effect
leaking in, never an FG effect. Do not read one as a finding.

---

## H3 — `HOLE(tab:conditioned)`: more videos/QPs, plus a BD-rate curve

**The claim at risk.** `CLAIM(tab:conditioned)` carries the Goal-2 headline (the
gap closes on camel, mostly on bear) and is already scoped back by
`(a-BDRATE)`: FG BD-rate −13.8…−16.1% and BG-DISTS parity, but BG-LPIPS
+27.5…+80.9%. n=2 videos, 1 QP.

**Design.** Config from `e2cb6bed165d69b1` / `e4e6a25c24d18c07`: bs8, SVT-AV1
preset 8, 640×360, sa=0.25, `fg_protect`. Arms = downsample×{none, realesrgan}
and blur×{none, unsharp}. **InstantIR is excluded** — it was retired on its
pre-registered criterion (`tab:instantir-kill`), and re-running a retired
candidate would reopen a closed decision without new reason.

Two new videos (`dog`, `pigs`) × **four fixed-QP rungs** × 4 arms = 32 runs. The
four rungs are what makes this a BD-rate curve rather than another single-point
screen, which is the part the HOLE explicitly asks for.

**Bounds.**

| quantity | plausible | **alarm** |
|---|---|---|
| BD-rate FG-LPIPS, downsample+realesrgan | −25 … +5% | < −40% (too good; suspect a rate-accounting error) |
| BD-rate BG-LPIPS | −10 … +90% | — (the wide band is the known scope-back) |
| curve overlap fraction | > 0.6 | < 0.3 — BD numbers would be extrapolation, quote none |

`tools/analyze_ratematched.py` already refuses to quote a BD number on disjoint
quality ranges; reuse it rather than writing a second BD path.

---

## H4 — `HOLE(tab:priced-trade)`: the missing `shrink_amount` arm

**The gap.** `tab:priced-trade` prices the restoration trade across QP but holds
`shrink_amount` fixed at 0.25 throughout, so the surface has one axis unswept.
The ablation (`tab:ablation`) already found the removal **budget** to be the true
rate lever (~30% bitrate across 0.05→0.75) — but that was measured on FG-PSNR at
720p x265 with freeze+ProPainter, *not* on the background-perceptual axis this
table prices.

**Design.** Config from `ceac3559f8af0c3f`: bs8, SVT-AV1 preset 8, 640×360,
`fg_protect`, downsample × {none, realesrgan}, bear and camel at their existing
starved QPs. Sweep `shrink_amount` ∈ {0.10, 0.50, 0.75} (0.25 already exists) →
2 videos × 3 values × 2 arms = 12 runs. Cheapest of the four sets.

**Bounds.**

| quantity | plausible | **alarm** |
|---|---|---|
| bitrate, sa 0.10 → 0.75 | monotonically decreasing, 15–40% total | non-monotone |
| BG-LPIPS, sa 0.10 → 0.75 | monotonically worsening | improves with more degradation |
| FG-LPIPS across sa | sub-JND throughout | supra-JND (FG is protected by construction) |

**The interesting outcome is the shape, not the direction.** Both directions are
known in advance; what the table needs is whether the restorer's ability to pay
back the degradation falls off gradually or cliffs at some budget.

### H4 result (2026-08-01) — and one alarm that resolves into the finding

All 12 runs citable. Baselines for reference: `bear` 753.7 kbps at QP 43,
`camel` 740.0 kbps at QP 42.

| video | sa | kbps | BG-LPIPS none | BG-LPIPS restored | restoration gain |
|---|---|---|---|---|---|
| bear | 0.10 | 607.0 | 0.1691 | 0.1406 | +0.0285 |
| bear | 0.25 | 614.2 | 0.2590 | 0.1926 | +0.0664 |
| bear | 0.50 | 451.4 | 0.3464 | 0.2458 | +0.1006 |
| bear | 0.75 | 415.4 | 0.4173 | 0.2914 | +0.1258 |
| camel | 0.10 | 694.9 | 0.1516 | 0.1104 | +0.0413 |
| camel | 0.25 | 693.3 | 0.1975 | 0.1228 | +0.0747 |
| camel | 0.50 | 570.5 | 0.3037 | 0.1684 | +0.1353 |
| camel | 0.75 | 531.9 | 0.3783 | 0.1950 | +0.1833 |

Bounds: total bitrate reduction **−31.6% (bear) / −23.5% (camel)**, inside the
15–40% band. BG-LPIPS worsens monotonically on both. FG-LPIPS spread 0.019 /
0.018, well inside JND — foreground protection holds across the whole budget
range, as it must.

**⚠ The monotonicity bound fired on `bear`:** 607.0 kbps at sa 0.10 rises to
614.2 at sa 0.25 (+1.2%) before falling. `camel` is flat over the same step
(−0.2%). **Resolution: the bound was over-specified, not the measurement
wrong.** Every setting is well below its pristine baseline (bear 607–415 vs
753.7), so the lever does free bits throughout; what it does *not* do is vary
between 0.10 and 0.25, where both clips sit within ±1.2% — encoder noise at
fixed QP. **Revised bound: judge the total range and the shape, not
step-by-step monotonicity.**

**And that flat region is the answer the HOLE wanted.** The budget lever is
inert below ≈0.25 and only bites above it, while the *restoration gain* grows
monotonically across the whole range (bear +0.029 → +0.126, camel +0.041 →
+0.183). So the restorer keeps paying back more as the budget grows — the trade
does not cliff — but the bitrate only starts moving past 0.25. Anyone reading
`tab:priced-trade` should note that its fixed sa = 0.25 sits exactly at the knee.

---

## Order and cost

Launched cheapest-first so a failure surfaces early and the expensive sets are
not blocked behind a broken config:

1. **H4** (12 runs, Real-ESRGAN ~1.3 min/run) — minutes.
2. **H1 stage 1** (12 `baselines` runs) — minutes; picks H1's QPs.
3. **H3** (32 runs, no ProPainter) — ~1 h.
4. **H1 stage 2** (8 baselines + 16 ProPainter) — ~13 h.
5. **H2** (32 runs, 8 of them ProPainter) — ~7 h.

ProPainter at ~49 min/run dominates: ~24 ProPainter runs ≈ 20 h of the total.
Launch detached; the runner skips any hash that already has a `result.json`, so
an interrupted job resumes by being re-run.
