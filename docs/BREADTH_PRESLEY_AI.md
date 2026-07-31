# Breadth: does PRESLEY (not just ELVIS) survive outside DAVIS?

**Bounds pre-registered 2026-07-30, before any run.** Committed before launching,
per the bound-before-believing rule. Three findings this session were caught by
this discipline; two were real effects and one prediction was simply wrong, which
is the point.

## Why this experiment exists

Reviewer comment (dataset diversity, `reviewers_comments.md`): *"Only 10 videos
from the DAVIS dataset are used… far from representative of real-world streaming
content."* Marked **Done** on 2026-07-28 via `tab:breadth` (18 unused DAVIS) and
`tab:breadth-ext` (5 MOSEv2 + 4 YouTube-VOS).

**But `tab:breadth-ext` rests entirely on `elvis` runs.** Querying every result on
disk: the 9 non-DAVIS clips carry `baselines` and `elvis` arms only, and there are
**zero `presley_ai` runs outside DAVIS**. So the article's breadth evidence
speaks for ELVIS block-removal + inpainting, not for the degrade/restore pipeline
the paper is now organized around. Every three-goal result (S1, S1b, O2, and the
whole Wave-1 falsifier set) also uses the DAVIS-only n=8 probe suite.

A reviewer who checks will find this. This experiment closes it.

## Design

Mirror the existing ELVIS arm exactly and change only the component, so the new
rows drop straight into `tab:breadth-ext` beside the ELVIS ones and form a
matched triple (pristine baseline / ELVIS / PRESLEY) per clip per rate point.

- 9 clips: `mosev2/{8i1uo3x9,fii86rku,jxmcdk8k,ptq7rtia,zofozj6l}`,
  `youtube_vos/{0e4068b53f,282651c6f7,30fe0ed0ce,b1a8a404ad}`
- Native per-clip resolution (three differ: 640x360, 360x640 portrait, 640x480)
- x265, preset medium, **fixed QP 32 and QP 37** — both already exist for
  baselines and ELVIS. Fixed QP only; never VBR.
- `block_size: 8`, `shrink_amount: 0.25`, `fg_protect: true`,
  `composite_output: true`, `alpha=beta=0.5` — all identical to the ELVIS arm
- `degradation: downsample`, `restorer: realesrgan` (the project's default
  restorer; a second restorer is deliberately out of scope for this pass)

**18 runs.** LPIPS/DISTS backfilled afterwards (one `--backfill-*` flag per call).

**n = 9** clears the n>=6 bar in `research-log/hard-rules.md` rule 2b. But MOSEv2
(5) and YouTube-VOS (4) are different populations: **report split as well as
pooled, and do not pool if the two disagree in sign.**

## Bounds, written before reading any number

The honest prior is that this goes badly, and the reasons are specific.

1. **Bitrate, PRESLEY vs pristine baseline.** Standing result: bit relocation is
   regime-dependent — it needs a bit-starved operating point, and at comfortable
   bitrates plain encoding wins outright. **x265 QP32 is a comfortable point**,
   QP37 less so. Expect **-10% to +15%** at QP32 and **-20% to +10%** at QP37.
   **ALARM outside -35%…+35%**, or if QP37 is not cheaper than QP32 relative to
   baseline.
2. **FG quality, PRESLEY vs baseline.** `fg_protect` + `composite_output` means
   foreground pixels are passthrough, so FG deltas arise only from the encoder
   reallocating bits, not from restoration. Expect **|dFG-LPIPS| <= 0.02** and
   **|dFG-PSNR| <= 1.0 dB**. **ALARM above 0.05 FG-LPIPS** (the JND) — that would
   mean foreground protection is not holding on these clips, which is a
   correctness question about the masks, not a finding.
3. **BG quality, PRESLEY vs baseline.** This is where degradation shows up.
   Expect **dBG-LPIPS +0.01 to +0.12** (worse; we are discarding information and
   asking a GAN to invent it). **ALARM above +0.25** or on any *improvement*
   beyond -0.01 — restoring above a pristine encode is not credible.
4. **PRESLEY vs ELVIS.** The paper's expected chain is
   `presley_ai > elvis > roi > baseline` on FG at matched bitrate. On these clips
   ELVIS scored a strict WIN on 5/9. Expect PRESLEY to land **within +/-0.04
   BG-LPIPS of ELVIS**, direction genuinely uncertain. **No alarm band** — this is
   the open question, not a check.
5. **The dataset-specific risk, stated as the main hypothesis.** These clips were
   chosen *because* they have weaker foreground/background separation. The
   removability score's x10 background boost (`preprocessing.py:655-675`) depends
   on a meaningful FG/BG split; where UFO masks are diffuse or cover most of the
   frame, the score degrades toward "complexity alone" and selection should get
   worse. **Expect the non-DAVIS result to be worse than DAVIS**, and expect
   MOSEv2 (multi-object, dense) to be worse than YouTube-VOS. A negative result
   here is a **real outcome to report**, not something to tune away.
6. **Degraded-area sanity.** `shrink_amount: 0.25` must degrade ~25% of blocks on
   every clip, as it does on DAVIS. **ALARM if the realized footprint deviates by
   more than 3 percentage points** on any clip — that would mean the mask or the
   budget behaves differently on this data and the comparison is not matched.

## Decision rule, fixed in advance

- If PRESLEY is **within JND of ELVIS on BG-LPIPS and no worse on bitrate**, the
  breadth claim extends to PRESLEY and `tab:breadth-ext` gains its rows.
- If PRESLEY is **perceptibly worse than ELVIS or than the baseline**, that is
  reported as the finding, and the paper's breadth claim is explicitly narrowed
  to ELVIS in the text rather than left ambiguous.
- Per hard rule 2b a sub-JND gain is **not** a win.
- Any run with non-empty `invariant_failures` is excluded and named.

---

## Result (2026-07-31) — PRESLEY beats ELVIS on these clips, at a bitrate premium

All 18 runs completed with **empty `invariant_failures` and zero NaN**;
LPIPS/DISTS backfilled across all 54 non-DAVIS runs (108/108 succeeded).
MOSEv2 (5) and YouTube-VOS (4) **agree in sign on every comparison**, so the
pre-registration's condition for pooling is met and the pooled n=9 is reported
alongside the splits. Individually each split is `underpowered` at n=5 and n=4,
exactly as rule 2b predicts — the splits are shown for sign agreement, not as
independent results.

### PRESLEY vs pristine baseline

| | QP 32 | QP 37 |
|---|---|---|
| bitrate | **−16.60%** (2/9 higher, p=0.18) | −11.69% (3/9, p=0.51) |
| FG-LPIPS (primary) | +0.0056 `sub_jnd_significant` | +0.0044 `no_consistent_direction` |
| BG-LPIPS (primary) | +0.0470 `sub_jnd_significant` | +0.0333 `no_consistent_direction` |
| BG-PSNR | −3.41 dB `perceptual_loss` | −1.91 dB `perceptual_loss` |

Foreground protection **holds on this data**: the largest FG-LPIPS delta is
0.0056 against a 0.05 JND, an order of magnitude inside it. The background is
perceptibly worse on PSNR — expected, that is where the degradation is applied.

### PRESLEY vs ELVIS — the finding

| | QP 32 | QP 37 |
|---|---|---|
| bitrate | **+23.14%** (9/9, p=0.0039) | +14.95% (7/9, p=0.18) |
| BG-LPIPS (primary) | **−0.1268, 9/9, `perceptual_win`** | **−0.1202, 9/9, `perceptual_win`** |
| BG-DISTS | −0.0608, 9/9, `perceptual_win` | −0.0553, 9/9, `perceptual_win` |
| BG-PSNR | +6.88 dB, 9/9 | +6.37 dB, 9/9 |

Mandated wording for BG-LPIPS at both rate points: *"consistent across the
suite, statistically significant, and above the perceptual threshold — may be
worded as an improvement."* All four metrics agree in sign on all 9 clips at
both QPs.

**So the breadth claim does extend to PRESLEY — but not as a free win.** On
exactly the clips chosen for weak foreground/background separation, degrade +
restore is perceptibly better than block-removal + inpainting, and it costs
15–23% more bits to be so.

## Bounds ledger — two breaches and one underspecified rule

| bound | predicted | observed | outcome |
|---|---|---|---|
| 1 bitrate vs baseline, QP32 | −10%…+15% | **−16.60%** | **BREACHED** (more saving than predicted; alarm was −35%, not reached) |
| 1 bitrate vs baseline, QP37 | −20%…+10% | −11.69% | in range |
| 2 FG-LPIPS | \|d\| ≤ 0.02, alarm >0.05 | +0.0056 | in range, no alarm |
| 3 BG-LPIPS vs baseline | +0.01…+0.12 | +0.0470 | in range |
| 4 PRESLEY vs ELVIS | within ±0.04 BG-LPIPS | **−0.127** | **BREACHED, by 3x** |
| 5 non-DAVIS worse than DAVIS | qualitative | partly | see below |
| 6 realized footprint | 25% ±3pt | 25.0% (24.4% on the two portrait clips) | in range |

**Breach 1 investigated before being reported.** −16.6% is more bitrate saving
than predicted at a comfortable QP. It is not a measurement artifact: the
realized degraded footprint is exactly 25.0% on 16 of 18 runs (bound 6), and the
background pays for the saving with a −3.41 dB PSNR `perceptual_loss`. Removing
three quarters of the pixels in a quarter of the blocks plausibly buys ~17% of
the bitstream at QP32. **The prediction was simply too narrow**, in the same way
S1b's k=2 damage prediction was.

**Breach 4 is the finding itself.** The prior said "direction genuinely
uncertain, within ±0.04". PRESLEY beat ELVIS by three times that margin, in the
same direction on 9/9 clips, on two independent perceptual metrics, at two rate
points. **The pre-registered hypothesis (bound 5) was that this would go worse
than DAVIS because these clips have weak FG/BG separation. On the
PRESLEY-vs-ELVIS axis that was wrong.** The plausible reason is that weak FG/BG
separation hurts *block removal* more than it hurts *downsampling*: an inpainter
asked to hallucinate a blacked-out region with no clean foreground to anchor to
has nothing to work from, while a 2x downsample still transmits a quarter of the
pixels. That is a hypothesis, not a measurement — it is untested here.

**The decision rule was underspecified, and this is stated rather than
smoothed over.** It read: extend the claim if PRESLEY is *within JND of ELVIS
and no worse on bitrate*. The actual outcome — **perceptibly better on quality,
15–23% worse on bitrate** — is a case the rule does not cover, because it was
written expecting PRESLEY to be at best equal. The rule cannot be reinterpreted
after the fact to declare a pass. What can be said without stretching it:

- The quality result is unambiguous and passes every citability gate.
- **It is not a matched-rate comparison.** The paper's expected chain
  (`presley_ai > elvis` on quality) is specified *at matched bitrate*, and this
  is not that. A rate-matched PRESLEY-vs-ELVIS comparison on these clips is the
  experiment that would settle it, and it has not been run.
- So: report the win, report the premium, and **do not claim the chain is
  confirmed on non-DAVIS data**.

## Limitations

n=9 pooled (5+4, each split underpowered alone), x265 preset medium only, two
QPs, one restorer (Real-ESRGAN), one inpainter for the ELVIS arm (ProPainter),
one operator (downsample), one budget (25%), `block_size` 8 inherited from the
ELVIS arm rather than PRESLEY's default 16. Not rate-matched. The clips remain
short; the reviewer's "long-form content" sub-point is untouched by this
experiment.
