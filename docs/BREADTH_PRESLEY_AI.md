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
