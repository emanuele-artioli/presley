# Wave 1 — cheap falsifiers

Each item is <=1 day and is designed to **kill its own workstream** cheaply,
before any multi-day build. Bounds are written *before* the number is read (host
rule: bound before believing); a result outside its stated range is an alarm to
investigate, not a finding to report.

Probe videos come from `tools/select_probe_videos.py`: **camel, motorbike,
drift-straight, dancing** (cluster medoids over `scripts/audit_videos.py`
attributes). `bear` is deliberately excluded -- it never separates from camel at
any k from 2 to 8 (0.92 sd apart vs a 4.15 sd median pairwise distance), so
running both buys nothing.

JND thresholds (`src/presley/compare.py`): PSNR 0.5 dB, SSIM 0.05, LPIPS 0.05,
DISTS 0.05, VMAF 6.0. **Deltas within JND are "no perceptible difference" --
never a trend, never a win.**

| ID | Goal | Test | Status |
|---|---|---|---|
| **F1** | **1** | **All-intra leave-one-SB-out bit map vs EVCA** | **DONE — direction CLOSED, EVCA already captures 93-99%** |
| F2 | 1 | 64x64-snapped vs scattered 16x16 selection | todo |
| **F3** | **2** | **`--tune 0` (VQ) vs the PSNR default** | **DONE — confound confirmed; `sub_jnd_significant` at n=7 (p=0.0156)** |
| **F4** | **2** | **`--film-grain` strength sweep, denoise+synthesize** | **DONE — direction CLOSED for this content; -12.8% bits at a perceptible quality cost** |
| **F5** | **2** | **Transform-aligned AC truncation vs Gaussian blur** | **DONE — my stated mechanism REFUTED; metrics disagree, no win claimed** |
| **F6** | **3** | **Encoder-side FG gate: does restoring FG clear JND?** | **DONE — gate mechanism validated, but no FG restorer worth gating yet** |
| **F7** | **2** | **Chroma-first degradation: oracle-colorization ceiling** | **DONE — O3 CLOSED; only 5% of bits are chroma, and a perfect colorizer wins sub-JND** |

---

## F3 — SVT-AV1 `tune`: is the PSNR default a confound?

**Question.** `encode_video_svtav1_qp` passed `rc=0:q={qp}` with no `tune`.
SVT-AV1 offers 0=VQ, 1=PSNR, 2=SSIM. If the default is PSNR, then every
perceptual claim in the paper was measured under an encoder optimizing an
objective the paper does not claim -- a confound under all of it.

**Bounds, stated before measuring.** If unset is bitwise-identical to `tune=1`,
the confound is real. For VQ vs PSNR at matched rate expect LPIPS to improve
0.005-0.03, PSNR to drop 0.2-1.0 dB, bitrate to move <=10%. Alarm if LPIPS moves
>0.1 or bitrate >30% -- too large for an RDO tuning flag.

**Method.** camel + all four probe medoids, 640x360, preset 8, fixed QP.
Baseline `tune=1 q=43`; VQ arm re-encoded at the QP whose byte size lands closest
to it, so the comparison is rate-matched rather than QP-matched (VQ is +6.8% bits
at equal QP, so a same-QP comparison would flatter it). Decoded with ffmpeg --
OpenCV cannot decode AV1 in this environment and returns empty frame lists,
which silently produces NaN metrics.

**Result 1 — the default is PSNR.** With `tune` omitted, SVT-AV1 logs
`tune : PSNR` and emits a file **bitwise identical** to `tune=1` (md5
`4e4625cd50efec6e`, 298158 B). At QP 43 on camel: VQ +6.8% bits, SSIM-tune
-9.4%. Confound confirmed.

**Result 2 — but the effect is sub-JND everywhere.** Rate-matched, VQ vs PSNR:

| video | dPSNR | dLPIPS | dDISTS | dBits |
|---|---|---|---|---|
| camel | -0.14 | -0.0172 | -0.0148 | -0.2% |
| motorbike | -0.21 | -0.0305 | -0.0105 | +1.2% |
| drift-straight | -0.36 | -0.0173 | -0.0106 | -2.5% |
| dancing | -0.11 | -0.0270 | -0.0097 | +2.4% |
| elephant | -0.21 | -0.0150 | -0.0107 | -1.5% |
| **mean** | | **-0.0214** | **-0.0112** | |

(Negative = VQ better on LPIPS/DISTS, worse on PSNR.)

**Verdict (revised 2026-07-29 under hard rule 2b).** The original write-up said
"consistent 5/5, so not video-determined". That was an over-claim: the exact
two-tailed sign test floors at 2/2^n, so **n=5 floors at p=0.0625 and can never
reach alpha=0.05**. At n=5 this was `underpowered`, not significant. (An earlier
draft of the spawned-chip brief also quoted p=0.031 for 5/5 -- that is the
ONE-tailed value and is invalid here, because the direction was read off the
data first. Never quote 0.031 for 5/5.)

**Extended to n=7** by adding tennis (cluster 1) and india (cluster 3), chosen
because the original five over-represented cluster 4 (camel + elephant):

| video | dLPIPS | dDISTS |
|---|---|---|
| tennis | -0.0160 | -0.0059 |
| india | -0.0140 | -0.0059 |

Unanimous **7/7 on both LPIPS and DISTS**, exact two-tailed sign p = **0.0156**
(family size 1 -- this is a single comparison, not one of a candidate family, so
no Holm correction applies). Mean dLPIPS -0.0196, mean dDISTS -0.0097, max |d|
0.0305 -- all below the 0.05 JND.

`presley-compare`'s suite layer returns **`sub_jnd_significant`** for both
metrics. Mandated wording, which is the only phrasing allowed:

> consistent and statistically significant across the suite but BELOW the
> perceptual threshold -- report as a small, reproducible, imperceptible
> effect; never as a perceptual win, a quality improvement, or a 'better'
> result.

So: **the confound is real but benign.** Past restorer comparisons are not
invalidated by having run PSNR-tuned, because switching the objective does not
move any metric perceptibly. This closes a risk rather than opening one, and VQ
must **not** be reported as a free win -- that would be exactly the
"imperceptible delta dressed up" failure the hard rules forbid.

**Landed.** `tune` is now an optional `codec_params` key on all three SVT-AV1
callers (baselines, elvis, presley_ai), defaulting to omitted so output stays
bit-exact and all 694 existing result hashes remain valid. The point is
reproducibility -- the paper should be able to state the tuning it used rather
than inherit an invisible default.

---

## F4 — SVT-AV1 film grain: denoise server-side, synthesize client-side

**Question.** AV1 film-grain synthesis is PRESLEY's own thesis already shipped
in a codec: strip a texture server-side because it is expensive to code, and
regenerate it client-side from parameters. If the codec's built-in version of
that already captures the available win on natural video, the Goal-2 operator
family has to beat it rather than merely beat plain encoding -- and if it does
not fire at all on this content, that is worth knowing before building
operators premised on the same idea.

**Mechanism check (run before the suite, and it changed the design).**
SVT-AV1 v1.8.0 takes `film-grain=N` (N=1..50) plus `film-grain-denoise=1` via
`-svtav1-params`. `N` is a *denoising strength*, not a cosmetic grain amount,
and it dominates the result:

- **`film-grain=1` is a literal no-op** -- same byte size as grain-off (71719 B)
  and a bitwise-identical decode.
- **`film-grain=8` cost +2.7% bits** on bear rather than saving any.
- **`film-grain=50` saved 18.7% bits** (58287 B vs 71719 B).

So a single-strength on/off test would have measured whichever arbitrary point
was picked. **F4 sweeps strength**, which is the same correction F5 earned the
hard way.

**The decode is verified, because this is where the measurement could have
died silently.** The bitstream does carry grain (`trace_headers`:
`film_grain_params_present=1`, `apply_grain=1`, fresh `grain_seed` per frame),
but that only proves the encoder wrote it. dav1d 1.5.3 (`--filmgrain 0/1`,
installed in the separate `av1tools` conda env) gives the controlled A/B on one
identical file: high-frequency energy **17.49 grain-suppressed vs 19.33
grain-applied**. ffmpeg's libaom-av1 decode of the same file scores **19.34**,
matching the grain-applied case, so **ffmpeg does synthesize the grain** and the
normal decode path is sound.

Worth recording because the intuitive check gives the wrong answer:
`film-grain=50` decodes *smoother* than grain-off (19.33 vs 21.19) even with
grain correctly applied, because the denoiser removes more texture than the
synthesizer puts back. Reading that as "the decoder is dropping the grain" is
wrong; only the fg0-vs-fg1 comparison on one file settles it. Compare
strength levels against each other, never a grain arm against a grain-off arm,
when asking whether synthesis happened.

**Suite: n=8, pre-registered.** `tools/select_probe_videos.py -k 8` verbatim --
motorbike, drift-straight, drift-turn, color-run, dancing, dogs-jump,
bike-packing, bear. n=8 is chosen so the exact two-tailed sign test can reach
p=2/2^8=0.0078 and survive Holm correction over the 3 strength levels; the k=4
probe set would floor at 0.125 and could not have been significant at all. Note
this set swaps camel for bear, which W0.3 showed are interchangeable (0.92 sd
apart). 640x360, preset 8, fixed QP 43, `tune` omitted -- the same operating
point as F3, so the arms differ only in the grain flags.

**Two measurements, kept separate.** (a) bits at matched QP -- what the operator
saves; (b) quality at matched rate, with the grain arm's QP re-searched to land
nearest the control's byte size, exactly as F3 did, since comparing quality at
equal QP would flatter whichever arm spends more bits.

**Bounds, stated before reading any number.**

- *Bits at matched QP.* Level 50: saves 5-30%, larger on textured clips. Level
  8: -5% to +5%, i.e. plausibly costs bits (bear did). **Alarm if** any level
  saves >50%, or if level 50 costs bits on a majority of the suite.
- *Quality at matched rate, level 50 vs off.* PSNR **-0.3 to -2.5 dB** --
  synthesized grain is uncorrelated with the source, so a PSNR loss is expected
  and is not by itself evidence of harm. LPIPS and DISTS **-0.04 to +0.01**.
  **Alarm if** PSNR drops >4 dB (grain far too strong, or a rate-match failure),
  or if LPIPS improves by >0.05, which would clear JND -- too large for a codec
  flag on 640x360 content, and more likely a decode or pairing bug than a win.
- *Expected shape.* Bit saving tracks `bg_texture`; DAVIS clips are mostly
  clean digital video with little real grain, so the honest prior is **a small
  saving on a minority of clips and near-nothing on the rest**, which would
  close the direction rather than open it.

**Result 1 — bits at matched QP (q=43), grain arm vs control.**

| video | L=8 | L=25 | L=50 |
|---|---|---|---|
| bear | +3.7% | -5.7% | -19.2% |
| bike-packing | +3.2% | -4.5% | -14.8% |
| color-run | +4.1% | +0.0% | -5.7% |
| dancing | +1.2% | -3.4% | -10.9% |
| dogs-jump | +8.3% | -1.8% | -13.9% |
| drift-straight | +2.1% | -0.7% | -7.3% |
| drift-turn | +2.7% | -3.9% | -14.7% |
| motorbike | +3.7% | -4.4% | -15.8% |
| **mean** | **+3.6%** | **-3.0%** | **-12.8%** |

**Level 8 costs bits on 8/8 videos.** The grain parameters are not free, and at
low strength the denoise saves less than they cost. Only level 50 saves on the
whole suite. Rate-matching then landed within 3.1% of the control's byte size
on every arm (mean 1.1-2.0%), so the quality comparison below is at equal spend.

**Result 2 — quality at matched rate. It is a perceptible degradation.**
Via `assess_metric`, family_size=3 (the three strength levels), n=8:

| level | dPSNR | dSSIM | dLPIPS | dDISTS | LPIPS verdict |
|---|---|---|---|---|---|
| 8 | -0.43 | -0.0087 | +0.0044 (5+/3-) | +0.0000 | `no_consistent_direction` |
| 25 | -1.00 | -0.0291 | +0.0222 (7+/1-) | +0.0147 | `no_consistent_direction` |
| 50 | **-2.01** | **-0.0663** | **+0.0675 (8/8 worse)** | +0.0427 | **`perceptual_loss`** |

(Positive dLPIPS/dDISTS = worse.) At level 50 the sign test is unanimous 8/8 on
every metric, p=0.0078, Holm-corrected 0.0234, and PSNR, SSIM and LPIPS all
clear JND **in the worse direction**. Mandated wording:

> consistent across the suite, statistically significant, and above the
> perceptual threshold, but in the WORSE direction -- this is a perceptible
> degradation; word it as a cost or a regression, never as a win

**This surfaced a defect in the suite layer itself**, now fixed: `assess_metric`
picked between its two above-JND verdicts on `clears_jnd` alone, so this arm
first came back as `perceptual_win` with wording saying it "may be worded as an
improvement". F4 is the first suite to reach that branch in the worse direction.
No landed verdict changed. See the `perceptual_loss` verdict in `suite.py`.

**Alarm raised and closed.** dLPIPS +0.0675 breached the registered bound
(-0.04 to +0.01) and dogs-jump's -3.48 dB PSNR breached the -2.5 dB floor. Both
breaches are in the *harmful* direction, so they inflate no claim, but they were
investigated before being reported. Closed on four checks: the rate match is
tight (<=3.1%); grain synthesis is verified applied (see the mechanism check);
the effect is **monotone in strength on all 8 videos with no exceptions**, a
dose-response a pairing or decode bug does not produce; and the sign is what the
content predicts. The bound was simply set too generously -- the pre-registered
prior said "near-nothing", and the honest answer is "actively harmful".

**Verdict: direction CLOSED for this content.** AV1's own
degrade-server-side/restore-client-side is a real bit saving (-12.8% at level
50) that is **not free**: at matched rate, plain encoding at a finer QP beats
denoise-plus-synthesize on every video in the suite. There is no operating point
in 1..50 that both saves bits and holds quality -- low strength costs bits for
nothing, high strength buys bits with perceptible damage.

**Limitation, and it bounds the claim tightly.** DAVIS is clean digital video
with little real grain, so the denoiser removes genuine detail and the
synthesizer replaces it with noise uncorrelated with the source. This closes
film-grain synthesis **on this dataset**, not on grainy cinematic source, where
the mechanism is designed to pay off. Do not generalize this to "parametric
texture regeneration does not work" -- it says the codec's built-in version does
not help on the content PRESLEY actually evaluates on. The useful transfer to
Goal 2 is narrower and sharper: a restoration prior that is *statistical* rather
than *conditioned on the source* loses at matched rate here, which is an
argument for content-conditioned restorers over parametric resynthesis.

---

## F7 — chroma-first degradation: is there enough chroma to be worth taking?

**Question.** O3 is the only axis orthogonal to everything tried -- every
operator to date is luma-structure focused. The premise is that chroma acuity is
far below luma acuity, chroma is a non-trivial share of the bitstream, and
colorization conditioned on preserved luma is well-posed. The premise that can
be falsified cheaply is the middle one.

**Why this can be settled without building a colorizer.** The best a
colorization prior can ever do is return the *original* chroma. So the ceiling
of O3 is exactly: flatten chroma, spend the freed bits on luma, then reinstate
ground-truth chroma at the decoder. That oracle is free to compute -- take the
decoded luma plane from the degraded arm and the chroma planes from the
reference. **If a perfect colorizer does not beat the control, no real one
will**, and O3 dies without a single model being trained.

**Method.** Same n=8 pre-registered probe suite, 640x360, preset 8, fixed QP 43,
`tune` omitted -- identical to F3 and F4 so the arms differ only in chroma.
Three arms:

- **control** -- normal encode at q43.
- **flat, unrestored (floor)** -- chroma set to neutral (`lutyuv=u=128:v=128`)
  before encoding, then rate-matched back to the control's byte size by lowering
  QP, so the freed bits are actually spent on luma rather than banked. This is
  the outcome with *no* restoration.
- **flat + oracle chroma (ceiling)** -- the same bitstream, decoded, with the
  reference's U and V planes substituted back in. Simulates a perfect
  colorization prior.

Whole-frame rather than BG-only on purpose: whole-frame is the **upper bound**
on the available saving, and an upper bound is what kills or spares a
workstream. A BG-only version can only save less.

**Bounds, stated before reading any number.**

- *Bit saving from flattening chroma at matched QP.* AV1 is already 4:2:0, so
  chroma is subsampled before we touch it; expect **8-25%**. **Alarm if <2%**
  (would mean the flatten did not reach the encoder) **or >45%** (too large for
  two already-subsampled planes on natural content).
- *Oracle ceiling vs control at matched rate.* The freed bits go to luma, so
  expect PSNR **+0.2 to +1.5 dB better** and LPIPS **-0.03 to +0.01**. **Alarm
  if the oracle is worse than control on PSNR** -- at equal bits with identical
  chroma that should not happen, and would indicate a rate-match or
  plane-recombination bug rather than a finding. **Alarm if better by >3 dB.**
- *Unrestored floor.* Expected to be badly worse (a desaturated frame); it is
  recorded to bracket the gap the colorizer would have to close, not as a
  candidate result.

**Decision rule, fixed in advance.** O3 survives **only** if the oracle ceiling
clears JND against the control on the primary metric. A sub-JND ceiling means
the entire workstream is competing for an imperceptible prize, which hard rule
2b forbids reporting as a win no matter how consistent it is.

**Result 1 — there is barely any chroma to take.** Bits freed by removing
**all** chroma information at matched QP:

| video | dBits | | video | dBits |
|---|---|---|---|---|
| color-run | **-12.59%** | | drift-turn | -4.97% |
| drift-straight | -6.47% | | bike-packing | -3.72% |
| dancing | -5.02% | | motorbike | -3.16% |
| dogs-jump | -3.06% | | bear | -1.59% |

**Mean -5.07%**, and that is the *upper bound*: whole-frame, total chroma
destruction, with no restoration cost counted. A BG-only operator can only save
less. (color-run is aptly named -- it is the one genuinely chroma-heavy clip in
the suite, and even it gives up only 12.6%.)

**Alarm raised and closed.** The registered bound was 8-25% with an explicit
alarm below 2%; bear came in at 1.59% and 7/8 videos fell under the bound. This
was checked before being reported, because "the filter never reached the
encoder" produces exactly this signature. It did reach it: the flat arm's
decoded chroma deviates from neutral by **exactly 0.000** and its mean HSV
saturation is **exactly 0**, against 5.007 and 55.1 for the control. The chroma
really is gone, and it really was only worth ~5%. The bound was set from
general-video intuition and was simply too optimistic for this operating point.

**Result 2 — even a perfect colorizer wins only an imperceptible amount.**
At matched rate (within 3.2%), n=8, family_size=1:

| arm | dPSNR | dSSIM | dLPIPS | dDISTS | LPIPS verdict |
|---|---|---|---|---|---|
| **ceiling** (oracle chroma) | +0.65 | +0.0031 | **-0.0305** (8/8 better) | -0.0225 | **`sub_jnd_significant`** |
| floor (no restoration) | -4.82 | +0.0034 | +0.1435 (8/8 worse) | +0.0986 | `perceptual_loss` |

The ceiling is unanimous 8/8 better on LPIPS and DISTS (Holm p=0.0078) but
**-0.0305 against a 0.05 JND -- it does not clear it**. PSNR does clear JND
(+0.65 vs 0.5) and comes back `perceptual_win`, but PSNR is a corroborating
metric and `assess_metric` attaches the hard rule 3 caveat to it automatically;
it cannot carry this claim.

**Verdict: O3 CLOSED, on the pre-registered decision rule.** The rule fixed in
advance was that O3 survives only if the oracle ceiling clears JND on the
primary metric. It does not. The full shape of the trade is worse than that
single line suggests:

- the prize is **5% of the bitstream** and a **sub-JND** quality change;
- to collect it, a colorization prior would have to close the floor-to-ceiling
  gap of **0.174 LPIPS** (from +0.1435 to -0.0305) **perfectly**;
- any real colorizer lands strictly between those two numbers, and the entire
  span between "perfect" and "useful" is smaller than the JND it is trying to
  win.

So the premise fails at its middle claim. Chroma acuity being low and
colorization being well-posed are both true and both irrelevant, because chroma
is **not** a non-trivial share of this bitstream. No colorization model needs to
be built to know this.

**Scope.** Measured at fixed QP 43 -- the starved operating point where hard
rule 8 says generative methods can pay off at all, and the one the paper's
claims live at. AV1 is already 4:2:0 before we touch it, and at coarse QP the
chroma planes are cheap to code (the control's own decoded chroma deviates from
neutral by only 5.007). At a much finer QP chroma would be a larger share, but
that is not a regime where the rest of the approach applies. This closes O3 for
PRESLEY's operating point, which is what the falsifier was for.

---

## F6 — encoder-side FG gate (blocked on GPU; the cheap version is a tautology)

**Question.** FG is protected, so `composite_passthrough` reproduces transmitted
FG pixels bit-exact and the restorer's FG output is discarded. Would using the
restored FG ever be better? If so, an encoder-side per-block gate (~1 bit/block
through the existing side channel) could take the better of the two, giving FG
a construction-level guarantee of never being worse than transmitted.

**Bounds, stated before measuring.** Real-ESRGAN runs over the whole frame
including FG, which was never degraded; SR on undegraded content usually adds
artifacts, so expect restored FG *worse* by 0.5-5 dB PSNR, LPIPS ambiguous, and
5-35% of FG superblocks favouring restoration. **Alarm if** restoration wins
globally by >2 dB (would contradict fg_protect) **or the favoured fraction is
exactly 0% or 100%** (misalignment bug).

**Result: the alarm fired, and it was correct.** Across all 24 clean 640-wide
Real-ESRGAN runs (35,612 FG superblocks): delta exactly **+0.00 dB** and
**0.0%** of blocks favoured restoration, on every single run.

**Diagnosis.** `_adaptive_block_pyramid_upscale` (`restoration.py:624`) writes
**only blocks with strength > 0**. Verified on `e2cb6bed165d69b1` frame 0:
57,582 pixels differ between restored and transmitted, **all of them inside the
degraded region, zero outside**. So `restored_frames/*.png` is bit-identical to
the transmitted frame on every protected FG pixel.

The measurement was therefore comparing a frame against itself. It carries no
information about restoring FG, and **must not** be read as "restoration never
helps FG" -- that conclusion would be flatly wrong and would kill a workstream
for a non-reason. This is the bound-before-believing rule paying for itself.

**What F6 actually requires.** New GPU work: run a restorer over the protected
FG region and compare against the transmitted FG. Note the target is real --
protected FG is *not* pristine, it is codec-damaged at starved QP. Measured
transmitted FG-PSNR across those 24 runs spans **18.80 dB (tennis, qp62) to
32.89 dB (pigs, qp43)**, so at the starved end there is substantial headroom for
a restorer to clean up codec artifacts.

Because FG is never deliberately degraded, the matched operator here is a
same-resolution restoration prior (NAFNet, or an SR model at scale 1), not the
downscale/SR pairing used for BG.

**Corollary worth noting.** For Real-ESRGAN, `composite_passthrough` is
redundant -- the restorer already passes non-degraded blocks through untouched.
It still matters for in-painters (ProPainter/E2FGVI), which regenerate whole
frames and discard the transmitted prior.

### F6, part 2 — the real experiment (NAFNet on codec-damaged FG)

**Bounds, stated before measuring.** NAFNet is a GoPro *motion-deblur* model,
out-of-domain for codec artifacts, so expect PSNR -1.0 to +0.5 dB (most likely
slightly negative), LPIPS -0.02 to +0.05, and 10-40% of FG superblocks favouring
restoration. Alarm at >+3 dB (too good for an out-of-domain model) or another
0%/100% fraction.

**Method.** Five runs spanning the FG-damage range, 24 frames each. NAFNet
(width 64, fp32) applied to the *transmitted* frames; measured on protected FG
pixels only (FG mask AND NOT degraded). `FG_gated` applies the encoder-side
gate: per 64x64 block, keep whichever of transmitted/restored is closer to the
reference.

| video | qp | FG_tx | FG_nafnet | delta | blocks won | FG_gated | gate gain |
|---|---|---|---|---|---|---|---|
| tennis | 62 | 18.38 | 18.35 | -0.03 | 35.1% | 18.41 | +0.03 |
| tennis | 58 | 21.38 | 21.28 | -0.10 | 25.3% | 21.39 | +0.01 |
| bear | 51 | 26.12 | 26.08 | -0.04 | 27.1% | 26.13 | +0.01 |
| camel | 62 | 24.62 | 24.53 | -0.09 | 40.5% | 24.67 | +0.05 |
| pigs | 43 | 33.29 | 33.16 | -0.12 | 24.7% | 33.30 | +0.01 |

**Verdict — three separate conclusions, and they point different ways.**

1. **NAFNet is the wrong FG restorer.** Net loss on all five runs (-0.03 to
   -0.12 dB). Unsurprising: it was trained to remove motion blur, and what
   damages protected FG is codec quantization.
2. **The gate mechanism is validated.** It turns a net-losing restorer into a
   net gain (+0.01 to +0.05 dB) *by construction* -- FG is never worse than
   transmitted, exactly the guarantee claimed for it. That property held on
   every run, as it must.
3. **But there is nothing here worth gating.** The gain is 10-50x below the
   0.5 dB JND. "Restore FG too" is **not** worth pursuing with any restorer
   currently wired.

**What is not killed.** 24.7-40.5% of FG blocks are ones where *some* restorer
beat the transmitted pixels -- so the headroom is not zero, the model is just
wrong. And the class of model that would fit is precisely the
**codec-conditioned** one (MoE-DiffIR et al.) from Goal 3, since codec artifacts
are exactly what it is trained on. F6 and the Q6 mechanism argument therefore
converge on the same recommendation. Revisit FG gating *only* after a
codec-conditioned restorer is wired; the gate itself needs no further work.

**Limitation.** Measured on mask-weighted PSNR over 24 frames/run, not on
`foreground.lpips_mean` / `dists_fg`. Under the hard rules that makes this a
screen, **not a citable FG claim**. It is sufficient for the negative
conclusion (the magnitudes are far below any JND), but a positive FG result
would have to be re-measured with the sanctioned metrics.

---

## F1 — is a true per-superblock bit map better than the EVCA proxy?

**Question.** The selection score uses EVCA complexity as a stand-in for "how
many bits this block costs". If the true marginal bit cost ranks blocks
differently, a measured bit map would beat the proxy and Question A (build
`bitcost.py`) is worth days of work. If not, the numerator of the objective is
already solved and the direction closes.

**Bounds, stated before measuring.** Spearman rho between EVCA SC and true
marginal bits: 0.4-0.8 (EVCA is *designed* as a bitrate predictor). Selecting by
true bits should beat EVCA by 0-15% of bits at matched 25% budget. Alarm at
rho < 0 (anti-correlation would mean a bug) or >30% difference.

**Method.** Exact leave-one-superblock-out, all-intra so each measurement is
independent and no additivity assumption is needed. For each frame: encode it
(SVT-AV1, preset 8, QP 43, single frame), then re-encode 60 times with one 64x64
superblock mean-filled; the size difference is that superblock's exact marginal
bit cost. Compare the resulting map against EVCA SC pooled to the same grid.

**Result.**

| video | frames | rho | EVCA captures |
|---|---|---|---|
| camel | 8 | +0.938 (p=1.3e-221, n=480) | 99.4% |
| dancing | 3 | +0.958 | 99.1% |
| motorbike | 3 | +0.754 | 93.0% |

"EVCA captures" = bits freed by selecting the top-25% of superblocks by EVCA SC,
as a fraction of the bits freed by selecting the top-25% by *true* marginal cost.

**Verdict — direction CLOSED.** EVCA already recovers 93-99% of the achievable
bit saving. A perfect bit oracle would add at most 1-7%, and only two of the
three videos leave even that much on the table. **Do not build `bitcost.py`.**
That is days of work saved by an afternoon of encoding.

Two of three rho values landed *above* the predicted 0.4-0.8 band. That is not
an alarm but it does need explaining: the all-intra design measures *intra* cost,
and EVCA SC is a spatial-complexity measure, so in this regime the two are close
to measuring the same thing. The honest reading is that 93-99% is likely an
**upper** bound on EVCA's skill -- under inter coding, where temporal prediction
dominates and EVCA's TC term would carry the load, the proxy could rank less
well. The conclusion still holds for the intra-dominated starved-QP regime the
paper operates in, but a reviewer could fairly ask about inter, and the answer
is that this experiment does not cover it.

**Where this leaves Goal 1.** The numerator (bits) is solved; W0.2 showed the
denominator (damage after restoration) varies by **6.2 dB within a single run**.
The entire opportunity is in the denominator. This is the strongest single
argument for the plan's reframing, and it now rests on two independent
measurements rather than one.

---

## F5 — transform-aligned AC truncation vs Gaussian blur

**Prediction under test (O2).** Gaussian blur is a pixel-domain kernel that
straddles transform block boundaries, producing ringing the codec must still
encode; AC truncation writes exact zeros in the codec's own basis. So AC
truncation should be **cheaper in bits** at equal perceptual loss, i.e. it
should strictly dominate blur.

**Bounds, stated before measuring.** AC truncation better on LPIPS by 0.00-0.05
at matched rate, and/or 5-25% fewer bits at matched quality. Alarm if AC is
*worse* by >0.05 LPIPS (refutes the mechanism) or bits differ by >50%.

**Method.** camel, 30 frames, 640x360. Same 25% superblock selection (top-k by
EVCA) fed to both operators. Blur: `cv2.GaussianBlur` k=15. AC truncation:
per-channel DCT in YCrCb on the codec's own 8x8 grid, keeping only the top-left
2x2 coefficients. Both encoded SVT-AV1 preset 8 across QP 39/41/43/45/47, then
compared on a common bitrate axis by log-rate interpolation over the
**overlapping** rate range -- not at hand-picked matched pairs.

**Result 1 — the mechanism prediction is REFUTED.** AC truncation costs
**+36% MORE bits than blur at equal QP** (154,488 vs 113,441 at QP 39; 88,741
vs 66,762 at QP 47). "Codec-aligned zeros are cheaper" did not hold. The
straightforward reason: keeping 2x2 of 8x8 coefficients still preserves
substantial content, while a 15-tap blur removes far more information. The two
operators were never matched in degradation strength, and the bit ordering is
dominated by that, not by transform alignment.

**Result 2 — at matched rate the metrics disagree, systematically.**

| arm | QP | bytes | LPIPS | DISTS |
|---|---|---|---|---|
| blur | 39 | 113,441 | 0.3029 | 0.1576 |
| blur | 43 | 86,223 | 0.3212 | 0.1668 |
| blur | 47 | 66,762 | 0.3384 | 0.1750 |
| actr | 39 | 154,488 | 0.2235 | 0.1568 |
| actr | 43 | 115,141 | 0.2445 | 0.1662 |
| actr | 47 | 88,741 | 0.2666 | 0.1756 |

Interpolated over the overlapping rate range [88,741, 113,441]:
- **LPIPS: AC better by -0.0530, at 100% of rate points** -- clears the 0.05 JND.
- **DISTS: AC worse by +0.0095, at 0% of rate points** -- sub-JND, but perfectly
  consistent in direction.

**Verdict — no win claimed.** The project's rules require LPIPS *and* DISTS for
a quality claim, and here they conflict in sign at every rate point. That is a
systematic disagreement, not noise, so cherry-picking LPIPS would be exactly the
failure the hard rules exist to prevent. Plausible reading: at matched rate AC
truncation preserves edge structure that LPIPS's AlexNet features reward, while
introducing 8x8 blocking that DISTS's texture term penalises -- but that is a
hypothesis, not a measurement.

**What this does and does not kill.** It kills *my stated mechanism* for O2
(codec-aligned zeros are not cheaper). It does not kill AC truncation as an
operator: a 0.053 LPIPS advantage holding at 100% of rate points is a real
signal worth understanding. The next step is not more blur-vs-AC at these
settings but (a) sweeping operator strength rather than only QP, so the two are
matched in degradation rather than only in rate, and (b) measuring
**post-restoration**, since the Goal-2 family is defined as (operator, prior)
pairs and this comparison is entirely pre-restoration.

**Limitations.** One video, 30 frames, one arbitrary strength setting per
operator (keep=2, k=15), pre-restoration, and no FG/BG split.
