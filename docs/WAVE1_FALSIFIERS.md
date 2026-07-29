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
| F4 | 2 | `--film-grain` on/off, selective | todo |
| **F5** | **2** | **Transform-aligned AC truncation vs Gaussian blur** | **DONE — my stated mechanism REFUTED; metrics disagree, no win claimed** |
| **F6** | **3** | **Encoder-side FG gate: does restoring FG clear JND?** | **DONE — gate mechanism validated, but no FG restorer worth gating yet** |
| F7 | 2 | Chroma-first degradation probe | todo |

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
