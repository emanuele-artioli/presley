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
| **F2** | **1** | **64x64-snapped vs scattered 16x16 at matched area** | **DONE — signaling-overhead hypothesis REFUTED for 16→64; the bs8 penalty is a cliff, not a gradient** |
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

## F2 — is the block-size penalty about signaling, or about which pixels?

**Question.** The existing ablation says small blocks cost bits. Pulled from the
results index at matched video / resolution / degradation / component / QP,
varying only `block_size`:

| run | bs8 | bs16 | bs24 | bs32 |
|---|---|---|---|---|
| bear 1920 downsample (high rate) | 1695556 | 1549119 | 1534283 | 1538121 |
| bear 1920 downsample (**starved**) | 571279 | 447603 | **428253** | 431274 |
| camel 1920 downsample (high rate) | 1675033 | 1483127 | 1456313 | 1451829 |
| camel 1920 downsample (**starved**) | 586483 | 446677 | 423451 | **421026** |

bs8 costs **+9-15% at high rate and +27-39% starved** — and starved is the
regime hard rule 8 says matters. That is a large effect sitting in data already
on disk.

**But it does not isolate the mechanism.** Changing `block_size` changes two
things at once: the granularity of the *signal*, and *which pixels get
degraded* (a finer grid tracks the score field better and selects a different,
more scattered set). The bits could be going to partition signaling and broken
prediction, or simply to a more fragmented degradation pattern. The plan's
choice of a 64x64 measurement grid rests on the first reading.

**The isolating test.** Hold the degraded **area exactly constant** and vary
only alignment:

- **scattered** — pick the top-N 16x16 blocks by removability score.
- **snapped** — score each 64x64 superblock as the mean of its sixteen 16x16
  members, then take whole superblocks until **exactly the same number of
  16x16 blocks** is degraded.

Same operator, same count of degraded blocks, same QP; the only difference is
whether the degraded region is superblock-aligned or fragmented. Area is matched
*by construction* rather than approximately: the snapped arm is chosen first,
its exact block count is read off, and the scattered arm is given that count.

**Method.** The n=8 probe suite at 640x360, `shrink_amount` 0.25 (the dominant
operating point — 430 of the indexed runs), operator `downsample` scale 0.5 via
`filter_frame_downsample` with an explicit `sel` mask, so this is the pipeline's
real operator and not a reimplementation. Scores are the cached
`removability_a0.50_b0.50.npy` at bs16, so both arms rank pixels with the same
score field. Fixed QP 43, preset 8, `tune` omitted — same as F3/F4/F7. The
64x64 superblock is a 4x4 group of 16x16 blocks; the two leftover block rows
(22 = 5*4 + 2) are handled as partial edge superblocks, exactly as AV1 pads and
crops internally.

**Bounds, stated before reading any number.**

- *Bits, snapped vs scattered, at matched area and QP.* Snapping should **save
  3-20%**. **Alarm if snapping COSTS more than 5%** (would invert the
  hypothesis) **or saves more than 40%** (too large to be signaling alone at
  equal degraded area, and more likely an area-match bug).
- *Quality.* Snapping should be **slightly worse**: it is forced to degrade
  whole superblocks, so it spends its area budget on lower-scoring pixels the
  scattered arm would have skipped. Expect dPSNR **-0.1 to -1.5 dB** and dLPIPS
  **+0.000 to +0.030**. **Alarm if snapping is better on bits AND better on
  LPIPS by enough to clear JND** — a free lunch on both axes at equal area
  means the selection or the area match is broken, not that we found something.
- *Area.* Identical by construction; asserted, not hoped for.

**Decision rule, fixed in advance.** The signaling-overhead reading survives
only if snapping saves bits with a significant sign test at n=8. If it does not,
the bs8 penalty is about *which pixels get degraded*, not about superblock
alignment, and the 64x64 grid loses this particular justification (it would
still keep the two independent ones: kvazaar's per-CTU stats being the only
per-block bit source here, and LPIPS's validated 64x64 patch size).

**Manipulation check, run before trusting any null.** A null is only worth
reporting if the arms really differ. Snapping cuts the degraded region's
boundary length by **2.9x** (213.0 -> 73.8 mean 4-connectivity boundary edges on
bear) and moves **23%** of the selected blocks (77.2% overlap). The intervention
does exactly what the hypothesis asks for.

**Result — snapping does not save bits.**

| video | scattered | snapped | dBits |
|---|---|---|---|
| motorbike | 83994 | 86861 | **+3.41%** |
| color-run | 181968 | 188094 | +3.37% |
| dogs-jump | 59023 | 60146 | +1.90% |
| bear | 155640 | 158587 | +1.89% |
| bike-packing | 184823 | 186962 | +1.16% |
| dancing | 350502 | 353371 | +0.82% |
| drift-turn | 186729 | 186792 | +0.03% |
| drift-straight | 185176 | 183772 | **-0.76%** |
| **mean** | | | **+1.48%** |

Snapping saves bits on **1 of 8** videos. The exact two-tailed sign test gives
p=0.0703 and the direction is not unanimous, so the ~1.5% *cost* is itself
**not** significant either -- the honest statement is that there is **no bit
saving from superblock alignment**, not that alignment is expensive.

Quality at the same QP and the same area, snapped vs scattered:

| metric | mean d | verdict |
|---|---|---|
| PSNR | +0.2554 (8/8 better) | `sub_jnd_significant` |
| LPIPS | -0.0128 (8/8 better) | `sub_jnd_significant` |
| DISTS | -0.0046 (7/8 better) | `no_consistent_direction` |

Both are consistent and significant but **below JND**, so under hard rule 2b
this is "a small, reproducible, imperceptible effect" and never a win.

**Verdict: the signaling-overhead hypothesis is REFUTED for 16 -> 64.** The
pre-registered rule was that it survives only if snapping saves bits at n=8. It
does not, despite a 2.9x reduction in fragmentation. So the bs8 penalty in the
ablation is **not** superblock alignment: alignment beyond 16x16 buys nothing.
Read together with the ablation table, the picture is a **cliff between 8 and
16, not a gradient up to 64** -- bs8 costs +27-39% starved while bs16, bs24 and
bs32 sit within 4% of each other. Whatever bs8 breaks (prediction, partition
signaling, or simply too many tiny regions), 16x16 is already past it.

**Both predictions in the bounds were wrong, and neither tripped an alarm.**
Snapping was predicted to save 3-20% and instead cost 1.48%; it was predicted to
be slightly *worse* in quality and is instead imperceptibly *better*. The
alarms were set at "costs >5%" and "better on bits AND above-JND better on
LPIPS", and neither fired, so this is a failed prediction rather than a
suspicious measurement -- which is the distinction the bounds exist to draw.

The quality direction has a plain mechanism: the scattered arm spends its whole
area budget on the highest-scoring blocks, which are the most textured, and 2x
downsampling destroys most there. The snapped arm is forced to spend part of its
budget on flatter neighbours inside the same superblock, where downsampling
costs almost nothing. At equal area that trade is imperceptibly favourable.

**Consequence for the plan, stated plainly.** The 64x64 measurement grid loses
*this* justification -- there is no bitrate argument for it over 16x16. Its two
independent justifications stand: kvazaar's `--stats-file-prefix` per-CTU stats
are the only per-block bit source in this environment, and 64x64 is LPIPS's
validated BAPPS patch size. The finding is also mildly good news for selection:
**16x16 selection is free relative to 64x64**, so the finer grid can be used for
its better score tracking without paying for it in bits.

---

## S1 evaluation — does the graded downscale map actually help?

Not a falsifier of an existing claim -- a first look at whether S1's code
change (commit `c5c8af6`: `filter_frame_downsample` now supports
`downsample_levels`, backward-compatible at the default) does anything. The
code lands the *mechanism*; this is the first real experiment through it.

**Design.** n=8 probe suite, 640x360, block_size 16, `shrink_amount` 0.25,
`fg_protect`, fixed QP 43, svtav1, restorer Real-ESRGAN. Two arms per video,
**identical selection** (same blocks chosen by `select_removal_mask_global`,
independent of `downsample_levels`, so this isolates grading from selection
the same way F2 isolated alignment from selection):

- **binary** (`downsample_levels` omitted, i.e. 1) -- every selected block
  downsamples by factor 2, the historical behavior.
- **graded** (`downsample_levels: 3`) -- each selected block's own
  removability score quantizes into level 1-3, downsampling by 2/4/8x
  accordingly, restored through Real-ESRGAN's pyramid at 1-3 rounds per block.

**Bounds, stated before reading any number.**

- *Bits.* Grading pushes already-high-score blocks to a steeper downscale, so
  the graded arm should encode to **fewer bytes** than binary at the same QP
  and the same selected block set -- more information discarded, same rate
  control. Expect **3-20% additional saving**. **Alarm if the graded arm
  costs MORE bits than binary** (more aggressive downsampling should never
  cost bits at fixed QP with identical selection -- a positive delta points
  at a selection or area mismatch between arms, not a finding) or if the
  saving exceeds 40% (implausibly large for re-encoding roughly a quarter of
  the frame more aggressively).
- *Foreground quality.* FG blocks are excluded from selection identically in
  both arms, so FG-PSNR/LPIPS should be **near-flat, sub-JND** -- consistent
  with the alpha/beta finding, not the F6 tautology (this compares the SAME
  region across two DIFFERENT bitstreams, not restored-vs-transmitted within
  one, so a real if small rate-distortion-coupling delta is expected, not an
  exact zero).
- *Background quality.* Genuinely open. Two directions are both plausible:
  Real-ESRGAN's pyramid gets *more* rounds on graded's steeper blocks, which
  could recover comparably; or the coarser starting point could compound
  error across rounds and cost real quality. No pre-existing result bounds
  this either way -- W0.2 only established that damage-after-restoration
  varies 6.2-8.2 dB *within* a single run, which says the axis matters, not
  which direction grading moves it. Recorded as exploratory rather than
  bounded to a specific range; the one thing that would be an alarm rather
  than a genuine result is a BG-quality improvement large enough to clear
  JND with no corresponding bit cost anywhere -- a free win on both axes at
  matched selection would mean a measurement or pairing bug, the same
  free-lunch alarm used throughout Wave 1.

**Result 1 -- bits, and the pre-registered alarm fired.** Graded costs *more*
bits than binary on 7 of 8 videos (mean **+2.55%**), the opposite of what was
bounded (3-20% saving) and the exact alarm condition stated in advance.

| video | binary bps | graded bps | dBits |
|---|---|---|---|
| bear | 535734 | 574279 | **+7.19%** |
| drift-straight | 781828 | 829885 | +6.15% |
| motorbike | 440926 | 455857 | +3.39% |
| dogs-jump | 229181 | 232812 | +1.58% |
| bike-packing | 857600 | 867718 | +1.18% |
| dancing | 1232231 | 1244953 | +1.03% |
| color-run | 457394 | 458782 | +0.30% |
| drift-turn | 672843 | 670089 | -0.41% |

**Alarm investigated and closed as a genuine result, not a bug.** Two checks
before believing a "more bits from more compression" outcome: (a) selection is
identical between arms by construction -- confirmed directly from the saved
`strength_maps.npz`: bear binary has 18040 degraded blocks at level 1; bear
graded has **exactly** 18040 degraded blocks total, split 7590/10223/227
across levels 1/2/3 -- same budget, genuinely graded, not collapsed to one
level. (b) the downscale-factor direction is correct (level 3 -> `16//8=2` px
core), independently pinned by
`tests/test_degradation_downsample_levels.py`. With selection and mechanism
both verified, this is a real effect: **7/8 is directionally consistent but
underpowered** (exact two-tailed sign p=0.0703, same floor logic as every
other Wave-1 suite -- n=8 cannot clear alpha=0.05 on a 7/8 split), so the
honest statement is "consistent but not yet significant," not "confirmed."

**Result 2 -- quality moves the same direction as bits: worse, not better.**
Via `assess_metric`, family_size=1, n=8:

| region | metric | mean d | direction | verdict |
|---|---|---|---|---|
| FG | psnr | +0.0037 | 4+/4- | `no_consistent_direction` |
| FG | lpips | +0.0015 | 7+/1- | `no_consistent_direction` |
| FG | dists | +0.0024 | 7+/1- | `no_consistent_direction` |
| BG | psnr | **-0.9139** | 8/8 worse | **`perceptual_loss`** |
| BG | lpips (primary) | +0.0152 | 8/8 worse | **`sub_jnd_significant`** |
| BG | dists | +0.0058 | 8/8 worse | `sub_jnd_significant` |

FG is essentially untouched, as bounded (FG blocks are excluded from selection
identically in both arms). BG is unanimous 8/8 worse on **every** metric.
BG-PSNR clears JND (-0.91 dB against 0.5) but is a corroborating metric here
(hard rule 3 -- BG-LPIPS is primary), so it cannot carry the claim alone; the
primary metric's own verdict is `sub_jnd_significant`, mandated wording "a
small, reproducible, imperceptible effect" -- direction is worse, not better,
so read it as a small reproducible *cost*, never a win. Unlike F5, there is no
metric-disagreement problem: PSNR, LPIPS and DISTS all agree in sign.

**Verdict: naive grading (by the existing removability score alone) does not
help -- it costs bits and quality together, not one for the other.** This
kills the specific hypothesis "push already-high-score blocks to a steeper
downscale" as a standalone improvement. The registered "free lunch" alarm
(BG-quality win with no bit cost) never had a chance to fire; what happened
instead is closer to the opposite failure mode.

**A mechanism, consistent with F2 and F4.** A block downsampled to a 2x2 core
and linearly stretched back to 16x16 (level 3) creates a much larger internal
discontinuity against its untouched, sharp neighbors than a level-1 block
does. Wave 1 has now surfaced this family of cost twice: F2 showed the codec
pays for boundary mismatch when the *selected footprint* fragments; this looks
like the same mechanism triggered by *within-block* severity instead -- the
encoder spends bits reproducing a sharper transition than the content
actually needs, and Real-ESRGAN's per-block SR (validated in this project only
against level-1 inputs, since this pyramid path had never run before S1) has
no more information to recover at level 3 than at level 1, so quality does not
compensate for the extra spend either.

**This does not retire the `downsample_levels` mechanism, only naive use of
it.** The code (commit `c5c8af6`) is correct and byte-identical at the
default; what failed is grading purely on the existing bits-cost proxy (F1's
numerator) with no regard for restorability (W0.2's denominator, which is the
axis this session's own diagnostic work identified as the real opportunity).
The natural next step is a level assignment informed by *predicted damage*,
not just score magnitude -- i.e. reserve aggressive levels for blocks that
tolerate them, rather than pushing every high-score block equally hard.

Full data: 16 result hashes listed in commit history; `strength_maps.npz`
under each `results/<hash>/` for the exact per-block level maps.

---

## S1b — damage-aware grading: test the CEILING before building a predictor

**Scoped 2026-07-30, not yet run.** Direct follow-up to the S1 result above.

**Question.** S1 showed that grading by the existing removability score fails.
That score is F1's *numerator* (how many bits a block costs) and contains
nothing about W0.2's *denominator* (how well a block comes back after
restoration). So the open question is not "does grading work" but "was it
graded on the wrong quantity". Before building any damage predictor, test
whether a **perfect** one would even help -- the same oracle-ceiling move that
killed O3 in F7 for the cost of an afternoon.

**Why a ceiling test is the right shape.** A damage predictor is real work
(S2's structure-tensor coherence, or a learned model). If an oracle that
*already knows* each block's true damage cannot beat plain binary, the graded
direction is dead outright and no predictor is worth building. If the oracle
does win, the size of its win becomes the budget for how good a cheap proxy
has to be -- exactly the framing F7's 0.174-LPIPS gap provided.

**Design -- hold the level histogram fixed, vary only the assignment.**
The S1 result already confounds two things (grading changes both *how many*
blocks sit at each level and *which* ones). Isolate the second, the way F2
isolated alignment by matching degraded area by construction:

- **Arm C (binary)** -- already run, the 8 `levels=1` hashes above.
- **Arm A (naive graded)** -- already run, the 8 `levels=3` hashes above.
  Level assigned by score magnitude.
- **Arm B (oracle damage-aware)** -- NEW. Takes **Arm A's exact per-video
  level histogram** (e.g. bear's 7590/10223/227 across levels 1/2/3) and
  reassigns which blocks get which level, sorting by *measured damage
  tolerance* instead of by score: blocks that suffer least at a steep level
  get the steep levels. Same block count per level, so the bit spend is held
  roughly fixed by construction and only the assignment varies.

**Measuring the damage input.** `tools/mine_block_damage.py` already emits
exactly the needed quantity -- per-64x64-SB `delta_psnr`, damage attributable
to degrade->restore over and above what the codec did at that QP -- and the S1
runs already carry its inputs (`block_mse.npz` / `block_psnr.npz` are present).
Per-block damage at each level comes from **uniform-level probe runs**: all
selected blocks at level 2, and all at level 3 (level 1 is Arm C, already on
disk). That gives damage_b(k) for k in 1..3 per superblock.

**Blocker to clear first, found while scoping:** the miner joins each restored
run to a *matched pristine baseline* (same video/resolution/codec/QP), and at
640x360 svtav1 QP43 only **bear** has one (`a07560c409dc38ce`). The other seven
probe videos have none. Seven `component: baselines` runs must be added first
-- these are plain encodes with no restoration, so they are cheap, but the
mining silently yields nothing without them.

**Cost.** 7 pristine baselines + 16 uniform-level probe runs (2 levels x 8
videos) + 8 Arm-B confirmation runs = **31 runs**. The 16-run S1 batch took
roughly 30-40 minutes wall clock including evaluation, so budget ~1-1.5 h GPU
plus offline mining/allocation. Note LPIPS/DISTS are **not** computed by
default (S1's runs came back PSNR/SSIM-only and needed a backfill pass) --
either run the backfill afterwards or the perceptual verdicts will be missing.

**Bounds, to be registered before reading any number** (drafted here, confirm
at run time):

- *Damage vs level.* Mean `delta_psnr` should increase monotonically with
  level -- level 3 discards 16x more pixels than level 1. **Alarm if level 3
  is not clearly worse than level 1 on average**, which would mean the level
  is not reaching the pixels and the whole probe is measuring nothing (the
  same silent-no-op failure `film-grain=1` produced in F4).
- *Damage spread within a level.* W0.2 measured a 6.2-8.2 dB p90-p10 spread at
  level 1. Expect a comparable or larger spread at levels 2-3. **If the spread
  is small, there is nothing for a damage-aware assignment to exploit and the
  direction dies right there** -- this is the cheapest possible kill and it
  needs no Arm-B run at all, so check it before spending the 8 runs.
- *Arm B vs Arm A (matched histogram).* Expect BG-LPIPS to improve by
  0.005-0.04 if damage tolerance is a real, exploitable axis. **Alarm above
  0.08** -- that would exceed the entire binary-vs-naive gap and points at a
  histogram or pairing mismatch rather than a finding.
- *Arm B vs Arm C (the decision).* The real question.

**Decision rule, fixed in advance.** The graded direction survives only if
oracle damage-aware grading beats plain binary on **either** axis: BG-LPIPS
clearing JND at matched bits, **or** a bitrate saving at sub-JND-equal
quality (a Goal-1 win). If the oracle achieves neither, grading is closed for
good -- not merely "graded on the wrong quantity" -- and S2's structure-tensor
proxy should not be built, because its ceiling would already be known to be
worthless. Per hard rule 2b a sub-JND quality gain is **not** a win, so a
consistent-but-imperceptible Arm-B improvement counts as a failure of this
decision rule, not a partial success.

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

## O2 re-test — sweep operator STRENGTH, measure POST-restoration

**Registered before any run. Bounds committed before launching.**

F5 left O2 unsettled, not closed: it refuted the *stated mechanism*
("codec-aligned zeros are cheaper in bits" — they are not, AC truncation cost
+36% MORE bits at equal QP), but the residual signal it left behind is real —
a 0.053 LPIPS advantage for AC truncation holding at 100% of rate points,
which DISTS contradicts in sign at 100% of rate points. F5 could not settle
that because it swept **QP only** (so the two operators were matched in rate
but never in degradation strength) and measured **pre-restoration** (so it
could say nothing about a Goal-2 family that is defined as (operator, prior)
pairs). This re-test removes exactly those two limitations.

### What changed in the code

`filter_frame_ac_truncate` (`src/presley/degradation.py`) makes F5's throwaway
script a first-class pipeline operator: per-channel DCT in YCrCb on the codec's
own 8x8 grid, keeping the top-left `keep` x `keep` coefficients, with the same
`sel` budget contract as blur/downsample and a binary strength map. `keep` is
the strength knob (`keep=8` is a no-op, `keep=1` is DC-only). `presley_ai` now
also reads **`blur_kernel`** from the experiment — it never did, which is the
mechanical reason F5 could only ever test one blur strength; `roi.py` has read
it all along. Both defaults (`blur_kernel: 15`, `ac_keep: 2`) reproduce the
previous hardcoded behavior, so no existing experiment hash or output moves.

NAFNet is admitted for `ac_truncate` in `RESTORER_DEGRADATIONS`: it is a
conditioned restorer doing one full-frame forward and pasting untouched blocks
back, so it reads the map only as "was this block degraded" — the same binary
units blur emits. **The same prior on both operators is the point**: the
operator is then the only variable.

### The strength ladders (measured on bear, 5 frames, all blocks degraded)

| blur `k` | degraded PSNR | | `ac_keep` | degraded PSNR |
|---|---|---|---|---|
| 7  | 22.83 dB | | 4 | 24.81 dB |
| 15 | 21.12 dB | | 2 | 21.68 dB |
| 31 | 19.78 dB | | 1 | 19.72 dB |

Three rungs each, chosen so the two ladders span an overlapping degradation
range (blur 19.8–22.8 dB, AC 19.7–24.8 dB) with the middle rungs nearly
matched. This is the axis F5 never had.

### Design

96 runs: **8 pre-registered probe videos** (motorbike, drift-straight,
drift-turn, color-run, dancing, dogs-jump, bike-packing, bear —
`tools/select_probe_videos.py -k 8`; n=8 is the smallest n whose exact
two-tailed sign test can reach p=0.0078 and survive Holm) x **6 operator
rungs** x **2 restorers** (`none` = the pre-restoration control at matched
strength, `nafnet` = the post-restoration measurement). 640x360, block_size 16,
`shrink_amount: 0.25` with `fg_protect`, svtav1 preset 8 **fixed QP 43**
(never VBR). Comparison is BG-LPIPS/BG-DISTS of the restored output vs the
ORIGINAL, on a common bitrate axis by log-rate interpolation over the
overlapping rate range — not hand-picked matched pairs (same procedure as F5,
so the two are comparable). LPIPS/DISTS are backfilled
(`presley.evaluation.backfill`); runs return PSNR/SSIM-only.

### Bounds, stated before measuring

| # | Measurement | Plausible range | Basis | ALARM outside |
|---|---|---|---|---|
| M1 | bits: AC vs blur at the matched middle rung | −10% … +30% | F5 saw +36% at *unmatched* strength; matching should shrink it | [−40%, +60%] |
| M2 | pre-restoration BG-LPIPS, AC − blur, matched rate | −0.080 … +0.020 | F5's −0.053 on one video, now over 8 | \|Δ\| > 0.15 |
| M3 | pre-restoration BG-DISTS, AC − blur | −0.010 … +0.030 | F5's +0.0095, sub-JND, 100% consistent | outside [−0.05, +0.08] |
| M4 | **post-restoration BG-LPIPS, AC − blur** | −0.040 … +0.040 | restoration compresses operator differences; NAFNet's training favours blur | \|Δ\| > 0.12 |
| M5 | **post-restoration BG-DISTS, AC − blur** | −0.020 … +0.030 | as M3, damped by restoration | \|Δ\| > 0.06 |
| M6 | sign consistency across the 8 videos | 2/8 … 8/8 | anything is possible at n=8 | 8/8 in the direction *opposite* to the mean |
| M7 | NAFNet restoration gain (BG-LPIPS, transmitted − restored) | −0.020 … +0.060 | NAFNet is a weak CNN gauge, not a strong prior | gain > 0.15 or < −0.10 |
| M8 | `invariant_failures` | empty, all 96 | — | any non-empty run (non-citable) |

### Decision rule, fixed in advance

**O2 survives** only if, post-restoration at matched rate, AC truncation beats
blur on **both** LPIPS and DISTS, with a consistent sign across videos and a
suite verdict of at least `sub_jnd_significant`, **and** costs no more than
+10% bits. Anything else — the two metrics still disagreeing in sign, AC losing
on either metric, or the advantage evaporating after restoration — **closes
O2**, on the grounds that a (operator, prior) pair that cannot beat the
incumbent operator with the same prior is not a Goal-2 family member.
