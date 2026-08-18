# Pre-registration — overnight wave, 2026-08-17

Written **before** any run in this wave executed. Bounds and alarms are stated
first, per `research-log/hard-rules.md`. A result outside its band is an alarm:
investigate implementation/eval/data before reporting it, and never cite it until
the alarm is closed or the band is revised **with a stated reason**.

Wave plan and rationale: `~/.claude/plans/if-you-can-t-see-transient-pine.md`,
Part F. Every experiment here needs **no code change** — that is the selection
criterion for this wave.

---

## W1-A — Breadth on a common four-rung ladder (plan F13, absorbs F10)

**Why.** `fig:breadth` currently plots a *matched-QP bitrate delta* and reports it
as a saving. A saving is only a saving at matched quality, which needs a BD-rate,
which needs a real ladder. The ELVIS arm has two rungs (QP 32/37) and the PRESLEY
arm four (32/37/42/47) on a partly different clip set, so the figure also compares
unequal numbers of clips.

**Design.** Harmonize both arms onto **QP {32, 37, 42, 47}** — the ladder the
PRESLEY arm already uses, so no new rung is invented — over the union of the
breadth clips, plus the matching pristine baselines. Only missing cells are run;
`presley-run` skips any hash that already has a `result.json`.

Fixed: `x265`, `preset medium`, 640×360, `block_size 8`, `alpha=beta=0.5`,
`shrink_amount 0.25`, `fg_protect true`, `composite_output true`.
ELVIS arm `removal_mode blackout`, `inpainter none` (transport-only, as landed).
PRESLEY arm `degradation downsample`, `restorer realesrgan`.

**Bounds.**

| quantity | plausible band | basis |
|---|---|---|
| ELVIS-arm BD-rate vs baseline, per clip | −60% … +60% | the existing 2-rung deltas span −80%…+45%; BD-rate integrates so should be tighter |
| PRESLEY-arm BD-rate vs baseline, per clip | −60% … +60% | same |
| fraction of clips saving bits at matched quality | 0.30 … 0.80 | the matched-QP version is 22/33 ≈ 0.67; matched-quality should be **lower**, since some of that saving was bought with quality |
| FG-LPIPS delta, either arm | ≤ 0.05 on ≥ 80% of clips | `fg_protect` is on; the existing breadth run holds under 0.02 on most clips |

**Alarms.**
- Any clip whose BD-rate is outside ±60% → check ladder overlap before reporting;
  a non-overlapping ladder produces meaningless extrapolated BD-rate.
- **Overlap fraction < 0.50 on any clip → that clip is not reportable**, same gate
  `tools/analyze_ratematched_n13.py` applies.
- Matched-quality saving fraction **higher** than the matched-QP 0.67 → alarm.
  Holding quality fixed should cost, not gain; if it gains, suspect the BD
  direction convention.
- Any run with non-empty `invariant_failures` is dropped and re-run, never
  reported with a disclosure.

**What it settles.** Whether "the bridge saves bits on most clips" survives being
asked at equal quality. It may not — that is an acceptable outcome and would
replace a wrong claim with a correct one.

---

## W1-B — Graded λ at a block size that can carry it (plan F7, closes D3)

**Why.** `CLAIM(tab:graded)` retired graded multi-level downsampling, but it was
measured at `block_size 16`, where the method section's own bound
λ ≤ log2(b/8) admits **one** level — so its deepest rung carried each block at
2×2 samples. The retirement was measured where grading cannot work.

**Design.** `block_size 64` at 1920×1080, where the bound admits three levels, so
the deepest rung leaves 8×8 samples. Two arms differing **only** in
`downsample_levels` (absent = binary, 3 = graded), on the eight probe clips, four
fixed-QP rungs, plus pristine baselines. Fixed: `svtav1 preset 8`,
`shrink_amount 0.25`, `fg_protect`, `realesrgan`, `composite_output`.

**Bounds.**

| quantity | plausible band | basis |
|---|---|---|
| graded-vs-binary BD-rate on BG-LPIPS | −25% … +25% | at b=16 grading cost bits on 7/8; at b=64 the degeneracy is removed, so the honest prior is centred on zero |
| clips where graded beats binary | 2 … 6 of 8 | anything at the extremes is suspicious at this n |
| FG-LPIPS delta between arms | ≤ 0.02 | both arms hard-exclude the foreground, so the FG must be near-identical |

**Alarms.**
- FG differs by more than 0.02 between arms → the exclusion is not holding and the
  comparison is invalid.
- Graded better on 8/8 → too clean for a knob that failed at b=16; check that
  `downsample_levels` actually took effect (the level map should have ≥3 distinct
  non-zero values).
- Either arm's realized removal rate departing from the 0.25 budget → the level
  floor is misbehaving.

**What it settles.** Whether the graded transport is genuinely retired or was
retired out of regime. **Either outcome is publishable**, and the current
`sec:ablation` wording is unsafe until one of them exists.

---

## Reporting rule for this wave

Numbers enter the paper only through a `CLAIM(anchor)` with `src=` hashes and an
empty `invariant_failures` on every cited run. Nothing from this wave may be
worded as a win without passing `presley-compare` (JND gate) and, for N>1, the
suite layer with its two-tailed test and Holm correction over candidates tried.

---

## W1-C — Placement control, 2×2 factorial (plan F1; F2 retired)

Added 2026-08-17, **before any W1-C run executed**. F2 ("matched-footprint
uniform") is retired: uniformity means covering the whole frame, so constraining
it to 25% coverage makes it a different placement of the same budget rather than
a uniform arm, and at λ=1 there is no milder per-block strength to match with.

**Why.** PRESLEY asserts that *placement* matters — it ranks blocks by
removability and hard-excludes the foreground — and the article contains no
measurement of either. `f1-oracle-bits` measures the ranking against a *bit*
oracle (a proxy, not delivered quality); `sec:downsample-vs-uniform` varies
placement and coverage together and isolates neither.

**Design.** Budget (0.25) and strength (λ=1) fixed. Two binary factors:

| arm | `selection` | `fg_protect` | contrast against A isolates |
|---|---|---|---|
| A | score | true | incumbent |
| B | random | true | **the ranking** |
| C | score | false | **the exclusion** |
| D | random | false | both; a floor for the pair |

6 videos × 4 rungs × 4 arms, `svtav1 preset 8`, 640×360, `block_size 8`,
`downsample` + `realesrgan`, `composite_output`, seed fixed at 1.

**Validity condition, enforced in code and tested.** The random map is
substituted *before* the clustering blur, so arms B and D get contiguous patches
like the incumbent. An unblurred random map scatters into singletons, which this
project already measures as expensive to code — a random arm without the blur
would lose on **fragmentation** rather than on placement and would answer a
different question. `tests/test_random_selection_control.py` asserts this, plus
equal budget across arms and that the exclusion still binds on the random arm.

**Bounds.**

| quantity | plausible band | basis |
|---|---|---|
| A vs B, BD-rate on BG-LPIPS | −20% … +5% | the score captures 0.833 of oracle bits against a 0.402 random null, so score placement should help; but the whole cost-axis headroom is only ~5% of bitrate, so a large win is not expected |
| A vs C, FG-LPIPS delta | **≥ 0.03 on most videos** | dropping the exclusion degraded 12.8% of foreground blocks on `bmx-trees` and cost 3 dB FG-PSNR |
| A vs D | worst of the four on FG | both protections removed |
| realized removal rate, every arm | 0.25 ± 0.01 | budget is matched by construction |

**Alarms.**
- **B beating A** on background BD-rate → the ranking is worse than chance;
  check the blur is applied to the random map and that the seed varies per frame.
- **A vs C showing no foreground difference** → the exclusion is not binding;
  invalidates the arm, not the design.
- Removal rate departing from 0.25 on any arm → budgets are not matched and no
  contrast is interpretable.
- Any arm's FG-LPIPS better than A's → suspect the evaluation is inheriting the
  method's mask (hard rule 7).

**What it settles.** Whether the two placement choices the architecture is built
on do measurable work at fixed budget and strength. A null on the ranking is a
publishable result and would say the budget, not the ordering, is what matters.

---

# Alarm log — fired 2026-08-18, before any result was read

## A1. W1-B: the graded arm numerically diverges on `camel`, all four rungs

`camel` at `downsample_levels: 3` fails the restoration-clipping invariant on
every rung (1.28–1.31% of output pixels driven to 0/255 against 0.29–0.34% in the
input it received; the limit is 0.50%). Its **binary** arm is clean on all four
rungs, and **every other video is clean on both arms** — 7 of 8 videos usable,
1 of 8 broken and only in the graded arm.

| video | binary ok/n | graded ok/n |
|---|---|---|
| bear, bike-packing, color-run, dancing, dogs-jump, drift-turn, motorbike | 4/4 | 4/4 |
| **camel** | 4/4 | **0/4** |

**This is not a missing datapoint, it is a property of the arm under test.** The
graded pyramid applies the restorer once per level, so each round re-amplifies
whatever the previous round pushed toward the rails; the binary arm applies it
once and does not. That the divergence appears *only* in the graded arm, on all
four of its rungs, and on none of the seven other clips, is consistent with
amplification rather than with a bad encode or a corrupt input.

**Consequences, in order.**

1. The graded-vs-binary comparison has **n=7 paired videos**, above the n≥6 floor,
   so it can still reach significance (minimum two-tailed p at n=7 is 0.0156).
2. **`camel` must not be silently dropped.** Numerical instability confined to the
   graded arm is evidence about grading, and reporting "7 of 8" without saying why
   would hide the most interesting thing the experiment found.
3. Before the comparison is cited, diagnose the clipping in the multi-round path.
   If it is fixable, re-run `camel` and report n=8. If it is intrinsic to applying
   a GAN restorer repeatedly, that is a **finding about λ>1** and belongs in the
   text as one.

**Not yet done, and it gates the W1-B claim.**

## A2. W1-A: three breadth runs fail the same clipping invariant

`drift-straight` (×2) and `scooter-board` (×1) at 640×360 on the PRESLEY arm.
Isolated rather than systematic — 3 of 272. They are dropped from the affected
ladders, and the overlap gate is applied per clip as pre-registered; a ladder that
loses a rung to this must be checked for overlap ≥ 0.50 before its BD-rate is used.

## A3. W1-C: the ranking bound fired high, and the basis for it was wrong

Observed: arm A beats arm B (random ranking) on **5 of 6** videos, median contrast
**−52.1 pp** of BD-rate on BG-LPIPS. The pre-registered band was −20%…+5%, so this
is far outside it.

**The band's basis was the error, not the result.** It was derived from
`f1-oracle-bits`: "the entire remaining headroom on the cost axis is worth about
5% of the bitrate". That figure is about **bits freed at fixed coverage** — how
close the score gets to a bit oracle. This experiment measures **rate at matched
background quality**, which is a different axis. Random placement degrades blocks
that are perceptually expensive as well as cheap, so background quality collapses
at equal rate and the BD-rate integral blows up accordingly. A 5%-of-bits ceiling
does not bound a quality-matched rate comparison.

**Band revised to −70%…0%, with that reason.** The result stands; the prediction
was mis-derived. Recorded rather than quietly widened.

## A4. W1-C: the exclusion contrast is INVALID on this clip set — a null by construction

Observed: A vs C shows A **worse** on background BD-rate on 6/6 (median +12.7 pp,
expected — arm C may degrade more of the frame) and **no foreground difference at
all**: mean FG-LPIPS 0.2493 (A), 0.2423 (C), spanning 0.2417–0.2541 across all
four arms. That range is 0.012, far inside the 0.05 perceptual margin. The
pre-registered bound (A vs C FG-LPIPS delta ≥ 0.03 on most videos) is breached.

**Diagnosis: the exclusion never binds on these six clips.** Measuring directly
how many foreground blocks arm C actually degrades:

| video | FG blocks | FG degraded by arm C | share of C's selection that is FG |
|---|---|---|---|
| bear | 11.8% | **0.3%** | 0.1% |
| camel | 16.4% | **0.1%** | 0.1% |
| dog | 12.7% | **0.0%** | 0.0% |
| pigs | 32.3% | **0.1%** | 0.2% |
| bike-packing | 24.4% | 2.6% | 2.5% |
| dogs-jump | 8.2% | 14.5% | 4.7% |

On five of six clips the score-based top-25% contains essentially **no** foreground
blocks even with protection switched off, because the γ=10 background priority has
already pushed them out of contention. Hard exclusion therefore has nothing left to
remove, and A and C run the same selection.

**This is a null by construction, not a measurement.** It must not be reported as
"the exclusion buys nothing". What it does support, and this is sharper than either
alternative:

> The soft γ priority is sufficient on typical content; hard exclusion is insurance
> against the minority of clips whose foreground is complex enough to outrank the
> background.

**Fix: the clip set was chosen wrong for this contrast.** It needs clips that
exercise the exclusion — ones where the foreground score distribution overlaps the
background's. `bmx-trees` is the known example (FG mean score 0.127 against BG
0.113; a purely score-based top-k degraded 12.8% of its foreground and cost 3 dB of
FG-PSNR) and it is **not in the six**. Re-run arms A and C on a clip set selected by
that criterion before the exclusion is claimed either way.

**The A-vs-B ranking contrast is unaffected** — it does not depend on the exclusion
binding, and both arms carry `fg_protect: true`.

## A5. Gamma cannot replace hard exclusion — measured, not argued (answers the design question)

Sweeping γ over the clips whose foreground the score actually reaches, measuring
the share of foreground blocks a purely score-based top-25% degrades:

| video | γ=10 | γ=25 | γ=50 | γ=100 | γ=1000 |
|---|---|---|---|---|---|
| bmx-trees | 8.56% | 6.74% | 6.16% | 5.90% | **5.66%** |
| tennis | 11.46% | 9.09% | 8.41% | 8.02% | **7.71%** |
| dogs-jump | 14.54% | 11.98% | 11.22% | 10.93% | **10.61%** |

**γ asymptotes.** A hundredfold increase buys about three percentage points and the
curve is flat by γ=50. No value of γ drives foreground degradation to zero.

**Why it is structural rather than a tuning failure.** γ is a *scale* factor applied
before the clustering blur. The blur then mixes γ-boosted background scores into
foreground blocks at the mask boundary, and because both the boosted background and
the leaked foreground scale with γ together, the resulting **ranking converges to a
fixed limit**. A scale factor cannot express a constraint that the ranking must
never cross a region boundary; only a rank-independent exclusion can. Deriving γ
from α/β would not change this, because the leak is scale-invariant.

**Conclusion: keep both mechanisms, change no code.** The soft priority does the
work on typical content (5 of 6 W1-C clips see ~0% foreground degraded at γ=10) and
the hard exclusion is what makes the guarantee hold on the minority that need it.
This is the paragraph already in §3.2 and it is now measured.

**On "if the foreground does get degraded, maybe that is fine":** two existing
results say otherwise. Without exclusion `bmx-trees` loses **3 dB of FG-PSNR**, and
`sec:restorability` shows complex blocks are precisely the ones that come back
*worst* after restoration — so the blocks that would leak are the blocks least
likely to be recovered.

## A6. W1-B result: the graded retirement survives being measured in regime

| | |
|---|---|
| FG-LPIPS delta between arms | max **0.0026** (bound ≤0.02) — **passes**, exclusion holds, comparison valid |
| overlap | min **0.89** (gate ≥0.50) — **passes** |
| direction | graded **worse on 6 of 7** clean videos, median BD-rate **+8.2%** |
| significance | exact two-tailed **p = 0.125** — **underpowered**, not a null |
| band | ±25% breached by `bear` (+53.5%) and `camel` (+70.5%, excluded) |

**D3 is closed.** The graded transport was previously retired on a measurement taken
at `block_size 16`, where the λ bound admits one level and the deepest rung carried
2×2 samples. Re-measured at `block_size 64`/1080p, where the bound admits three, the
direction is the same. `sec:ablation`'s wording is no longer unsafe.

**But it is underpowered, and must be worded as such.** At n=7, 6/7 gives p=0.125;
only 7/7 would reach 0.0156. Two more clean videos would settle it.

**`camel` is not a dropped datapoint** — its graded arm breaches the clipping
invariant on all four rungs while its binary arm is clean, and binary breaches 0 of
32 runs against graded's 4 of 32. That asymmetry is evidence about the graded
transport (each pyramid round re-amplifies toward the rails), and it points the same
way as the BD-rate.

---

# Wave 2 — the two re-runs the alarms require. Registered 2026-08-18, before running.

## W2-A — Exclusion contrast on a clip set that exercises it (fixes A4)

**Why.** A4 voided the original exclusion contrast: on 5 of 6 clips the score-based
top-25% contained almost no foreground even with protection off, so arms A and C ran
the same selection.

**Clip selection is on a pre-outcome criterion**, declared here before any run: rank
the corpus by the share of foreground blocks a purely score-based top-25% degrades,
and take the top six. Measured:

| clip | FG degraded, protection off |
|---|---|
| drift-chicane | 16.07% |
| lindy-hop | 14.55% |
| dogs-jump | 14.54% |
| tennis | 11.46% |
| dogs-scale | 9.52% |
| bmx-trees | 8.56% |

This selects on *whether the mechanism is exercised*, never on the outcome. Clips
where the exclusion cannot bind cannot inform a claim about it in either direction.

**Design.** Arms A (`score`+`fg_protect`) and C (`score`, no protect) only — the
ranking contrast is already settled by W1-C and does not need repeating. 6 clips × 4
rungs × 2 arms + baselines, svtav1, 640×360, bs8, downsample+realesrgan.

**Bounds.** FG-LPIPS delta A vs C ≥ **0.03** on ≥4 of 6 clips (the 3 dB FG-PSNR loss
on `bmx-trees` is the basis); C cheaper on background BD-rate on most clips (it
degrades more of the frame — expected, and the cost is meant to show in FG).

**Alarm.** If FG still does not separate on clips selected *for* binding, the
exclusion does not buy measurable quality and the article must say so. That is a
real possible outcome and is why the criterion was fixed in advance.

## W2-B — W1-B to n≥9 (fixes A6's underpowered verdict)

**Why.** Graded is worse on 6 of 7, but at n=7 that is p=0.125. Only 7/7 reaches
0.0156 at this n, so the direction cannot be claimed.

**Design.** Three further 1080p clips (`tennis`, `pigs`, `breakdance`) at
`block_size 64`, 4 rungs, both arms + baselines — same configuration as W1-B.
Expected 9–10 clean videos after `camel`'s exclusion.

**Bounds.** Graded worse on ≥7 of 9; FG-LPIPS delta between arms ≤0.02 (the validity
gate — max observed so far is 0.0026); overlap ≥0.50 per clip.

**Alarm.** Any new clip whose graded arm breaches the clipping invariant joins
`camel` as evidence about the graded transport rather than being dropped. Two or
more such clips would make instability, not BD-rate, the headline finding.

**Stopping rule, fixed now:** n is set by these three clips. We do not add clips
until the sign test crosses a threshold.

## A7. W1-A: bounds fired, and the investigation found a METRIC ARTEFACT, not a result

Observed at matched quality (BD-rate on FG-LPIPS over the new four-rung ladders):

| arm | saves bits at matched QUALITY | median BD-rate | for contrast, at matched QP |
|---|---|---|---|
| ELVIS (blackout, `inpainter: none`) | **1/24** | **+81.7%** | 23/24 "save" |
| PRESLEY (downsample+realesrgan) | 5/22 | +12.3% | 19/22 "save" |

Both breach the pre-registered band (0.30–0.80 of clips saving). Per the alarm rule
the numbers were investigated before being reported, and **the ELVIS row does not
survive that investigation.**

**The two foreground metrics disagree, and only for the arm with holes.**

| arm | median ΔFG-PSNR | beyond the 0.5 dB margin | median ΔFG-LPIPS | beyond the 0.05 margin |
|---|---|---|---|---|
| ELVIS | **−0.10 dB** | **10/104 (10%)** | **+0.0686** | **72/104 (69%)** |
| PRESLEY | — | — | +0.0047 | **0/101 (0%)** |

By PSNR the ELVIS foreground is essentially untouched; by LPIPS it is damaged on
nearly every point. Those cannot both describe the same pixels, and PSNR is the one
computed exactly on the mask.

**Diagnosis: LPIPS bleeds across the mask boundary.** LPIPS compares deep features
whose receptive fields span tens of pixels, so a *black hole* adjacent to a
foreground block changes the foreground's feature map even though the foreground
pixels are unchanged — which is exactly what the flat FG-PSNR says. The control is
the PRESLEY arm: downsampling leaves content in every block, there are no holes, and
its FG-LPIPS delta is 0.0047 with **zero** points beyond the margin. Holes bleed;
degraded-but-present content does not.

**Consequences.**

1. **The ELVIS-arm BD-rate on FG-LPIPS is not usable**, and neither is any
   FG-LPIPS claim on a *transport-only* hole arm. This affects `fig:breadth`'s ELVIS
   panel and any other mask-restricted-LPIPS number taken on an un-inpainted frame.
   It does **not** affect arms evaluated after in-painting, where the holes are
   filled before the metric sees them.
2. **The original matched-QP framing was not simply wrong.** The project rule already
   says that at matched QP, with foreground quality verified indistinguishable, the
   question is "who encodes fewest bits at indistinguishable FG quality". For the
   PRESLEY arm that verification now **passes at every one of 101 points**, so its
   matched-QP saving on 19/22 clips is legitimate and does not need a BD-rate.
   For the ELVIS arm the verification must be done on **FG-PSNR**, where it passes on
   90% of points.
3. **What the crosses in `fig:breadth` were marking is real** but was attributed to
   the wrong cause: they flag LPIPS bleed on hole arms, not foreground damage.

**This supersedes the plan's E1j-2**, which assumed the fix was to redraw everything
as a BD-rate. The correct fix is per-arm: verify indistinguishability on a metric
that does not bleed, then report matched-QP rate where it passes.

---

# F11 result — TOST does NOT fire; regime stability stays unproven either way

`analyze_r1_regime_scope.py` was **broken**, not blocked: it imported `ARMS` from
`analyze_ratematched_n13` but matched with a naive `cfg.get(k) != v`, which cannot
handle the `restorer_params` whitelist tuple or the `selection_rule` default absent
from older configs. Every `presley_ai` arm was therefore rejected and the tool
reported "no ladder has all three arms" on a tree that has them. Patched to use the
sibling's `_mismatch`/`_DEFAULTS`.

**With that fixed, `CLAIM(sec:regime-stability)` reproduces exactly** — 5/13
negative, p=0.581, median |contrast| 0.0223. The claim was always sound; the tool
had regressed. The provenance worry raised earlier is closed.

**TOST on the 13 contrasts: `not_shown_equivalent` at margins 0.05 and 0.03.**
The 90% CI on the mean is [−0.0168, +0.0128], inside ±0.05 — but two ladders exceed
the margin outright (`camel` −0.0585, `youtube_vos/0e4068b53f` −0.0535), so max
|delta| is 0.0585 and equivalence is not established.

**This refutes the fix proposed in plan E1i-1.** I predicted TOST would plausibly
fire because the *median* |contrast| is 0.0223 against a 0.05 margin. The median was
the wrong statistic: equivalence needs the whole distribution inside the bound, and
two ladders are not.

**Consequence for the text: none, and that is the point.** The §4.2 edit already
rests the operating-range claim on the primary result — the background win is
present on 13 of 13 ladders spanning both ends of the range — and reports p=0.58 as
a null rather than as evidence of stability. That wording is now confirmed correct
by a second test. What must NOT be written is "restoration is regime-stable" as a
demonstrated equivalence; the honest phrase is that no dependence is detectable and
none is ruled out.

---

# F14 — what predicts the relocation sign. Registered 2026-08-18, before computing.

**Why.** Bit relocation succeeds on some content and fails on others, and five
earlier attempts to explain the split all failed. Foreground *area* was the most
recent, and its failure is informative: area is a geometric proxy for bits, so the
sixth attempt measures bits and complexity directly instead of another geometry.

**Outcome variable.** The PRESLEY arm's **matched-QP bitrate delta** against the
pristine baseline, averaged over the four rungs, at 640×360/x265. Matched QP rather
than BD-rate because A7 established that this arm's foreground is indistinguishable
from baseline at **0 of 101** operating points — which is precisely the condition
that makes a matched-QP rate comparison the sanctioned analysis. The ELVIS arm is
excluded: its FG-LPIPS carries the mask-bleed artefact.

**Candidates — the whole family, declared now.**

| # | candidate | mechanism |
|---|---|---|
| 1 | background share of total EVCA complexity | if the background carries few bits, degrading it cannot free many, whatever its area |
| 2 | FG/BG mean-complexity ratio | relocation should pay when the background is complex *relative to* the foreground |
| 3 | baseline bits per pixel at the reference rung | the starved-bitrate rule says generative transports pay only when the codec is starved |
| 4 | mean temporal complexity | motion governs how much inter prediction already removes before we degrade anything |
| 5 | spatial-complexity dispersion (CV) | a flat score map gives selection nothing to choose between |

**Statistic.** Spearman rank correlation between each candidate and the bitrate
delta across clips, **Holm-corrected over all five**, losers reported in full.

**Bounds.**

| quantity | plausible band | basis |
|---|---|---|
| best candidate's \|rho\| | 0.0 … 0.8 | five prior attempts found nothing; a very high rho would be suspicious at this n |
| number of candidates significant after Holm | 0 … 2 | if three or more fire, suspect they are the same quantity in different clothes |

**Alarms.**
- \|rho\| > 0.8 on any candidate → check for circularity: the candidate must be
  computable from the *source* clip alone, never from anything downstream of the
  degradation or the encode.
- Candidates 1 and 2 both firing → they share the EVCA complexity term; report as
  one finding, not two.
- Any candidate significant *before* Holm but not after → report as a loser, and do
  not narrate it as "trending".

**Pre-declared negative outcome.** If nothing survives Holm, that is the result:
*relocation is content-dependent and six families of explanation have now failed to
predict it.* That is publishable and is what the content-dependence subsection
(plan E1j-1) reports.

# F14 result — nothing valid predicts the split, and the outcome variable revealed something bigger

## F14 proper: 0 of 5 candidates survive

| candidate | rho | p raw | p Holm | verdict |
|---|---|---|---|---|
| background share of complexity | −0.032 | 0.886 | 0.886 | loser |
| FG/BG complexity ratio | +0.244 | 0.262 | 0.541 | loser |
| **baseline bits per pixel** | **−0.571** | **0.004** | **0.022** | **survives Holm, then FAILS validity** |
| mean temporal complexity | −0.321 | 0.135 | 0.541 | loser |
| spatial-complexity dispersion | +0.319 | 0.138 | 0.541 | loser |

**The one survivor is a shared-denominator artefact.** The outcome is
`presley_rate/baseline_rate − 1` and the candidate is `baseline_rate/pixels`, so
both contain the baseline rate. The pre-registered circularity rule (candidates must
come from the source clip alone, never from anything downstream of an encode) is
what caught it — the magnitude alarm at |rho|>0.8 did not fire.

Permutation control, breaking the real baseline/PRESLEY pairing while keeping both
marginals: **rho = −0.678 ± 0.092 under the null**, against **−0.583 observed**
(z = +1.03). *The null pairing produces a stronger correlation than the data do.*
The relationship is arithmetic, not empirical.

**So the pre-declared negative outcome stands: six families of explanation have now
failed.** The four source-only candidates lost cleanly. Relocation is content
dependent and nothing we have tried predicts which content.

## The bigger finding: the saving REVERSES across the rate ladder

Computing the outcome per rung rather than averaged, on 23 clips with all four rungs,
foreground indistinguishable throughout (median gap 0.0007–0.0047, all far inside the
0.05 margin, so a matched-QP rate comparison is valid at every rung):

| QP | saves bits | median delta | median FG-LPIPS gap |
|---|---|---|---|
| 32 (richest) | **19/23** | **−6.6%** | +0.0047 |
| 37 | 16/23 | −2.5% | +0.0028 |
| 42 | 5/23 | +10.4% | +0.0044 |
| 47 (most starved) | **1/23** | **+31.4%** | +0.0007 |

**This contradicts the project's starved-bitrate rule**, which states that generative
transports "only pay off where the codec is bit-starved" and that a comfortable-rate
result *understates* them. Measured across a real ladder, the opposite holds: the
saving is largest where the codec is comfortable and becomes a large penalty where it
is starved.

Plausible mechanism, not yet tested: at coarse quantization the encoder has already
discarded the high-frequency background detail that degradation exists to remove, so
there is nothing left to free — while the resampled block content and the edges at
degraded/undegraded boundaries are new signal the encoder must still code. The
side channel cannot explain it (4.19 kbps against a penalty of tens of percent).

**Consequences.**
1. The starved-bitrate rule in `research-log/hard-rules.md` is **contradicted by
   direct measurement** and must not keep steering experiment design until
   reconciled.
2. Any single-rung result inherits the rung's sign. `fig:breadth`'s original two-rung
   view sat at QP 32/37 — the two rungs where the saving exists.
3. This is *bit relocation* only. It does not touch `sec:regime-equivalence`, which
   is about restoration and is measured on background LPIPS.

## RETRACTION — the starved-bitrate rule is NOT contradicted

The previous entry claimed hard rule 8 was falsified. **That was wrong**, and the
error was a category mistake: the rule is about a **quality** payoff, and I tested a
**rate** delta.

Rule 8 says generative methods "only pay off where the baseline is visibly
quality-limited: hallucinating detail is only cheaper than coding it when the codec
can't afford the detail." Its recorded basis is SVT-AV1 measured as *bits for equal
quality*. My test was x265 measured as *bits at equal QP*. Neither the codec nor the
comparison basis matched.

Testing the quality axis on the same 23 x265 clips, background LPIPS, PRESLEY minus
baseline (negative = PRESLEY better):

| QP | median BG gap | PRESLEY better | median rate delta |
|---|---|---|---|
| 32 (richest) | **+0.0409** | **0/23** | −6.6% |
| 37 | +0.0286 | 2/23 | −2.5% |
| 42 | +0.0131 | 3/23 | +10.4% |
| 47 (most starved) | **−0.0153** | **17/23** | +31.4% |

**The quality advantage flips exactly as rule 8 predicts.** At the starved rung
PRESLEY's restored background beats the baseline's crushed background on 17 of 23
clips. The rule is confirmed on a second codec and a larger corpus than it was
written from.

## What is actually true, and it is still worth reporting

**The method trades rate against quality, and the ladder decides which side you are
on.** Foreground stays indistinguishable throughout (median gap 0.0007–0.0047), so
the background is the whole story:

- **Comfortable rungs:** PRESLEY is *cheaper and worse* — −6.6% bits at a background
  gap of +0.0409, which is close to the 0.05 perceptual margin.
- **Starved rungs:** PRESLEY is *dearer and better* — +31.4% bits at a background gap
  of −0.0153.

Consequences for the article:

1. **`fig:breadth`'s two rungs are QP 32/37 — the comfortable end.** So its "saves
   bits on 22 of 33" reports the regime where the saving is real *and the background
   is measurably worse*. That is what the crosses were marking, and it is why the
   figure needs its quality qualification stated in the running text, not just in the
   caption.
2. **A single-rung claim in either direction is not a claim about the method.** The
   four-rung ladder is what makes the trade visible, which is the strongest argument
   yet for W1-A's harmonization.
3. **The BD-rate over the whole ladder mixes the two regimes** and is therefore hard
   to interpret on its own — the per-rung table is the honest presentation.
4. **Hard rule 8 stays. Nothing is removed.** The audit of "why the rule appeared to
   hold" is answered: it holds, and it held for the reason it states.

# LPIPS-bleed audit — scope is ONE claim, not three

Swept every CLAIM anchor for hashes belonging to hole transports evaluated
*without* restoration (166 such runs on disk; 242 more have an in-painter and are
unaffected, since the holes are filled before the metric sees them).

Three claims cite such hashes:

| claim | risky hashes | affected? |
|---|---|---|
| `tab:breadth` | 36 | **YES** — the ELVIS breadth arm is `inpainter: none` and the claim uses **FG**-LPIPS |
| `tab:goal2` | 4 | **no** | 
| `tab:goal2-breadth` | 8 | **no** |

The two Goal-2 tables use their `[none]` runs as the **matched unrestored control**
for a *background* gain. The bleed is an FG-side artefact — a black hole next to a
foreground block contaminates the foreground's deep features — whereas on the
background of a hole arm the holes *are* the content being measured, which is the
intended measurement, not an error.

**So the correction is confined to `tab:breadth`'s foreground statement.** Its
sentence "the LPIPS delta staying under 0.02 on every clip at both rungs" covers
both arms; it holds for the PRESLEY arm (no holes, measured delta 0.0047 median,
0/101 points beyond the margin) and is contaminated for the ELVIS arm (median
+0.0686 while FG-PSNR moves only −0.10 dB). Scope the sentence to the arm it is
true of, and use FG-PSNR for the hole arm.

**Standing rule this establishes:** a mask-restricted deep-feature metric (LPIPS,
DISTS) is only valid on a region whose *neighbourhood* is also intact. On a
transport that empties neighbouring blocks, use a pixel-exact metric for the
protected region, or measure after restoration.

# F3 — boundary-artifact analysis: there is no seam, and the boundary is the best place to be

**Why.** The introduction argues ROI coding stalled on the *seam* — coding
neighbouring regions at different quality leaves a visible boundary — and that a
generative layer removes the need for one. That is currently asserted, not measured.

**Method.** Among **degraded** blocks only, compare those touching an undegraded
neighbour ("seam") against those whose four neighbours are all degraded
("interior"). Both got the identical operator at the identical strength, so any
difference is positional.

**First pass was confounded, and the confound was large.** Comparing raw block PSNR
gave seam blocks +1.753 dB better (303/308 runs, p=4e-52) — an implausibly clean
result in an unexpected direction, which is what flagged it. The cause: the
clustering blur makes selections contiguous, so *interior* blocks sit in the middle
of large degraded regions, which the selector chose because they are the most
removable — the most complex content, which has lower PSNR before anything is done
to it. Raw PSNR compared content difficulty, not position.

**Controlled result — per-block damage against each block's own pristine baseline:**

| | |
|---|---|
| median seam-minus-interior damage | **−0.517 dB** (negative = the seam is damaged *less*) |
| seam damaged more | 23 of 149 runs |
| Wilcoxon signed-rank | p = 3.2e-19 |
| runs beyond the 0.5 dB margin | 75 of 149 |

**Blocks at the degraded/undegraded boundary come back better than blocks in the
interior of degraded regions, by about the perceptual margin.** The mechanism is
the conditioning the architecture is built on: a degraded block adjacent to intact
content gives the restorer real high-frequency neighbours to work from, while a
block surrounded by other degraded blocks has none.

**What this supports, and what it does not.** It refutes the specific worry that
this transport concentrates damage at region borders — the mechanism a visible seam
would require. It is *not* a measurement of seam **visibility**: a discontinuity can
exist with low damage on both sides, and testing that needs a gradient statistic
across the boundary, which this does not compute. So B1 may say the boundary is not
where damage concentrates, and may not say "no visible seam".

# Two audit items closed

## `tab:budget-knee` — the inertness claim is consistent with equivalence, but n=2

The claim that the rate lever is "inert below 0.25" is an equivalence claim, which
a difference test cannot support (significance audit, item 1). Tested properly:

Bitrate change from `shrink_amount` 0.10 to 0.25 is **−0.24%** and **+1.18%**;
TOST returns `equivalent` against a ±5% bound. For contrast, 0.25 → 0.50 moves
**−22.1%**, so the knee itself is real and sharp.

**But the effective n is 2, not 4.** The four cells are two videos × two restorer
settings, and **bitrate is fixed at encode time**, so restorer variants of the same
encode carry an identical rate — the duplicate rows inflate n without adding
information. At n=2 the tool's own warning applies: a bootstrap over two points
reproduces the observed range rather than a sampling distribution and will certify
almost any tight pair.

**Verdict: "consistent with equivalence", not evidence of it.** Wording must not be
upgraded to "demonstrably inert" on this evidence. Two more videos would settle it,
and this is the cheapest outstanding significance item in the article.

**Generalisable trap, worth the research log:** when pooling runs for a *rate*
statistic, deduplicate by whatever fixes the rate. Any arm that differs only after
the encode contributes one measurement, not several.

## `tab:av1-breadth` vs `fig:regime-reversal` — 4 of 6 rows are redundant

The figure plots the SVT-AV1 starved bridge arm (bs16) on `bear`, `camel`, `dog`,
`pigs` at four recalibrated rungs — the *blackout* fill. The table carries those
same four plus two extra rows: `dog`/freeze and `pigs`/freeze.

So they are **not** two renderings of one experiment, as an earlier plan entry
claimed; the table adds a fill comparison the figure does not show. But the extra
content is two numbers (freeze costs +34.4% and +30.4% against blackout's +27.3%
and +18.3%, i.e. freeze is the worse fill in the starved regime), which is one
sentence.

**Action: delete the table, keep the figure, move the freeze contrast into the
running text.** A genuine page saving that no pending result can invalidate.

# Wave 2 results

## W2-B — graded vs binary at n=10: direction holds, significance does not

| | |
|---|---|
| clean clips | **10** (`camel` excluded, see below) |
| graded worse | **8 of 10**, median BD-rate **+5.6%** |
| exact two-tailed sign | **p = 0.109** — still underpowered |
| max FG delta between arms | 0.0026 (validity bound 0.02) — **passes** |
| min overlap | 0.89 (gate 0.50) — **passes** |
| clipping breaches | graded **4**, binary **0**, all on `camel` |

The pre-registered direction bound (graded worse on ≥7 of 9) is met. Significance is
not: 8/10 gives p=0.109, and only 9/10 (p=0.021) or 10/10 (p=0.002) would clear it.
The two clips where graded wins are `tennis` (−11.1%) and `motorbike` (−4.2%).

**The stopping rule fixed in advance applies: we stop here.** n was declared as these
three clips and is not extended because the p-value is close. The honest statement is
that grading is worse on the large majority of clips measured in a regime where it can
work, without reaching significance at this n.

`camel` remains the only clip whose graded arm breaches the clipping invariant, on
all four rungs, with binary at 0 of 40. That asymmetry is evidence about the graded
transport and is reported, not dropped.

## W2-A — the exclusion protects the foreground on PSNR, not on LPIPS

Six clips selected on the pre-outcome criterion, so the mechanism is exercised
(8.6–16.1% of foreground degraded when protection is off).

| metric | result |
|---|---|
| **FG-PSNR**, dropping protection | worse on **6 of 6**, median **−0.56 dB**, beyond the 0.5 dB margin on 3 of 6, exact two-tailed **p = 0.031** |
| **FG-LPIPS**, dropping protection | median **+0.0013**, beyond the pre-registered 0.03 on **0 of 6**, beyond the 0.05 margin on **0 of 6**, worse on 4/6, p = 0.69 |
| background BD-rate | median **−6.0%** — dropping protection is cheaper, as expected |

**The pre-registered bound was set on the wrong metric, and I should say so plainly.**
The bound was "FG-LPIPS delta ≥0.03 on ≥4 of 6", but its stated *basis* was a **3 dB
FG-PSNR** figure. Setting an LPIPS threshold from PSNR evidence is the same
axis-mismatch error as the retracted rule-8 test. Reporting the PSNR outcome is
therefore not post-hoc metric shopping — it is the axis the basis was always on — but
the mis-specification is mine and is recorded rather than quietly corrected.

**Reading:** hard exclusion buys about half a dB of foreground fidelity and no
measurable perceptual difference. Both halves matter. The fidelity gain is real and
significant at 6/6; the perceptual null means a viewer would probably not see the
difference on this content, which bounds how strongly the mechanism can be sold.

## The paper's 3 dB `bmx-trees` figure does not reproduce

`sections/presley.tex` states that without exclusion a score-based top-k "degraded
12.8% of foreground blocks and cost 3 dB of foreground PSNR on `bmx-trees`". Measured
here under a controlled A/C contrast on the same clip: **−0.42 dB**, not −3 dB.

The configurations differ (the original figure predates this contrast and was not run
as a matched pair), so this is not a refutation of the original measurement — but the
article cannot keep quoting 3 dB as *the* cost of dropping exclusion when a controlled
six-clip experiment puts the median at −0.56 dB. **Replace the 3 dB figure with the
controlled result.**
