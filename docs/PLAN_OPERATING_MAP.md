# Plan A — from "our pipeline wins" to an operating map

**Status:** proposed 2026-08-02, not started. Workstream 1 of 2.
Companion: `docs/PLAN_PRESENTATION.md` (workstream 2), which **depends on this
one** — you cannot choose how to present a result before knowing which results
survive scoping.

---

## The reframe

The article currently argues *this pipeline works*. The 2026-08-02 HOLE wave
showed that framing cannot survive: bit relocation does not replicate beyond
`bear`/`camel`, and no content attribute we tested predicts which clips it
works on. Defending "it works" means defending something the data contradicts.

The claim the data *does* support is stronger and more useful:

> **For a given kind of content at a given requested quality, there is a best
> choice of degradation transport and restorer, and we can say what it is and
> why.**

That converts our biggest weakness — the outcome depends on content and
operating point — into the contribution itself. A reader does not want "PRESLEY
wins"; they want to know what to deploy at 200 kb/s on a high-motion clip.

**Scope of a recommendation = (content class, rate/quality target).** Both axes
are needed: the same video flips sign along the QP ladder, and the same QP
flips sign across videos.

## The mechanism, and why it is not just a tradeoff

The reduction/restoration tension has a physical cause: **the more information a
transport leaves in a degraded block, the more that block costs to code and the
less the restorer has to invent.** Reduction and restoration are therefore
*read off the same underlying quantity* — residual information — from opposite
ends.

This is measured, not assumed. `NOTE(tab:priced-trade)` observed the ordering
on one video and flagged it as n=1. Re-run over the corpus
(`tools/audit_goal_scope.py` data, unrestored arms only):

- **26 operating points, 94 transport pairs: 79.8% concordant** with "more bits
  transmitted ⇔ less background damage". Kendall-style agreement **+0.596**.

So the ladder is real. **The 20% discordant pairs are the whole point of the
paper.** A transport that sits *off* the ladder — saving more bits than its
damage level should allow, or damaging less than its bit saving should cost —
is a genuinely better operating point rather than a different position on one
tradeoff curve. Those are the "best of both worlds" cases.

### ⚠ Pass rates are not effect sizes — do not build the map on them

`tools/audit_goal_scope.py` reports, per transport, the **fraction of cells
that pass a boolean gate**. Those percentages are *not* magnitudes:
"ac_truncate 87.5% reduction" means 87.5% of its cells saved *some* bits with
foreground within JND — it says nothing about how much. "blackout 75%
restoration" means 75% of its cells cleared 1 JND **against blackout's own
unrestored control**, not that 75% of anything was recovered.

An earlier draft of this plan glossed those rates as "strips most, restores
worst". **That was wrong**, and the magnitudes reverse it: ac_truncate has the
*highest* pass rate and the *smallest* median saving (−6.7%). A transport that
reliably saves 2% would score 100% on the gate and be useless.

**The map must be built on effect sizes measured within operating points**, not
on pass rates and not on medians pooled across cells (transports were run on
different videos, so a pooled median is confounded by coverage).

### The magnitudes, paired

Six operating points carry all four core transports *and* a baseline
(`bear`@51, `bear`@58, `camel`@50, `camel`@58, `dog`@50, `pigs`@50). Medians
of the per-cell values, best available fill for the restored column:

| transport | bits vs baseline | BG-LPIPS raw | **BG-LPIPS restored** | gain |
|---|---|---|---|---|
| blackout | **−24.6%** | 0.480 | 0.374 | 1.70× JND |
| blur | −15.1% | 0.368 | 0.343 | 0.24× JND |
| downsample | −14.8% | 0.297 | **0.220** | 1.61× JND |
| freeze | −8.3% | 0.304 | 0.330 | **−0.53× (harms)** |

**The ladder holds in magnitude, not just in rank:** blackout saves the most
bits and leaves the most damage; freeze saves the least and leaves the least.

**Why we do not simply blackout everything.** Blackout posts the largest
restoration *gain* (1.70× JND) purely because it starts from the worst place.
It *ends* at 0.374 while downsample ends at **0.220** — **3.1× JND worse** — in
exchange for 10 points more bit saving. Restoration cannot repay what blackout
destroys. **The gain figure flatters blackout; the absolute restored figure
decides against it.** Any exhibit that shows restoration gain without the
absolute end quality beside it will mislead a reader the same way.

**A Pareto frontier, which is the map in miniature:**

- **`downsample` — the quality choice.** Best absolute restored quality by a
  wide margin, at a mid-range bit saving.
- **`blackout` — the rate choice.** Only pick it when the budget genuinely
  requires −24.6%, accepting a visibly worse background.
- **`blur` — dominated.** Saves the same bits as downsample (−15.1% vs −14.8%)
  and restores 2.5× JND worse.
- **`freeze` — dominated twice.** Saves the least *and* restoration actively
  harms it, corroborating `CLAIM(tab:goal2-breadth)`.

So `downsample` is not "best at neither, acceptable at both" — the earlier
pass-rate reading. On magnitudes it is **the best absolute-quality transport
outright**, and blackout is its only frontier rival. n=6 operating points over
4 videos: descriptive, below the n≥6 *videos* significance threshold.

## What we already have

From `tools/audit_goal_scope.py` over 926 fixed-QP citable runs:

- **74 operating points with ≥2 transports** scored on BG-LPIPS; **33 with ≥3**;
  **19 with ≥4**; across **23 videos**.
- Densest: `bear`/`camel` svtav1 QP 50/51 (6 transports), then
  `dog`/`pigs`/`bear`/`camel` at QP 50/55/58/63 (4 transports).
- Joint goal-2 ∧ goal-3 success in **45 of 186** jointly-scored cells (24.2%).
- Per-cell restoration cost from `restoration_time_seconds` (1.1–5.4 fps).

**A prototype map is buildable today** on ~4 videos × ~4 rates × 4 transports
without new GPU time. That is the first deliverable and it gates everything else.

## Steps

Grouped into waves; a wave starts only when the previous one has reported.
Waves 2A/2B/2C are independent and should run in parallel worktrees.

### Wave 1 — build the map from existing data (no GPU)

1. **`tools/build_operating_map.py`** — for every operating point with ≥2
   transports, rank arms by a stated objective and emit the winner, the
   runner-up, and the margin in JND. Refuse to name a winner when the top two
   are within JND of each other: **most cells will not have a separable
   winner, and saying so is a result.**
2. **Objective must be explicit.** Default: lowest BG-LPIPS subject to
   bits ≤ baseline and FG-PSNR within JND. Report the map under at least two
   objectives (quality-first and rate-first) — a recommendation that survives
   both is far stronger than one tuned to a single scalarization.
3. **Ladder residuals — the scalar the article is missing.** Fit the
   bits↔damage ladder per operating point, then score every arm by its
   **residual** from it. This is deliberately analogous to BD-rate: BD-rate
   scores a codec against a rate-quality curve, and the residual scores a
   transport against the rate-damage curve its peers define. It gives one
   signed number per arm, comparable across operating points, where today we
   have only per-cell verdicts. Rank the off-ladder winners — those are the
   "best of both worlds" candidates, and the residual is what F1 in
   `PLAN_PRESENTATION.md` plots.
   ⚠ Report the residual in JND units alongside the absolute restored quality.
   Blackout is the standing warning: the largest gain and the worst absolute
   result. A residual alone can repeat that error.
4. **Cost as third axis.** Attach fps to each recommendation; mark
   Pareto-dominated arms (worse on both quality and cost).

**Pre-registered bounds for Wave 1.** Cells with a JND-separable winner:
plausible **25–60%**; alarm outside 10–80%. Distinct winners across the map:
plausible **2–4** of the available transports; alarm =1 (means the map is
really one global recommendation and the framing collapses) or =every transport
(means the objective is noise-driven). Ladder residual for the best off-ladder
arm: plausible **1–3× JND**; alarm >5× (suspect a mismatched control).

### Wave 2A — the content axis (the open problem)

The map is a lookup table until "this kind of video" is defined. FG area is
**refuted** (dog 0.114 ≈ bear 0.111, opposite outcomes). Candidates not yet
tested against *transport choice* (as opposed to against win/loss, where the
n=18 attribute test already failed): motion magnitude, temporal stability of
the hole region, background texture energy, and residual-information proxies
already computed by `EVCA`. **Pre-register that this may fail again** — two
attribute tests have already come back negative, and a third negative is a
legitimate result that turns the map into an empirical lookup rather than a
predictive rule.

### Wave 2B — fill the map's holes (GPU, cheap)

Cost is now known to be ~40× lower than the old estimate: ProPainter is
**39–77 s/run** at 640×360, not ~49 min. A full 4-transport × 4-rung sweep on
one new video is well under an hour.

Priority order: (i) complete 4-transport coverage at the QPs where only 2 exist,
(ii) add videos to reach **n≥6**, the threshold at which hard rule 2b permits a
significance claim — the map is currently descriptive only, and n≥6 would let
it be stated as a finding, (iii) only then consider new transports.

### Wave 2C — efficiency, and the real-time question (no GPU)

Add cost per goal to the existing tables from logged
`restoration_time_seconds`, and settle a question the article currently assumes
away.

**Real time is not a hard requirement, but "can we beat a baseline in real
time?" is worth answering — and the assumption that the baselines themselves
run in real time does not hold on this host.** Median encode throughput from
logged `encoding_time_seconds`:

| configuration | median fps | vs 24 fps |
|---|---|---|
| svtav1 baseline 640×360 | 39.3 | 1.6× |
| x265 baseline 640×360 | 25.8 | 1.1× |
| x265 baseline 640×480 | 20.0 | **0.8×** |
| kvazaar baseline 640×360 | 12.6 | **0.5×** |
| svtav1 baseline 1280×720 | 3.2 | **0.1×** |

So the baselines are *marginally* real-time at 360p and **not real-time at all
above it**. We cannot presently claim real time for anything, ours or theirs,
and saying so is more useful than an unqualified "our restorer is slow".

⚠ **One alarm to close before any of this is quoted:** svtav1 1080p measures
28.9 fps against 720p's 3.2 fps. Faster at higher resolution is impossible;
suspect a differing preset, threading setting, or contention from another
user's job on this shared box. **Investigate before publishing any throughput
number** — this is exactly the "bound before believing" case.

Deliverables: (i) a throughput row per configuration with the measurement
conditions stated, (ii) an explicit statement of what *would* be needed for a
real-time setup (encoder preset, resolution, hardware), since a reader whose
setup differs may sit on the other side of the line, (iii) cost attached to
each map recommendation so it is a decision variable, not a footnote.

### Wave 3 — conditional: new methods, and model compression

Do **not** commission new models or transports before the map exists. The wave
just completed showed the failure is content-dependent, not model-dependent;
adding a restorer would not have touched it. Wave 3 is justified only by a
specific empty region of the map.

**Quantization / distillation lives here, gated on Waves 1–2.** It was deferred
earlier on the grounds that a 2–4× win would not reach real time and so would
not change the offline/VOD positioning. Waves 1–2C can overturn that in two
ways, and if either fires it becomes worth doing:

- the map shows a **cheap transport/restorer is competitive on quality**, so
  cost becomes a decision variable between near-equal options rather than a
  footnote; or
- the throughput work above shows a configuration where **the encoder, not the
  restorer, is the bottleneck** — in which case compressing the restorer moves
  the whole pipeline toward a line it could actually cross.

**If it is triggered, scope it small first.** One model (Real-ESRGAN, the
shipped default and the cheapest generative option at 2.8 fps), one method
(int8 post-training quantization — no retraining, hours not days), on the
operating points the map already marks as recommended. Gate on: does BG-LPIPS
stay within JND of the fp16 result? Only extend to more models, or to
distillation (which needs training and is a different order of effort), if that
first probe both preserves quality and produces a speedup large enough to
change a recommendation in the map. A speedup that changes no recommendation is
not worth the second step.

## What would falsify this plan

- **No cell has a JND-separable winner.** Then there is no map, and the honest
  contribution is "transport choice does not matter within JND", which is
  publishable but a different paper. Wave 1 tests this first, deliberately.
- **One transport wins everywhere.** Then the map collapses to a single
  recommendation. Also fine, also a different (simpler) paper.
- **The ladder's off-ladder residuals are all within JND.** Then "best of both
  worlds" is not achievable with current transports and the tradeoff is hard.

Each of these is cheap to check in Wave 1 and each changes the article's
argument, which is why Wave 1 precedes everything.

## Hard rules that bind this workstream

Fixed-QP only; `actual_bitrate_bps` for the rate axis; BG-LPIPS is the Goal-3
verdict and BG-PSNR never is; FG claims only from true masked metrics; a run
with non-empty `invariant_failures` is not citable; `presley-compare` decides
whether a quality difference is real; **n≥6 videos before any significance
claim** (n≥8 for restorer comparisons). State bounds before reading numbers,
and record a fired bound as revised-with-a-reason rather than dropping it.
