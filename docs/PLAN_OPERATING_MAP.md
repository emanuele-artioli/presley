# Plan A — from "our pipeline wins" to an operating map

**Status:** proposed 2026-08-02. Wave 1 passed the gate 2026-08-03; **Waves 2A,
2B and 2C then ran, and gate condition 2 substantively FIRES.** This plan needs
re-scoping before any further work — see the box below. Workstream 1 of 2.

> ### ⛔ Gate condition 2 fires: the map collapses under quality-first
>
> Wave 1's coded check said "2 distinct winners, not firing". That check counts
> **per-cell** winners, and the 2 rested on a single cell. Three independent
> lines of evidence, from three waves that did not share code, now say the
> substantive condition is met:
>
> 1. **Per-cell (Wave 1):** 18 of 19 quality-first separable cells name
>    `downsample+realesrgan`.
> 2. **Paired significance (Wave 2B):** it beats `blackout+propainter` on 10/10
>    videos — and after Holm over the real family (m=14 candidates, losers
>    included) that is **the only comparison in the entire corpus that survives**
>    (p_Holm 0.027, n=10, clearing the n≥8 restorer bar). `blur+nafnet` misses at
>    0.0508; `freeze+propainter` is `underpowered (n<8)`. **Under rate-first
>    nothing is significant and the direction reverses.**
> 3. **Content axis (Wave 2A): NEGATIVE, and degenerate.** No EVCA-derived
>    attribute predicts transport choice. The one that fired (motion, rho +0.642)
>    was **withdrawn** — it fails its own pre-registered robustness subset,
>    correlates with a video's *cell count* as strongly as with the outcome, and
>    is beaten by dataset **provenance** at rho +0.907, which fired the alarm
>    bound and was handled as a confound. The deeper point: **T1 could not be
>    tested at all**, because 18 of 19 cells name the same arm and no attribute
>    can predict a constant.
>
> **What this kills.** The claim this plan was built on — *"for a given kind of
> content at a given requested quality, there is a best choice"* — is not
> supported. Content does not select the transport, and under the objective that
> matters most it is the same arm everywhere. The (content × rate) lookup table
> is dead.
>
> **What survives, and it is a cleaner paper than the one we were defending.**
> The plan already anticipated this: *"The map collapses to a single global
> recommendation. Also fine, also a different paper."* Concretely:
>
> - **One recommendation, with one significance-backed comparison behind it:**
>   `downsample+realesrgan`, n=10 videos, p_Holm 0.027, on BG-LPIPS.
> - **The transport is selected by the deployer's objective, not by the
>   content** — and even that axis is weaker than it looked, because rate-first's
>   preference has no significant support and Wave 2C found it **replaceable for
>   free in 14 of 50 cells, up to 9.8× faster at perceptually indistinguishable
>   quality**. Rate-first prefers the corpus's slowest fill.
> - **The corpus is confounded with coverage.** Provenance predicting an outcome
>   at +0.907 is a fact about how runs were scheduled, not about video content.
>   Any future breadth claim has to be read against that.
>
> **Before resuming:** re-scope this plan to the single-recommendation framing,
> or retire it in favour of a new one. Wave 2B's remaining batches are still
> worth running — they *balance* the coverage confound 2A identified, and
> `tools/analyze_content_axis.py` should be re-run afterwards, which costs
> nothing. But do not commission Wave 3 or restructure the article against the
> map framing.
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

## Hard rules that bind this workstream (read before designing anything)

Fixed-QP only; `actual_bitrate_bps` for the rate axis; BG-LPIPS is the Goal-3
verdict and BG-PSNR never is; FG claims only from true masked metrics; a run
with non-empty `invariant_failures` is not citable; `presley-compare` decides
whether a quality difference is real; **n≥6 videos before any significance
claim** (n≥8 for restorer comparisons). State bounds before reading numbers,
and record a fired bound as revised-with-a-reason rather than dropping it.

### ⚠ JND is an effect-size threshold, not a significance test

These are two different questions and this workstream must not conflate them:

- **JND** asks *is the difference perceptible?* — an effect size compared
  against a fixed perceptual threshold.
- **Significance** (hard rule 2b: two-tailed sign test, Holm over candidates,
  TOST for equivalence, n≥6 videos) asks *is the difference real, or sampling
  noise?*

A result can be either without the other. Reporting a JND multiple alone,
which most of the article currently does, answers only the first — and invites
a reviewer to ask why the threshold is what it is.

**The thresholds are exposed.** `src/presley/compare.py` describes them as
"literature just-noticeable-differences" but **cites nothing**, and it assigns
LPIPS and DISTS the *same* 0.05 despite their being different metrics on
different scales. PSNR 0.5 dB and VMAF 6 have reasonable provenance; the
perceptual-metric thresholds are adopted convention.

**Required of this workstream, cheaply:**

1. **Report both** wherever n permits — a JND multiple *and* a test — rather
   than letting a JND multiple stand in for significance.
2. **Sensitivity-analyse the threshold.** Recompute the map with the LPIPS
   threshold at 0.03 and 0.08. **A recommendation that survives all three is
   robust to the constant; one that flips is an artifact of it and must be
   reported as threshold-dependent.** This is a few minutes of compute and it
   answers the obvious reviewer challenge directly.
3. **Cite or downgrade.** Find provenance for each threshold, or state plainly
   in the article that it is an adopted operating convention and lean on the
   sensitivity analysis instead.


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

The magnitudes can invert the ranking a pass rate suggests: ac_truncate has the
*highest* reduction pass rate and the *smallest* median saving (−6.7%). A
transport that reliably saved 2% would score 100% on the gate and be useless.

**Build the map on effect sizes measured within operating points** — not on
pass rates, and not on medians pooled across cells, since transports were run
on different videos and a pooled median is confounded by coverage.

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

Grouped into waves. **Wave 1 is a falsification gate** (see its exit criteria
below); Waves 2A/2B/2C are independent of one another and run in parallel
worktrees once it passes.

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
plausible **2–4** of the available transports; alarm =1 or =every transport.
Ladder residual for the best off-ladder arm: plausible **1–3× JND**; alarm >5×
(suspect a mismatched control).

### ⚠ Wave 1 is the falsification gate — Wave 2 does not start until it passes

Wave 1 is deliberately ordered first because it can kill the framing cheaply,
before any GPU time or any restructuring of the article. Three outcomes end
this plan rather than continue it, and each must be checked explicitly and
reported:

1. **No cell has a JND-separable winner.** There is no map. The honest
   contribution becomes "transport choice does not matter within JND" — a
   different, simpler paper, and a legitimate one.
2. **One transport wins everywhere.** The map collapses to a single global
   recommendation. Also fine, also a different paper.
3. **Every off-ladder residual is within JND.** "Best of both worlds" is not
   achievable with the transports we have, and the reduction/restoration
   tradeoff is hard rather than beatable.

A fourth, softer stop: if the map's recommendations **flip under the threshold
sensitivity analysis** above, the map is an artifact of the JND constant and
must not be published as a decision rule.

**Do not begin Wave 2 until Wave 1 has reported against all four.** Waves 2A,
2B and 2C are independent of each other and should then launch together, each
in its own worktree.

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

**Priority order, revised 2026-08-03 against what Wave 1 measured.** The
original order (transport coverage → videos → new transports) was written
before the map existed and put the `none` controls nowhere. Wave 1 shows they
are the binding constraint:

1. **Matched `none` controls first — the cheapest fix in the corpus.** The
   ladder residual, the plan's BD-rate analogue and the scalar `F1` plots, is
   fitted on **12 of 130 operating points**, purely because a residual needs the
   arm's own unrestored control and those are far sparser than baselines. The
   **16 pending fixed-QP `none` controls already sitting unrun in the queue**
   (legitimate, never run, no one has decided whether they are wanted — they
   are) are the first batch. A control is an encode plus an evaluation with no
   restorer: the cheapest run type there is, and it multiplies the reach of runs
   already paid for.
2. **Transport coverage at the cells that cannot pose the question.** 24 of 84
   cells have **no deployable arm at all** and 14 have exactly one. Those are
   not holes in the sense of missing data — they are cells where nothing on
   offer both saves bits and holds FG. Adding a 3rd/4th transport there is what
   turns them into map cells; adding one where 4 already exist is not.
3. **Paired coverage for a significance test — sharper than "add videos".**
   The quality-first winners already span **8 videos**, so the raw video count
   is not the blocker. What hard rule 2b needs is the *same pair of arms*
   compared across ≥6 videos at comparable operating points. Target the
   specific pairing the map keeps naming (`downsample+realesrgan` vs its
   runner-up) rather than adding videos generically, or n will keep rising
   while no single comparison reaches n≥6.
4. **New transports last**, and only against a named empty region.

**Two things to watch while running, not afterwards:**

- **Re-run `tools/build_operating_map.py` after every batch.** 18 of 19
  quality-first winners are already the same arm. If new coverage makes it
  19/19, **gate condition 2 fires retroactively** — the map collapses to one
  global recommendation and this becomes a different paper. That is a cheap
  check and an expensive thing to discover at the end.
- **Do not let 2B and 2C contradict each other.** Rate-first's winner is
  `blackout+propainter` at **0.78 fps**, the slowest generative fill in the
  corpus; quality-first's is `downsample+realesrgan` at 4.48 fps, 5.7× faster.
  Any 2B batch that adds propainter arms adds them at ~6× the wall clock of a
  realesrgan arm, so schedule accordingly.

### Wave 2C — efficiency, and the real-time question (no GPU)

Add cost per goal to the existing tables from logged
`restoration_time_seconds`, and settle a question the article currently assumes
away.

> ### ⛔ The table that stood here was wrong. Corrected 2026-08-03 by Wave 2C.
>
> **Do not quote the old rows; they are kept only in git history.** Every one of
> them was a **pooled median across disjoint acquisition batches**, and the
> pooling — not the hardware — produced the anomaly the plan asked us to chase.
>
> The alarm was aimed at the wrong number. svtav1 at 1280×720 pools two batches
> whose ranges do not overlap: QP 30–37 runs at **0.7–1.2 fps** and QP 43–62
> runs at **23.8–37.4 fps**. Any central value of that mixture lands in a gap
> **no svtav1 720p run in the corpus has ever occupied**. Resolution was
> confounded with QP by accident of scheduling, and QP moves encode time here
> more than resolution does. Verified independently against `results/presley.db`.
>
> **QP-matched, the impossibility disappears and the ordering is monotone:**
>
> | QP | 640×360 | 1280×720 | 1920×1080 |
> |---|---|---|---|
> | 43 | 36.3 | 36.0 | 31.7 |
> | 50 | 39.3 | 32.8 | 30.7 |
> | 58 | 81.9 | 35.1 | 29.7 |
> | 62 | 73.9 | 27.6 | 28.2 |
>
> **This overturns the claim the table was built to support.** "The baselines
> are not real-time above 360p" was an artifact of the same pooling: QP-matched,
> svtav1 preset 8 clears 24 fps at **every** resolution measured. That sentence
> must not be repeated anywhere.
>
> A second correction fell out: **every `kvazaar` and `x264` baseline in the
> corpus is VBR** (kvazaar 8/8, x264 1/1 — confirmed). The old `kvazaar 12.6
> fps` row was therefore not fixed-QP evidence at all, and there is no fixed-QP
> timing for either codec. Both rows are dropped rather than caveated.
>
> **What survives, and it is the part that mattered:** we still cannot claim
> real time end to end — but **the gap is entirely restoration-side, and
> encoding was never the bottleneck.** Against a 41.7 ms frame budget, encode
> costs 11–15 ms at 360p and restoration costs ~206 ms, overrunning the whole
> budget ~5× on its own. Resolution is not a lever (restoration is full-frame by
> design and independent of `shrink_amount`), and a dedicated GPU plausibly
> recovers ~2×, not 5×. **Real time therefore requires a cheaper restorer, not
> better hardware** — which is exactly the ordering Wave 3(a)/(b) already sets
> out, now with a measured reason rather than an assumption.
>
> Full per-batch table with measurement conditions, and the reason a pooled cell
> median is never quoted again: `docs/WAVE2C_EFFICIENCY.md`.

**The lesson this cost, worth more than the numbers:** a median over a cell that
mixes acquisition batches is not a measurement of that cell. Report per batch
with n and dispersion, or report "not measurable". The plan asked for an alarm
to be closed and the alarm turned out to be pointing at the wrong row — closing
it properly overturned the premise it was attached to.

Deliverables: (i) a throughput row per configuration with the measurement
conditions stated, (ii) an explicit statement of what *would* be needed for a
real-time setup (encoder preset, resolution, hardware), since a reader whose
setup differs may sit on the other side of the line, (iii) cost attached to
each map recommendation so it is a decision variable, not a footnote.

### Wave 3 — conditional: cheaper inference, then new methods

Do **not** commission new models or transports before the map exists. The wave
just completed showed the failure is content-dependent, not model-dependent;
adding a restorer would not have touched it. Wave 3 is justified only by a
specific empty region of the map.

**Exhaust the cheap knobs before compressing anything.** Ordered by what each
costs to try, and every one of them is cheaper than post-training quantization:

*(a) The scope of restoration — the largest single lever, and unmeasured.*
`upscale_realesrgan_adaptive` applies the model to the **entire frame** by
design, not only to degraded blocks; its docstring gives the reason — blocks
must "see their neighbors during upscaling for proper context". The timing data
agrees: median Real-ESRGAN restoration is ~24–27 s at `shrink_amount` 0.10,
0.50 and 0.75 alike, i.e. **cost is independent of how much of the frame was
actually degraded**. A block-local or dilated-region variant would cut cost
roughly in proportion to the untouched area — but it trades away the neighbour
context the current design exists to preserve, so **measure it as a
speed/quality tradeoff rather than assuming it is free**. This is a design
parameter, not a defect.

*(b) Parameters that already exist and have never been swept.* Across the whole
corpus every restorer carries only **1–2 distinct parameter sets**, so this
space is effectively unexplored:

| knob | status |
|---|---|
| `denoise_strength` (Real-ESRGAN) | **never varied** — always 1.0 |
| `num_inference_steps` | only in retired InstantIR (1 vs 20, a correctness fix not a speed sweep) and Stream-DiffVSR, fixed at 4 |
| NAFNet `width` | 64 only; smaller widths never tried |
| model variant | one checkpoint per family; no lighter variants tried |
| `tile` / `tile_pad` | 0 vs 400 only |

**"Fewer steps" is the classic knob for any diffusion restorer and we have
never swept it for throughput.** If a diffusion backbone re-enters the map, its
step count belongs on the cost/quality frontier before its weights do.

*(c) Already settled, do not redo.* fp16 is the shipped default, full precision
is **1.5× slower**, and retiling does not help. So precision and tiling are
done for Real-ESRGAN; (a) and (b) are not.

Every knob in (a) and (b) is a config change costing one run, against hours for
post-training quantization and days plus new artifacts for distillation. They
also land on the same Pareto axis as precision, so they extend the frontier
rather than competing with it. **Only once (a) and (b) are exhausted does
compressing the weights become the cheapest remaining option.**

**Quantization / distillation lives here, gated on Waves 1–2.**

**The restorer is the bottleneck, which is what makes compressing it the right
lever.** Median wall-clock split over fixed-QP treated runs:

| configuration | encode | restore | restore share |
|---|---|---|---|
| `presley_ai` svtav1 640×360 | 6.7 s | 13.1 s | 66% |
| `presley_ai` x265 640×360 | 11.5 s | 24.6 s | 68% |
| `presley_ai` svtav1 1280×720 | 20.3 s | 78.7 s | 80% |
| `elvis` svtav1 640×360 | 3.7 s | 46.2 s | 93% |
| `elvis` x265 640×480 | 6.0 s | 304.8 s | 98% |

Restoration is **66–98% of wall clock** everywhere it runs. Compressing it is
therefore the lever that moves total throughput, and Amdahl's law says nothing
else will.

**If a configuration turns up where the encoder or the degradation step
dominates instead, quantization and distillation are the wrong tools for it.**
Those are classical algorithms, not networks: the remedies there are
parallelisation, moving a CPU-bound stage to GPU, or re-examining an
implementation we may simply have written inefficiently. Diagnose which stage
dominates before choosing a remedy.

**Triggers.** Do this when either holds after Waves 1–2:

- the map shows a **cheap restorer is competitive on quality**, so cost becomes
  a live decision variable between near-equal options; or
- **throughput is close enough to a threshold that matters** (real time, or a
  transcode-budget target) that a 2–4× restorer speedup would cross it.

**If compression is still wanted after (a) and (b), scope it small.** One model
(Real-ESRGAN — the shipped default and the cheapest generative option at
2.8 fps), one method (int8 post-training quantization: no retraining, hours not
days), on operating points the map already marks as recommended.

**Do not gate it on "quality must stay within JND of fp16".** Model precision
is a cost/quality tradeoff of exactly the same kind as the transport tradeoff
this plan is built around, and it belongs on the same footing: **measure the
quality lost and the cost saved, and let the map decide.** An int8 model that
gives up half a JND for 4× throughput may be the right recommendation at a
tight compute budget and the wrong one at a generous quality target — which is
precisely a map axis, not a pass/fail gate. Concretely: add model precision as
a third dimension alongside transport and fill, and report its Pareto frontier
the same way. Extend to more models, or to distillation (which needs training
and is a different order of effort), only if the first probe changes a
recommendation somewhere in the map.
