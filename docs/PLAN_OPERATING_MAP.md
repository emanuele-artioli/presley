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

This also explains the corpus-level pattern already measured:

| transport | reduction | restoration | reading |
|---|---|---|---|
| ac_truncate | 87.5% | 20.8% | strips most, restores worst |
| blur | 79.3% | 12.5% | " |
| downsample | 69.9% | 34.2% | **acceptable at both** |
| freeze | 59.4% | **0.0%** (0/22) | leaves most, restorer has nothing to add |
| blackout | 58.7% | **75.0%** | strips all, but restoration is well-posed |

`downsample` is best at *neither* goal. It is the only transport acceptable at
both, and **that** is why the article's main pipeline uses it — a fact the
article does not currently state. `blackout` is the ladder's most interesting
resident: it strips everything yet restores best, because a total hole is a
well-posed in-painting problem while a partial one is an ambiguous denoising
problem.

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
3. **Ladder residuals.** For every arm compute residual against the fitted
   bits↔damage ladder. Rank the off-ladder winners. These are the candidate
   "best of both worlds" findings.
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

### Wave 2C — efficiency reporting (no GPU)

Add cost per goal to the existing tables from logged `restoration_time_seconds`.
**Quantization/distillation is explicitly deferred, not forgotten:** the gap to
24 fps is 6–60× and fp16 is already the shipped default, so a 2–4× win would not
change the offline/VOD positioning. Revisit only if the map shows a cheap
transport is competitive, which would make cost a decision variable rather than
a footnote.

### Wave 3 — new methods, only if Waves 1–2 leave a gap

Do **not** commission new models or transports before the map exists. The wave
just completed showed the failure is content-dependent, not model-dependent;
adding a restorer would not have touched it. Wave 3 is justified only by a
specific empty region of the map.

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
