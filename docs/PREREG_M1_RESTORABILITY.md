# Pre-registration — M1: is post-restoration damage predictable at transmit time?

**Committed before any result exists.** Second and final scope hypothesis of
this paper (see `PREREG_R1_REGIME_SCOPE.md` for the first, and for why the
budget is two).

## Why this, and not the multivariate content model

The multivariate content model was considered and **rejected as uncitable at
feasible cost**. It inherits the confounded video-level unit that invalidated
all five prior attempts, *and* multiplies researcher degrees of freedom. A
4-predictor joint model wants ≳10 events per predictor: ~40 videos with
informative rates × ≥3 contested cells ≈ **360 new runs**. At the end of that
we would still be correlating video-level attributes against per-video rates —
the exact shape that failed five times.

M1 targets the same underlying question by a route the research log explicitly
names as untried, and costs **4 runs**.

## The mechanism argument for looking here

The selection score is a ratio — bits freed per unit of damage that survives
restoration — and only its **numerator** was ever modelled.

- **The numerator is nearly saturated.** The EVCA complexity proxy ranks blocks
  against a true leave-one-superblock-out bit oracle at ρ=0.669, capturing
  **0.833** of the oracle's bits against a random-selection null of **0.402**.
  Since the oracle's top quarter is 30.1% of total bitrate, the entire remaining
  headroom on cost is ~5% of bitrate. A separate probe put EVCA SC at 93–99% of
  a bit oracle's savings (all-intra).
- **The denominator is unmodelled and large.** Per-superblock post-restoration
  damage disperses **4.9 dB** (Real-ESRGAN on downsampling) and **8.4 dB**
  (ProPainter on hole filling) between its 10th and 90th percentile *within a
  single run*.

So the headroom is not in a better cost proxy. It is here, or nowhere.

## Hypothesis

> Within a run, some transmit-time-computable feature predicts a superblock's
> **post-restoration damage rank**.

"Transmit-time-computable" is the binding constraint: a feature the server can
compute before sending. Anything requiring the restored output is useless for
selection.

## Design

- **Unit:** the run. Existing corpus: **8** `probe_block_damage` runs
  (4 videos × 2 restorers), each already carrying per-superblock measured
  post-restoration damage.
- **Statistic:** within-run Spearman ρ between feature and damage rank over that
  run's degraded superblocks. Exact two-tailed sign test over runs
  (`suite.sign_test_p`).
- **Features — k = 5, declared now, losers included in the Holm family:**
  1. EVCA SC mean (spatial complexity)
  2. EVCA TC mean (temporal complexity)
  3. EVCA SC variance
  4. EVCA TC variance
  5. frame-edge indicator (block touches the frame boundary)

  No new feature engineering. These are quantities the shipped scorer already
  computes, which is what keeps the degrees of freedom pinned.

**Confound handling is structural, as in R1:** within-run ranks mean video,
dataset provenance, duration, cell count and operating point are all constant
within the unit and difference out exactly.

## Power

| n | unanimous two-tailed p | after Holm k=5 |
|---|---|---|
| 8 (existing) | 0.0078 | 0.039 — survives, tightly |
| **12 (with F1's 4 runs)** | **0.00049** | **0.0024** |

Run the 4 extra probes (2 videos × 2 restorers). Eight is too close to the line
for a claim this load-bearing.

## Bounds, registered before the numbers

| quantity | plausible | alarm |
|---|---|---|
| best within-run \|ρ\| | 0.15 – 0.55 | **> 0.80** — a transmit-time feature that nearly determines post-restoration damage is more likely leakage than a finding; check the feature is not derived from the restored output |
| runs concordant in sign, best feature | 8 – 12 of 12 | unanimous at median \|ρ\| < 0.10 → trivially-ordered, not an effect |
| EVCA SC / TC means | expected **weak** (they are cost proxies, and cost is already saturated) | a strong hit here contradicts the saturation result — reconcile before reporting |

## Both outcomes, written now

**Fires.** The missing term is predictable, and we can say what predicts it.
That is a direct, constructive answer to the "lack of technical insight" review
comment, and it names the next system to build.

**Does not fire.** This is the outcome I expect, and it is **worth more than a
positive content correlation would be.** It upgrades the paper's selection null
from *"α and β are inert"* (a parameter ablation) to *"the missing term is not
merely unmodelled — it is not predictable from any transmit-time signal we can
compute"* (a mechanism claim, with a stated feature family and a bound).

Either way it is reportable, which is the test of a hypothesis worth
pre-registering.

## Stopping rule

M1 runs once, over the 5 declared features. **Do not add a sixth feature after
seeing the five** — that is candidate shopping, and `--candidates-tried` exists
precisely to make it cost something. If a new feature family is wanted later it
is a new pre-registration with its own family size, not an extension of this one.
