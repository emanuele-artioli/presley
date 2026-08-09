# W1g — racing the corrected selection objective

**Date:** 2026-08-09. **Verdict: the correction does not pay.**
Reproduce with `python tools/analyze_selection_race.py --data-root .`

## What was tested

The article diagnoses the selection score as mis-specified: it models how many
*bits* a block costs and has no term for how well that block survives
restoration, and M1 showed the two are positively correlated — so the objective
preferentially degrades the blocks that come back worst.

This races the correction that diagnosis implies.

| arm | rule |
|---|---|
| control | `removability` — the existing score |
| corrected | `restorability` — that score divided by predicted post-restoration damage |

Everything else is held fixed: same clustering blur, same hard foreground
exclusion, same budget, same encoder, same QP rungs (43/50/55/60), same
restorer. Verified on `bear` that both arms degrade exactly 73800 blocks.

The damage predictor is a ridge fit on the five features pre-registered in
`docs/PREREG_M1_RESTORABILITY.md`, held out **by video** — `damagemodel.load`
raises rather than scoring a clip with a model that trained on it, because that
leak would be undetectable downstream. Held-out skill is Spearman **+0.400**
median, positive on all 13 folds.

## Result

48 runs, 6 videos × 4 rungs × 2 arms. Zero run errors, zero
`invariant_failures`, all 48 carry region LPIPS. Every cell inside the
pre-registered band `[-20%, +25%]`; no alarm fired.

BD-rate of corrected against control (negative = corrected needs fewer bits):

| video | BG-LPIPS | FG-LPIPS |
|---|---|---|
| bear | +17.8% | +3.1% |
| camel | +6.1% | +2.5% |
| dog | +4.7% | +0.7% |
| india | −2.6% | −0.6% |
| pigs | +5.5% | −0.5% |
| tennis | −2.0% | −0.1% |
| **median** | **+5.1%** | **+0.3%** |
| **corrected better** | **2/6**, p=0.6875 | 3/6, p=1.0000 |

## Reading

**The null is a net loss.** The correction was always going to start down on
the rate axis — measured at **+3.6% bits** on `bear` at QP 50 *before* the race
was run, and recorded in the analysis tool before any result was read — so it
had to earn that back through background quality. It returns nothing
measurable: 2/6 at p=0.69.

**What this does not say.** n=6 floors at p=0.031 on an exact two-tailed sign
test, and 2/6 gives 0.69. This cannot separate "no effect" from "a small effect
in either direction". It is a null, **not** a demonstration that the corrected
rule is worse.

**Scope.** One instantiation: a linear ratio with a ridge-fit damage model.
This is not evidence that restorability-aware selection is impossible, only
that the obvious formulation of it does not recover the loss.

**The foreground panel is the validity check, and it passes.** Both arms
hard-exclude the foreground, and FG-LPIPS is flat at median +0.3% on a 3/6
split — so the arms genuinely differ in *which blocks are selected* and in
nothing else. A large foreground move would have meant something other than
selection changed, and would have been a reason to distrust the run.

## Why it was worth running

A cheap screen (`tools/analyze_corrected_objective.py`) was run first, to kill
the one failure mode that would have made the campaign worthless in advance:
both terms are monotone in complexity, so the ratio could have been degenerate
and reselected the same blocks. It is not — the rules differ on ~9% of selected
blocks, and the predictor generalizes.

So the article can now state a finished thing rather than an open one: the
missing term was identified, shown predictable at transmit time, built into the
rule the diagnosis implies, raced, and measured. **Predicting restorability is
necessary and not sufficient.**

## Provenance

- Run file: `config/w1g_selection_race.yaml`
- Model: `config/damage_predictor.json` (13 leave-one-video-out folds)
- Screen: `tools/analyze_corrected_objective.py`
- Analysis: `tools/analyze_selection_race.py`
- Bounds stated in advance: `PLAN_SUBMISSION_PREP.md` §3.3
