# W5 — content axis, round 2: pre-registration

Written 2026-08-05 **before computing any of the three new attributes**. Round 1
is `docs/WAVE2A_CONTENT_AXIS.md` (negative on four attributes); this extends the
same tool, the same targets and the same decision rule to the three candidates
that were named in review and never tested.

## What this can and cannot settle — read before reading the result

**T1 remains untestable and no result here changes that.** 18 of 19 separable
quality-first cells name the same arm, so *which arm wins* has essentially no
variance for an attribute to predict. A hit on T2 or T3 would therefore say
only **where the map has choices to offer**, never *which choice to make*. It
cannot become a deployment rule. This paragraph exists so that a positive
result cannot be quietly upgraded into one afterwards.

**Expect negative.** Round 1 tested four attributes and found nothing; the
prior here is that content does not select the transport, and these three are
being tested because they were named, not because a mechanism predicts them.

## The three new attributes

| id | attribute | definition | why it was not tested in round 1 |
|---|---|---|---|
| **A5** | background motion | mean EVCA TC over **non**-foreground blocks | round 1 tested global motion (A1) and foreground motion (A2) but never the FG/BG *decomposition*. A5 is A1's complement and one line from A2 |
| **A6** | duration | `runs.video_frames` | a **64× spread** (38 to 2440 frames). Round 1 blamed a coverage/provenance confound for A1's near-hit and inferred it; A6 measures it directly |
| **A7** | foreground fraction | fraction of blocks flagged foreground by the first annotation | round 1 did *not* re-test it, relying on a prior refutation that was against a **different target** (arm win/loss, not T2/T3). A refutation against another target is not evidence here |

A6 is the one to watch. It is not a content property in the same sense as the
others — it is closer to a measure of how much of each clip exists — so a hit on
A6 is more likely to be the coverage confound made visible than a finding about
content. That reading is registered now, before the number.

## Fixed by this pre-registration

- **Unit = video**, never the cell. Attributes are constant within a video, so
  cells are not independent observations.
- **k = 7** for the Holm correction: the four round-1 attributes plus these
  three, whether or not each computes. Re-running round 1's four under the
  larger k makes their correction strictly more conservative; they were
  negative, so nothing can be weakened by it.
- **Targets unchanged**: T2 (separable vs tie among contested cells), T3
  (no-eligible-arm among all cells). T1 is refused by the tool, not tested.
- **Decision rule unchanged**: predictive requires n ≥ 6 videos, |rho| ≥ 0.6
  and p_Holm < 0.05. |rho| ≥ 0.6 without significance is reported as
  *suggestive* and is not a result.
- **Two-tailed permutation test**, 10 000 permutations, seed 0.

## Bounds

| quantity | plausible | alarm | basis |
|---|---|---|---|
| attributes reaching *predictive* | 0 of 3 | ≥ 2 of 3 | round 1: 0 of 4. Two hits out of three fresh attributes after four misses would more likely indicate an analysis error than a content axis |
| \|rho\| for any attribute | 0.0–0.6 | > 0.9 | round 1's largest was A1 on T3, and it did not survive its confound check. \|rho\| > 0.9 on n≈20 videos is a data-shape alarm, not a finding |
| A6 (duration) vs a video's **cell count** | 0.3–0.9 | — | the confound the attribute exists to expose; a high value here is the expected reading, not a surprise |
| videos contributing | T2: n≈19, T3: n≈23 | n < 6 for both | round 1's coverage. Below the hard-rule floor the target is reported as untestable rather than as a null |

Any attribute reaching *predictive* must additionally survive the round-1
confound checks (`dataset_confound`, `coverage_confound`) before being written
anywhere as a result — that is what stopped A1 in round 1.
