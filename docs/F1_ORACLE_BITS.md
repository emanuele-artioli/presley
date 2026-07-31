# F1: is the numerator actually solved? EVCA vs a true bit oracle, under inter coding

**Design and bounds pre-registered 2026-07-31, before any run exists.** Nothing in
this document was written after looking at a number, because there is nothing to
look at yet — the re-run has not been launched. Per the bound-before-believing
rule, this file is committed *before* the campaign.

**Status: IMPLEMENTED 2026-07-31** as `component: probe_oracle_bits`
(`src/presley/components/probe_oracle_bits.py`). The bounds and decision rule
below are **unchanged and still binding** — they were pre-registered before any
number existed and are not revised now that implementation details have shifted.

## Corrections to this design, found during implementation

Two things above were wrong. Both are recorded rather than quietly edited,
because the corrections change what the run costs and whether it could have been
cited at all.

**1. The probe would have been uncitable by construction.**
`invariants._check_metrics_present` (`src/presley/invariants.py:61-77`)
unconditionally requires `metrics.{foreground,background,overall}.psnr_mean` and
has **no exemption** for components that produce no reconstructed video. A pure
bit-accounting probe would therefore carry a permanent non-empty
`invariant_failures` and never be citable — which defeats the entire purpose,
since provenance is the only reason this re-run exists.

*Resolution:* the probe publishes its **reference encode** as `output_video`, and
the normal `evaluate_all` pass scores it against the original like any other run.
The metrics are then real measurements of real frames (the pristine quality of
that clip at that QP), not fabricated numbers. *Rejected:* adding a probe
exemption to `invariants.py` — the invariant file is the scientific failsafe, and
carving a hole other components could later slip through costs more than it saves.

**2. The superblock count is 50, not 60.** `get_evca_scores` tiles with
`width // block_size` (**floor**, not ceil), so 640x360 at 64x64 gives 10x5 = 50
full superblocks and the bottom 40-row strip is uncovered. This is adopted as the
definition rather than worked around: `filter_frame_mean_fill` derives its
geometry from the same array shape, so the score vector and `marginal_bits` are
aligned index-for-index **structurally**, and an off-by-one — the most likely
silent defect here — cannot arise. Revised cost: **8 x (50 + 1) = 408 encodes**,
measured at **~136 s per video**, so **~20 min total**, not 1–2 h.

**3. One clarification, not a correction.** The blended score omits the x10
background boost that `get_removability_scores` applies via the UFO mask. That
boost is a *selection policy*, not a cost model; including it would conflate "does
the complexity blend predict bits" with "does foreground protection change what we
pick". Only the former is in question here.

## Why this experiment exists

`NEXT(sec:implementation)` in the paper blocks the "93.0–99.4%" figure from any
reviewer-visible sentence, and that block is what currently keeps the paper's
"the numerator is solved" half unsayable. The figure comes from claim (a) in
`NOTE(sec:implementation)`:

> exact leave-one-superblock-out marginal bit cost (all-intra SVT-AV1 QP43,
> 64x64 SBs, mean-fill) correlates with EVCA SC at Spearman rho +0.754..+0.958,
> and selecting the top-25% of SBs by EVCA frees 93.0-99.4% of the bits a
> perfect bit oracle would free.

Four things disqualify it, and the paper already says so:

1. **No `results/<hash>` at all.** It was an ad-hoc encode sweep, so it is not
   reproducible from the registry and cannot carry a `CLAIM(...)` provenance line.
2. **n = 3 videos**, against the n>=6 bar in `research-log/hard-rules.md` rule 2b.
3. **3–8 frames per clip.** The real clips are 82 frames.
4. **All-intra, which is the fatal one.** With no temporal prediction, a block's
   bit cost is almost entirely its own spatial detail — which is what EVCA SC
   measures. The correlation is then close to tautological, and 93–99% is best
   read as an **upper bound on the proxy's skill**, not as its skill.

So the honest question is not "is EVCA a good proxy" but **"how much of that
93–99% survives when the encoder is allowed to predict across frames?"**

## Design

### The measured quantity

For one video, encode a reference, then for each 64x64 superblock position *i*
encode a variant in which SB *i* is mean-filled in **every** frame. The exact
marginal bit cost of that SB is

```
  marginal_bits(i) = bits(reference) - bits(variant_i)
```

This is deliberately the *operational* quantity rather than a per-block bit
tally read out of the bitstream: under inter coding, blanking one SB changes
bits inside it **and** in the neighbours that predicted from it, and what
selection actually needs to know is how many bits removing that block frees in
total. Reading per-SB bit counts out of the container would answer a different,
easier question.

### Fixed configuration

- **Coding structure: inter, the encoder's default GOP.** This is the entire
  point of the re-run; an all-intra arm is *not* included, because the old
  numbers already serve as the all-intra reference point.
- SVT-AV1, `preset 8`, **fixed QP 43** (`rc=0:q=43`) — matching the original
  probe so the two are comparable. Fixed QP only; never VBR.
- 64x64 superblocks. At 640x360 that is `ceil(640/64) x ceil(360/64) = 10 x 6 =
  **60 SB positions** per video.
- Mean-fill as the removal operator, as in the original.
- Full clips: 82 frames, not 3–8.

### Videos: the full n=8 probe set

`motorbike, drift-straight, drift-turn, color-run, dancing, dogs-jump,
bike-packing, bear` — the pre-registered probe suite from
`tools/select_probe_videos.py -k 8`, one per content cluster. A conclusion that
becomes a paper CLAIM re-runs on all of them; the n=6 -> n=18 collapse is why.

### The two scores compared against the oracle

Report **both**, because they answer different questions and only one of them is
what the paper's equation actually uses:

- **EVCA SC alone** — the quantity the original claim named.
- **The blended importance score at alpha = beta = 0.5** — what
  `Eq.~\ref{eq:importance}` actually computes, blending spatial and temporal
  complexity. Under inter coding the temporal term is exactly the information
  SC lacks, so if the blend does not beat SC here it is evidence the blend is
  not earning its place.

### The two reported statistics

1. **Spearman rho** between the score and `marginal_bits`, per video.
2. **Oracle-capture ratio**: bits freed by the top-25% of SBs *by score*,
   divided by bits freed by the top-25% *by true marginal cost*. This is the
   93–99% number's successor and the one the paper would quote.

### Cost

Measured on this host: one 82-frame 640x360 SVT-AV1 preset-8 inter encode takes
**1.8 s**. 8 videos x (60 variants + 1 reference) = **488 encodes ~= 15 min of
encoding**. Frame I/O dominates, so work in YUV rather than round-tripping PNGs.
Budget **1–2 h wall clock**, CPU-only — no GPU, no restoration, no metrics pass.
This is cheap; the reason it has not been done is that it needed designing, not
that it is expensive.

## How it becomes CLAIM-grade

The blocker is provenance, not measurement, so the tooling must produce a real
`results/<hash>`. A leave-one-out sweep does not fit the existing per-experiment
component model one-run-per-encode, so:

**Add a `component: probe_oracle_bits` to `experiments.yaml`, one entry per
video**, dispatched like any other component from `src/presley/runner.py`. Each
entry's run performs that video's whole 61-encode sweep and writes one
`result.json` carrying `marginal_bits` per SB, both score vectors, the two
statistics, and the usual config/bitrate fields. That gives eight hashes, a
`CLAIM(sec:implementation)` line that cites them, and re-runnability from the
registry — which is exactly what `NEXT(sec:implementation)` demands.

Do **not** compute this with a standalone script outside the runner: that is the
same defect that already disqualifies claim (b) (`tools/mine_block_damage.py`).

## Bounds, written before any number exists

The prior is that the number gets **worse**, and the reason is specific: under
inter coding much of a block's cost is motion-dependent, and SC does not measure
motion. A drop is the expected outcome and is a perfectly reportable result.

1. **Spearman rho (SC vs marginal_bits), per video.** All-intra gave
   +0.754..+0.958. Expect **+0.30 to +0.85** per video and a mean of
   **+0.45 to +0.75**.
   **ALARM if any video exceeds +0.95** — that reproduces the all-intra
   tautology and means the encode is not really inter; verify the GOP before
   reporting anything. **ALARM if the mean falls below +0.15** — that would mean
   the cost proxy is near-useless under realistic coding, which is a major
   negative that changes what the paper can say about Goal 1, and must be
   investigated for an implementation bug (wrong score alignment, off-by-one in
   the SB grid) before it is believed.

2. **Oracle-capture ratio at top-25%.** All-intra gave 93.0–99.4%. Expect
   **70% to 95%**, mean around **85%**.
   **ALARM above 99%** (tautology again). **ALARM below 50%** — that would
   invert the claim from "a better cost proxy is worth at most 1–7%" to "the
   proxy is leaving half the available bits on the table", which is a *paper-
   changing* result and must clear an implementation review first.

3. **Blended score vs SC alone.** Genuinely uncertain, which is why both are
   measured. Expect the blend to be **equal or better**, by **0 to +10
   percentage points** of capture ratio.
   **ALARM if the blend is worse than SC by more than 5 points** — the blend is
   what the paper's equation uses, and it losing to its own spatial half is a
   finding about `Eq.~\ref{eq:importance}`, not a footnote.

4. **Sanity on the oracle itself.** The top-25% oracle should free a meaningful
   share of the reference bitrate: expect **10% to 45%**.
   **ALARM outside 5%–60%**, and **ALARM on any negative `marginal_bits`** large
   enough to matter (small negatives are ordinary encoder noise; a large one
   means mean-filling a block *cost* bits, which points at the fill or the grid
   alignment, not at a finding).

## Decision rule, stated in advance

- Mean capture ratio **>= 70%** with mean rho **>= +0.45**: the numerator claim
  survives inter coding. The paper may say the cost proxy is close to a bit
  oracle, **quoting the inter number and naming the all-intra number as the
  upper bound it is** — never the old 93–99% on its own.
- Capture ratio **50–70%**: report as-is, with the drop from all-intra stated
  explicitly. "Mostly solved" is sayable; "solved" is not.
- Capture ratio **< 50%**: the numerator is **not** solved, and
  `NOTE(sec:implementation)`'s reading — that only the numerator was ever
  modelled — needs revising in the other direction too. This is a real possible
  outcome and gets reported, not retried with a different score until it passes.

In every branch the result lands with its `results/<hash>` set, and
`NEXT(sec:implementation)` is cleared only by the edit that lands the data.

---

# RESULT (2026-07-31, n=8, all runs citable)

`tools/analyze_f1_oracle.py`, over 8 `results/<hash>` with **empty
`invariant_failures`** — the citability fix works, which was the point of routing
this through the runner at all.

| video | rho SC | rho blend | cap SC | cap blend | random null | cap − null |
|---|---|---|---|---|---|---|
| bear | 0.704 | 0.710 | 0.765 | 0.765 | 0.353 | +0.412 |
| bike-packing | 0.886 | 0.929 | 0.881 | 1.000 | 0.382 | +0.499 |
| color-run | 0.896 | 0.909 | 0.940 | 0.927 | 0.546 | +0.394 |
| dancing | 0.713 | 0.824 | 0.922 | 0.952 | 0.533 | +0.388 |
| dogs-jump | 0.784 | 0.846 | 0.917 | 0.980 | 0.255 | +0.661 |
| **drift-straight** | **0.082** | 0.101 | **0.510** | 0.510 | 0.436 | **+0.075** |
| drift-turn | 0.681 | 0.674 | 0.909 | 0.918 | 0.416 | +0.493 |
| motorbike | 0.606 | 0.618 | 0.824 | 0.785 | 0.298 | +0.526 |

**Means are all in bounds and no alarm fired**: mean rho **+0.669** (band
0.45–0.75), mean capture **0.833** (band 0.70–0.95), mean oracle share **0.301**
(band 0.10–0.45). By the pre-registered decision rule (capture >= 0.70 **and**
rho >= +0.45) the numerator claim **survives inter coding**.

## Two per-video bounds BREACHED, both recorded rather than smoothed

**1. `drift-straight` breaches low on both statistics** — rho **+0.082** against a
0.30–0.85 band, capture **0.510** against 0.70–0.95. Investigated before
reporting, per the rule.
*A hypothesis was tested and refuted:* that its marginal-bit distribution is flat,
so the ranking cannot matter. Its coefficient of variation is 0.93 — mid-pack
(bear 1.24, color-run 0.68) — and rho vs CV across the eight videos correlates at
**0.031**, i.e. not at all. **The failure is not explained.** On this one video
the EVCA proxy carries essentially no information about which superblocks are
expensive, and the honest statement is that we do not know why.

**2. Two videos breach high** (`color-run` 0.896, `bike-packing` 0.886 vs a 0.85
ceiling). Minor, and in the direction the band was guarding against
(tautology) — but the tautology alarm at +0.95 did **not** fire, and these sit
well inside the all-intra range of +0.754..+0.958, so this reads as ordinary
per-video spread rather than a coding-structure problem.

## A methodological correction the bounds themselves missed

The capture ratio was pre-registered — by me, and by the original claim — **without
a null**. That was wrong. Random selection of k superblocks already captures a
**mean 0.402** of the oracle's bits, so a capture ratio is not "fraction of the way
to the oracle" and 93–99% never meant what it appeared to mean either.

Quoted against the null, the proxy is worth **+0.431 on average** (range +0.075 to
+0.661). **Any future quotation of a capture ratio must carry its null.**

## What the paper may now say

`NEXT(sec:implementation)` can be cleared for claim (a). Permitted wording:

- the inter-coding numbers (**rho +0.669, capture 0.833 against a 0.402 random
  null, n=8**), citing the eight hashes;
- 93.0–99.4% **only** as the all-intra upper bound it is, never on its own;
- the `drift-straight` caveat stated wherever the mean is quoted — on one of eight
  videos the proxy is barely better than chance, and unexplained.

Claim (b) (the denominator spread) is **still not CLAIM-grade** — it remains
computed by `tools/mine_block_damage.py` outside the runner, and nothing here
changes that.
