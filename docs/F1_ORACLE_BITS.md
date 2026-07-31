# F1: is the numerator actually solved? EVCA vs a true bit oracle, under inter coding

**Design and bounds pre-registered 2026-07-31, before any run exists.** Nothing in
this document was written after looking at a number, because there is nothing to
look at yet — the re-run has not been launched. Per the bound-before-believing
rule, this file is committed *before* the campaign.

**Status: DESIGNED, NOT LAUNCHED.** The next session executes it.

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
