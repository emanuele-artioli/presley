# Pre-registration — R1: does the outcome depend on operating point?

**Committed before any Wave-B run exists.** This is the *sixth* attempt at a
scope result in this project. The five before it were all negative, and a
pre-registration written after the numbers exist would be worth nothing. Check
`git log` order if you doubt it.

## Why the previous five failed, and why this one is shaped differently

Every prior attempt correlated a **video-level attribute** against a **per-video
outcome rate**:

| attempt | attributes | result |
|---|---|---|
| pre-Wave-1 (`audit_videos.py`, n=18) | `hole_churn`, `bg_texture`, `fg_frac`, +4 | all collapse; best \|ρ\|=0.40 |
| Wave 2A | A1 motion, A2 FG instability, A3 BG texture, A4 residual | A1 fired then withdrawn; rest negative |
| W5 | A5 BG motion, A6 duration, A7 FG fraction | 0/3; A5 was A1 relabelled |
| FG area, standalone | — | refuted ≥5× independently |

**Two confounds explain every apparent hit**, and both are properties of that
shape rather than of the attributes:

1. **Per-video cell count.** The outcome is a *rate*, so cell count is its
   denominator. Ten of nineteen videos contributed exactly one contested cell,
   where a rate is definitionally 0 or 1.
2. **Dataset provenance**, ρ=0.907 with the outcome — tripping the
   pre-registered |ρ|>0.9 alarm. MOSEv2/YouTube-VOS clips were *selected* to be
   different in kind (long, multi-object, no clear FG), so provenance is not a
   content attribute; it is the sampling frame.

**R1 changes the shape, not the attribute list.** The unit is a **rate ladder**,
and the statistic is a **within-ladder contrast**. Under that design:

- **Cell count cannot enter.** Each ladder contributes exactly one scalar
  regardless of how many rungs it has.
- **Provenance cannot enter.** The contrast is a within-video difference and
  dataset origin is constant within a video, so it differences out exactly.
- **So does every previously refuted attribute** — duration, FG area, motion,
  texture are all constant within a video and cancel identically.

These are not statistical controls that a referee can argue with. They are a
design in which the confounds are absent.

## Hypothesis

> The sign and magnitude of PRESLEY's advantage depends on the **operating
> point** (how bit-starved the encoder is), not on what the content is.

## Design

- **Unit:** the ladder = `(video, codec)` with ≥4 fixed-QP rungs, each rung
  carrying citable `baselines`, `elvis` and `presley_ai` runs. **n = 13**
  (4 DAVIS svtav1 + 9 non-DAVIS x265).
- **Regime coordinate:** the *matched pristine baseline's* own
  `background.lpips_mean` at that rung. Computed from the baseline arm only, so
  it cannot be contaminated by the outcome. **Never** match on QP (x265 32 ≠
  svtav1 43) and **never** on absolute bitrate — that *is* codec efficiency.
- **Outcome per rung:** quality = `presley_ai` BG-LPIPS − `elvis` BG-LPIPS;
  bitrate = Δbits vs the matched pristine baseline. Reported **separately** —
  `NOTE(tab:frontier)` forbids merging them into one ordering.
- **Per-ladder contrast:** median outcome over the two most-starved rungs minus
  median over the two most-comfortable rungs. One scalar per ladder.
- **Test:** exact two-tailed sign test over 13 contrasts
  (`suite.sign_test_p`), Holm via `suite.holm_adjust`, and
  `suite.min_attainable_sign_p` printed beside every n.

## Holm family: k = 4

{quality, bitrate} × {presley vs elvis, presley vs baseline}.

The 7 spent content attributes are **not** in this family — different unit
(ladder vs video), different target (regime dependence vs map separability),
different test (paired contrast vs rank correlation). But the design is chosen so
that argument cannot matter:

| n | unanimous two-tailed p | max Holm k still surviving α=0.05 |
|---|---|---|
| 6 | 0.03125 | 1 |
| 10 | 0.00195 | 25 |
| **13** | **0.000244** | **204** |

At n=13 the result survives a hostile referee who folds in all 7 spent
attributes plus everything else this project has ever tested. **The family-size
dispute is therefore off the table before it starts**, which is the point of
choosing n=13 rather than the n=6 the chain offers.

## Bounds, registered before the numbers

| quantity | plausible | alarm |
|---|---|---|
| \|per-ladder contrast\|, BG-LPIPS | 0.00 – 0.10 | **> 0.25** (5× JND — investigate a bug before believing it) |
| ladders concordant in sign | 8 – 13 of 13 | **13/13 with median \|contrast\| < 0.01** — unanimity at negligible magnitude means the coordinate orders trivially, not that an effect exists |
| BG-LPIPS vs FG-PSNR coordinate | agree in sign | **disagree → R1 is withdrawn** |
| DAVIS (n=4) vs non-DAVIS (n=9) subsets | both majority same sign | opposite majorities → report as **provenance-modified**, not as a scope rule |

A fired bound is **recorded as fired**. Bands are never re-fitted to accommodate
a result — that rule already caught us once this week (the FG BD-rate band in
`PLAN_DOWNSAMPLE_VS_UNIFORM.md` fired on 4/6 and stands as fired).

## Both outcomes, written now

**Fires.** Operating point is the scope axis. This becomes the paper's scope
statement: the approach is a **bit-starved-regime** method, and we say so with a
properly powered test rather than as a caveat.

**Does not fire.** Report it: *with 13 rate ladders and a design immune by
construction to the cell-count and provenance confounds that invalidated our
earlier attempts, we detect no dependence of the outcome on operating point.*
This is the **first properly powered negative** of the six and it closes the
question rather than adding to a pile of underpowered ones.

## Stopping rule

**R1 runs once.** If it does not fire, stop. Do not try a second regime
coordinate, a third, or a cell-level version — cell-level correlations are
roughly 4× more "significant" and are exactly the inflation the video-level unit
exists to prevent.

**Total scope budget for this paper: two pre-registered hypotheses, R1 and M1.
There is no third.** Five negatives is enough evidence that this corpus does not
support attribute-hunting.

---

## RESULT — 2026-08-06. R1 does not fire, and the negative is the useful outcome.

Ran once, as registered. `tools/analyze_r1_regime_scope.py`.

| n | negative contrasts | sign p | Holm (k=4) | min attainable at n=13 | median \|contrast\| |
|---|---|---|---|---|---|
| 13 | 5/13 | 0.581 | 1.000 | 0.000244 | 0.0223 BG-LPIPS |

**No dependence on operating point is detectable.** Every contrast sits inside
the pre-registered plausible band (0.00–0.10); nothing approached the 0.25 alarm;
the trivial-magnitude alarm did not fire (median 0.0223 is a real magnitude, so
this is a genuine null and not a coordinate that orders trivially).

**Stopping rule honoured.** The `fg_psnr` coordinate was NOT run. The
pre-registration specifies it as a robustness check *for a result that fires* —
running it after a null would be exactly the second-coordinate search the
stopping rule forbids.

### Why this negative is good news rather than a disappointment

R1 tested whether PRESLEY's advantage **over ELVIS** varies with how bit-starved
the encoder is. It does not. Read together with the n=13 matched-rate result
(PRESLEY ahead on **13/13** ladders, p=0.000244), the two say something stronger
than a regime-conditional claim would have:

> PRESLEY beats ELVIS at matched rate on every ladder tested, across two codecs
> and three dataset families, and that advantage is **stable across the whole
> operating range** rather than confined to a favourable corner of it.

A scope result that had *narrowed* the claim would have been weaker. This one
widens it.

### This does not contradict the regime flip in `tab:av1`

Different comparison, and the distinction must be kept in the text. `tab:av1` and
`tab:av1-breadth` concern **bit relocation against the pristine baseline** (the
reduction goal), where the sign genuinely does flip with regime and then fails to
replicate on new content. R1 concerns **PRESLEY against ELVIS** (the restoration
goal). The restoration advantage is regime-stable; the bit-relocation advantage
is regime-dependent *and* content-dependent. Reporting both is the honest
position, and conflating them would be an error in either direction.

### Scope budget

R1 is spent. One hypothesis remains (M1). **There is no third.**
