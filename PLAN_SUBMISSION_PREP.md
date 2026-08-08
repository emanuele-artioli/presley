# Implementation plan — PRESLEY to a submittable TOMM paper

Written 2026-08-07. Supersedes the *scope* of `HANDOFF_PAPER_REWRITE.md`; that
document's factual state (numbers, rules, tooling) still holds and is still
worth reading. Where this plan and the handoff disagree, this plan wins — the
handoff assumed "structural rewriting, no new experiments", and the brief has
since asked for four things that need runs.

Verified before writing: `tools/render_paper.sh` builds the PDF (45 pages,
2026-08-07), paper repo `main` @ `1bfd869`, code `main` @ `129827a`.

---

## 0. Answer to the question that was asked before deciding

### What "a run with non-empty `invariant_failures` is never citable" means

`src/presley/invariants.py` holds five machine-checkable versions of the
methodology rules. `presley-run` calls `check_result()` on every finished run
and writes the verdict — a list of strings, empty when clean — into the run's
own `result.json` under `invariant_failures`. The five checks:

1. **Metrics present and real.** FG/BG/overall PSNR exists, is finite, is
   positive. (`psnr_mean: true` would otherwise pass as 1.0 — `bool` is
   explicitly rejected.)
2. **Bitrate accounting describes the payload.** `actual_bitrate_bps` must
   agree within 1% with what `transmitted_size_bytes` implies. For
   elvis/presley_ai the on-disk output file is the *decode-side* restored
   video, tens of MB; if the rate axis silently used it, every RD claim would
   be measuring the wrong thing.
3. **Fixed-QP mandate.** A degrading component (`elvis`, `presley_ai`) running
   under VBR is marked uncitable, because under a bitrate target degradation
   cannot free bits — 25/25 matched VBR pairs encoded to *more* bits than
   pristine, zero counterexamples.
4. **Restoration did not make the background worse** than the degraded input
   it received, judged on LPIPS, never PSNR.
5. **Output not saturated.** A restorer that diverges numerically writes no
   NaN — the garbage lands at 0/255 and the run reads as merely disappointing.
   NAFNet did exactly this on bike-packing: 4.31% of pixels clipped against
   0.03% in its input, 6.79 dB of BG-PSNR destroyed, and nothing flagged it.

So the rule is not a style preference and it is not extra work: it is a field
that is already written into every result, and "never citable" means *the paper
must not quote a number from a run whose list is non-empty*. It exists because
each of the five failures produces a perfectly well-formed `result.json` whose
numbers mean something other than what the paper would claim from them.

**Recommendation: keep it, unchanged, and it costs this plan nothing.** Every
run currently cited already has an empty list. The one thing worth adding is
that the check has a third state the rule does not name — a run written before
a check existed carries *no* `invariant_failures` field at all, and a missing
verdict currently reads as "fine". `presley-invariants` backfills those. I will
run the backfill once before the paper's provenance appendix is generated, so
"every cited run is clean" is a checked statement rather than an assumed one.

---

## 1. Decisions taken on the eleven points

| # | Brief | Decision |
|---|---|---|
| 1 | Appendix use | **Yes, ~4 pages.** Not as an overflow bin — as *"What we got wrong"*, plus provenance the main text cannot carry. Detail in §2.6. |
| 2 | Figures over tables, JSON in descriptions | **Yes,** via ACM's `\Description{}` — the accessibility field, which is invisible on the page, mandatory for ACM, read by screen readers, and machine-parseable. Zero page cost. Detail in §2.4. |
| 3 | Re-check equations/algorithms against code | **Yes, and this is now Wave 1 priority** — I have already found that the method section's core equations do not describe the code. Detail in §2.2. |
| 4 | Retire JND in favour of significance tests | **Yes**, with one addition the brief does not name: claims of the form *"X and Y are indistinguishable"* need an **equivalence test**, not a significance test. Detail in §2.3. |
| 5 | `invariant_failures` | Explained above. **Keep.** |
| 6 | CLAIM/NOTE staleness | Treated as advisory. **CLAIM anchors are kept** (they are the provenance chain to `results/<hash>`); **NOTE wording constraints are verified against the data before being obeyed or overridden.** |
| 7 | `\rev`/`\del` | **Retire now**, as Wave 0, before any other paper edit. Tag the tracked-changes state first so it is recoverable. Detail in §2.1. |
| 8 | Speed by pipeline step | **Yes** — needs code (per-stage instrumentation across all four components; today only `presley_ai` splits out selection, and nothing splits out preprocessing) and a controlled batch. Detail in §3.1. |
| 9 | Bitrate at resolution | **Yes** — a real ladder. 360p/540p/720p/1080p. Detail in §3.2. |
| 10 | DiffBIR | **Yes, wire it**, at low priority and gated on the weights downloading. Reasoning in §3.4. |
| 11 | Don't present the selection score as a bad component | **Agreed, and it does not require softening anything.** The positive framing is already supported by data in hand, and there is a concrete fix worth attempting. Detail in §3.3 — this is the most important decision in the plan and the one most likely to change the paper's argument. |

---

## 2. Paper work (no GPU; can start immediately)

### 2.1 Wave 0 — retire revision tracking (must run alone, first)

119 `\rev{}` and 16 `\del{}` spans across `main.tex`, three section files and
`response_letter.tex`. `\rev` is *unwrapped* (content stays); `\del` is
*deleted* (that text was removed for the referee and only remains visible as a
tracking artefact).

- Tag the paper repo first: `git tag archive/tracked-changes-2026-08-07` and
  push. This is the version to diff against when the previously-submitted
  manuscript arrives.
- Brace-matched unwrap script, not a regex — several `\rev{}` spans contain
  nested braces, `\texttt{}`, and `$...$`.
- Remove the macro definitions from `main.tex`'s preamble.
- Render and confirm the page count moves by less than a page (it should be
  neutral; `\del` removal will shave a little).

**`response_letter.tex` is frozen at this point**, not maintained through the
rewrite. It cites tables by label and the rewrite renames or merges most of
them; keeping it in sync incrementally guarantees drift, and a letter that
contradicts the manuscript is the worst failure mode available. It gets rebuilt
in one pass at the end of Wave 3, against final labels.

This wave touches every file, so nothing else edits the paper until it lands.

### 2.2 Equations and algorithms vs. the code — this is a correctness problem

I checked `sections/presley.tex` against `src/presley/preprocessing.py` and
`src/presley/degradation.py`. The method section does not describe the code.
Five discrepancies, in descending severity:

1. **Equation 3 (importance) is wrong, and contradicts its own paragraph.**
   The paper says background blocks get their sign inverted
   (`R = −C'` for background). The code
   (`preprocessing.py:667`) *multiplies background scores by 10*. These are
   different functions with different orderings. The prose immediately below
   Equation 3 says "all background blocks (positive scores)" — which matches
   the code and contradicts the equation printed directly above it.
2. **Equation 4 (downsampling factor) describes code that has never run.**
   `DF = ⌊R⁻¹·maxDF⌉` is a graded multi-level scheme. Every reported
   experiment runs `levels=1`, where the strength map is `round(score) ∈ {0,1}`
   and the factor is the scalar `scale=0.5` — a single 2× downsample. The
   graded path exists (`levels>1`) and was measured once, as the S1b oracle
   probe, and did not work. The paper's headline degradation equation is
   describing the arm the paper retired.
3. **Equation 1 has the wrong temporal index.** The code pairs spatial
   complexity of frame *n* with temporal complexity of frame *n+1*
   (`spatial_3d[:-1] + temporal_3d[1:]`), and the final frame uses spatial
   only. The paper writes `α·S_n + (1−α)·T_n`.
4. **Smoothing is applied at the wrong point in the pipeline.** Equation 2
   smooths *complexity*, before the mask step. The code smooths *importance*,
   after the background boost. Algorithm 1 also leaves `C'` undefined for
   frame 0 (the `if n>0` guard has no else branch), then returns it.
5. **The actual selection rule is not in any algorithm.**
   `select_removal_mask_global` — global top-k under a budget matched to the
   per-row default, a 5×5 Gaussian blur so selected blocks cluster (cheaper for
   the codec), and a *hard* foreground exclusion applied after the blur — is
   PRESLEY's real selection mechanism and appears only as one prose sentence.
   The scores are also normalised to [0,1] at the end, which the paper never
   states but which Equations 4 and 5 both silently depend on.

**Work:** rewrite Equations 1–4 and Algorithm 1 to match the code; add a new
algorithm for budgeted global selection with hard foreground exclusion; add a
short equation for the bit-plane-packed side channel (`sidechannel.py`), which
the paper claims a cost for (4.19 kbps) without saying how it is computed; keep
Equation 5 (QP mapping) but mark it explicitly as the retired ablation it now
is. Add a test (`tests/test_paper_equations.py`) that asserts the numbered
constants in the method section against the code's defaults, so this cannot
drift again.

Also worth adding, and cheap: the **hard foreground exclusion is a measured
design result, not a hack**. Soft protection (BG ×10) failed on bmx-trees — FG
mean score 0.127 vs BG 0.113, top-k degraded 12.8% of foreground blocks, FG-PSNR
collapsed 3 dB. That is currently a code comment. It belongs in the paper.

### 2.3 Retire JND; replace with tests that match the claim being made

The JND constants (0.5 dB PSNR, 0.05 LPIPS/DISTS) are not cited to literature —
there is no established JND for LPIPS — and a TOMM referee will ask. Agreed:
they go.

But "replace with significance tests" is only half a replacement, because the
paper uses JND for two opposite purposes and only one of them is a significance
question. Sorting the ~60 JND-resting statements into three buckets:

- **(a) "X beats Y", currently gated by JND** → exact two-tailed sign /
  Wilcoxon, Holm over the candidate family with losers counted, plus a
  bootstrap CI on the effect size. `src/presley/suite.py` already does all of
  this. Some of these claims will get *stronger* — that is the expected and
  wanted outcome of the brief.
- **(b) "X and Y are indistinguishable", currently "within JND"** → a
  significance test cannot support this; a non-significant result is not
  evidence of equivalence. These need **TOST equivalence** against a declared
  margin, reported as "equivalent within ±δ, 90% CI [·,·]". The margin is then
  honest about what it is: *a reporting threshold we declare*, not a perceptual
  constant we cite. `suite.py` already emits an `EQUIVALENT` verdict with a 90%
  CI in at least one place, so the machinery is partly there.
  **This bucket is where the risk lives:** the paper's foreground-protection
  claims live here, and replacing "FG is within JND" with a bare significance
  test would convert a good claim into "FG differs significantly by a tiny
  amount". TOST is the correct instrument and keeps the claim intact.
- **(c) descriptive "1.42× JND"-style numbers in tables** → raw deltas with
  CIs. No threshold language at all.

**Work:** enumerate every JND-resting statement (there is a mechanical grep
starting point), classify, run the replacement test, rewrite. Where a test
changes the verdict, the claim changes and the CLAIM anchor is updated. Extend
`suite.py` with TOST if the existing equivalence path does not cover bucket (b)
generally. `src/presley/compare.py`'s JND table stays in the *code* as an
internal screening heuristic — it is useful for agents deciding what to look at
— but no paper-visible claim rests on it, and the hard rule in
`research-log/hard-rules.md` is amended to say so.

### 2.4 Figures, and the machine-readable half of them

Preference inverted from the handoff: **figures by default, tables only where a
table genuinely beats a figure** — exact provenance values, or heterogeneous
columns that do not share an axis. Rough target: ~12–14 figures, ~6–8 tables,
down from 8 figures and 32 tables.

Every figure is produced by a script under `tools/`, and every script emits
three artefacts from one run:

1. `Figures/<name>.pdf` — the figure,
2. `Figures/<name>.json` — the underlying values, semantic keys, flat arrays,
3. `Figures/<name>.desc.tex` — a `\Description{}` block: **one plain-language
   sentence stating what the figure shows, then the compact JSON.**

`\Description{}` is ACM's alt-text field. It is mandatory for ACM submissions,
does not render on the page, is what a screen reader reads, and is trivially
parseable. It costs zero page budget and serves a blind reader, a referee's
accessibility check, and an agent reading the source, from one artefact. The
plain sentence comes first so a screen reader is not reading raw JSON at a human.

A test regenerates each figure and diffs the JSON against the committed one, so
a figure that no longer matches `results/` fails CI rather than sitting stale in
the PDF.

Figure list (★ = needs a run from §3; others are buildable from committed data):

| # | Figure | Replaces |
|---|---|---|
| 1 | Pipeline overview (exists) | — |
| 2 | Matched-rate BD-rate, 13 ladders, sorted bars, all negative | `tab:ratematched-n13` |
| 3 | Regime reversal RD ladders (exists) | — |
| 4 | M1 scatter: superblock complexity vs post-restoration damage | new — the paper's best idea currently has no picture |
| 5 | Selection diagnosis: capture ratio vs oracle, against the random null | new |
| 6 | Three-axis frontier: quality / bitrate / restoration cost | `tab:frontier` + prose |
| 7 | Cost scaling with resolution | part of `tab:speed-scaling` |
| 8 ★ | Speed by pipeline step, stacked, per arm | new (§3.1) |
| 9 ★ | Bitrate vs resolution ladder | new (§3.2) |
| 10 | Restorer catalogue: BG-LPIPS per backbone with CIs | `tab:inpainters`, `tab:conditioned`, `tab:conditioned-twins`, `tab:conditioned-stream-diffvsr`, `tab:instantir-kill`, `tab:goal2`, `tab:goal2-breadth` (7 → 1) |
| 11 | Dataset breadth: per-clip outcome | `tab:breadth`, `-ext`, `-ext-presley`, `-ratematched` (4 → 1) |
| 12 | Selection ablation: effect sizes with CIs | `tab:ablation`, `tab:graded`, `tab:graded-oracle`, `tab:budget-knee` (4 → 1) |
| 13 | Qualitative crops: pristine / degraded / restored | new — a generative-restoration paper with no visual example is a real gap, and referee 2 asked |
| 14 ★ | Restorability-aware selection vs. current, matched rate | new (§3.3), conditional on the result |

Tables that survive: evaluation setup; the AV1 breadth negative (the exact
numbers matter and are the credibility anchor); selective-vs-uniform
downsampling; "what we retired and why"; and two provenance tables in the
appendix.

### 2.5 Structure and the page budget

Target split at ~467 words/page: Introduction 12%, Background 12%, Method 26%,
Evaluation 40%, Conclusions 8%. Evaluation goes from 66% of the body to 40% —
roughly a 68% cut — and it must come out as **whole results, merged into the
figures above**, not as compressed prose. The abstract goes from 504 words to
~200 (it currently contains a results section).

Reorganise the evaluation around the three goals rather than around the order
experiments happened to land in. `tools/paper_metrics.py` after every
structural edit; `tools/render_paper.sh` before any page-count claim.

### 2.6 The appendix (~4 of the 5 permitted pages)

The brief is right that the appendix is a poor place for things that belong in
the paper, and right about what it *is* good for.

- **A — "Methodological pitfalls" (~2 pp).** VBR laundering (25/25, zero
  counterexamples); the FG-metric bbox artefact (a "foreground" box averaging
  76% of frame area against a true foreground of 15%, which is why FG-VMAF and
  FG-FVMD are banned); pooled-vs-within-run dispersion reported as within-run;
  selection leakage inflating a correlation from 0.506 to 0.684; one-tailed
  p-values on a direction read off the data; a timing corpus mixing two device
  populations after a silent CPU fallback; a bitrate-saving claim retracted as
  a VBR-baseline overshoot; a runner that exits 0 when every experiment in a
  wave fails. Each as *the rule that prevents it*, with the evidence.
  This is unusual, useful, and cheap — the material is already written in
  `research-log/`, and it converts the project's most embarrassing history into
  the most reusable thing in the paper.
- **B — Provenance (~1.5 pp).** Every cited result hash against its claim, and
  the confirmation that all of them carry an empty `invariant_failures`.
- **C — Pre-registrations and fired bounds (~0.5 pp).** The two pre-registered
  analyses, what bounds fired, and how each was closed. `git log` shows both
  documents committed before the runs they govern; that ordering must not be
  destroyed by any history rewrite.

Everything else stays in the main text.

---

## 3. Experiments (GPU; runs in parallel with §2)

Per the host rule, plausible ranges are stated **before** the numbers are
looked at. A result outside its range is an alarm to investigate, not a finding
to report.

### 3.1 Speed by pipeline step

Today only restoration is timed cleanly, and only for `presley_ai` and `elvis`.
Preprocessing (frame extraction, EVCA complexity, UFO masks) happens before the
clock starts and is invisible. Selection, degradation, side-channel packing and
encoding are all folded into one `encoding_time_seconds`, and
`selection_time_seconds` exists on `presley_ai` only and has **populated zero
runs** (I checked: 0 of 1159 results carry it).

**Code:** add a `stage_times_seconds` dict to every component
(`baselines`, `roi`, `elvis`, `presley_ai`) with a shared vocabulary:
`preprocess_frames`, `score`, `select`, `degrade`, `encode`, `sidechannel`,
`decode`, `restore`, `composite`. It is an *output* field, so it changes no
experiment hash and invalidates nothing.

**Campaign:** repeat trials of one config collapse to one hash, so a repeat
cannot go through `presley-run` — and `_`-prefixed keys are excluded from the
hash, so that escape hatch does not work either. Follow the existing precedent:
`tools/timing_campaign.py` replays the restoration step from a finished run's
own artefacts into a scratch directory, records the device on every trial, and
never touches `results/` or the DB. Extend the same harness to replay whole
pipelines, stage by stage, 3 trials per cell, on a pinned GPU.

Cells: 1 baseline codec × 2, 2 ROI arms, 2 elvis arms (E2FGVI, ProPainter),
2 presley_ai arms (downsample+Real-ESRGAN, freeze+ProPainter), × 6 videos.

*Bounds.* Restoration dominates: Real-ESRGAN ≈22–24 s / 82 frames at 360p,
ProPainter 39–77 s. Encoding at 360p: 2–20 s. Selection: numpy top-k on an
80×45 grid × 82 frames — **0.05–5 s**, and my expectation is under 1 s. EVCA +
mask preprocessing: 5–120 s, reported separately as a cacheable one-off.
**Alarms:** selection costing more than encoding; any stage exceeding
restoration; a trial resolving to CPU (never averaged with GPU trials — that
mixture is the original defect that made the old timing corpus unusable).

### 3.2 Bitrate at different resolutions

Cache already holds `bear` and `camel` at 360p/540p/720p/1080p, and `dog`/`pigs`
at 1080p. Preprocessing the remaining (video, resolution) cells is EVCA + masks
— cheap, no restoration.

Ladder: 6 videos × {360p, 720p, 1080p} × 4 fixed-QP rungs × {pristine baseline,
presley_ai downsample+Real-ESRGAN} = 144 runs. 540p added on bear/camel only,
where it is free. BD-rate per (video, resolution), then a sign test across
videos *within* each resolution — n=6 is the significance floor, so this is a
claim, not a description.

*Bounds.* At 360p the starved saving is −13.8…−16.1% (bear/camel). At higher
resolution the background is relatively more compressible and the restorer has
more to work with, so I expect the saving to hold or improve: **plausible
−35%…+10% per rung.** Outside that band is an alarm — most likely candidates
are a mask that did not scale, or a QP ladder that is no longer starved at
1080p (the whole effect is regime-dependent, and "starved" is a
resolution-dependent QP). **The QP rungs must be recalibrated per resolution**,
not copied from 360p; skipping that recalibration would produce a clean-looking
result that means nothing.

Restoration cost at 1080p is ≈7.5× the 360p cost, so budget ~165 s/run for the
presley arm — the ladder is an overnight job, not an interactive one. Checkpoint
hourly; launch detached.

### 3.3 The selection component — reframe, then try to fix it

**This is point 11, and it is the most consequential item in the plan.**

The current text says the selection score "preferentially degrades the blocks
that survive restoration worst". That is true and it should not be softened —
referees have already seen it, and deleting it would trade the paper's best
credibility asset for nothing. But it is currently the *headline* of the
section, and it does not have to be. Two things are already true and already
measured:

- **On the axis it actually models, the score is close to optimal.** Against a
  leave-one-superblock-out bit oracle it captures 0.833 of the recoverable bits
  against a random-selection null of 0.402, and since the oracle's top quarter
  is 30.1% of total bitrate, *the entire remaining headroom on the cost axis is
  worth about 5% of the bitrate.* That is a positive, bounded result about a
  component, and the paper currently states it inside a paragraph about what is
  missing.
- **The hard foreground exclusion is a measured necessity**, not an
  implementation detail (see §2.2).

So the section is reordered: the score is a near-optimal *cost* estimator with a
measured bound on its remaining headroom; the second axis is identified,
measured, and shown predictable at transmit time (ρ +0.506, same direction on
120/120 runs, five features declared in advance); and the sign of that
prediction explains the α/β null. That is a contribution, stated positively,
with the honest part intact and subordinated to it.

**Then attempt the fix.** Everything needed exists: `probe_oracle_bits` gives
exact marginal bits on 8 videos, `results/block_damage_s1b.npz` gives
per-superblock post-restoration damage on 120 runs, and the five transmit-time
features are already extracted.

- Fit a damage predictor **D̂** on the declared features, on held-out videos.
- Add a selection rule that ranks by **predicted bits / predicted damage**
  instead of by complexity alone — a config-level arm, so the existing rule
  stays the default and no existing hash moves.
- Run it against the current rule at matched rate, n≥6 videos, two-tailed,
  Holm family sized for every ranking variant tried including losers.

*Bounds, and an honest warning.* Both terms are monotone in complexity, so the
ratio may be near-flat and selection may collapse toward random among background
blocks — in which case the corrected rule **loses**. Plausible BD-rate of the
corrected rule against the current one: **−20%…+25%**. I am not confident of the
sign. Using the full five-feature damage model rather than spatial complexity
alone is what gives it a chance, because that is the only way the denominator
carries information the numerator does not.

**If it wins**, the paper gains a fix and Figure 14. **If it loses**, the paper
gains a properly powered negative that closes the loop on its own diagnosis —
"we identified the missing term, built the predictor the diagnosis implies, and
it does not recover the loss at matched rate" — which is a stronger and more
finished statement than the current open-ended "this is left as future work".
Neither outcome weakens the paper, and that is why this is worth the GPU time.

### 3.4 DiffBIR

Not integrated anywhere (grep finds it only in the queued-experiments doc). The
prior handoff said "ASK before wiring"; the brief authorises it conditionally,
so: **wire it, at the back of the queue.**

Reasoning: the restorer-comparison claim is currently *"no backbone separates
from Real-ESRGAN"*, resting on BSRGAN, Real-HAT-GAN, Stream-DiffVSR, InstantIR
and NAFNet. DiffBIR is the strongest blind-restoration diffusion model not yet
tried, and it is the last open `HOLE` in the manuscript. Adding it makes the
catalogue's negative harder to dismiss as an artefact of which models were
picked. I do not expect it to win — InstantIR was killed and Stream-DiffVSR
lost — so this buys robustness on an existing claim, not a new one, which is
why it is last.

Gated on the weights being fetchable from this host. If they are not, the HOLE
is closed in the text as a scoped deferral with the reason stated, which is a
legitimate outcome and costs nothing.

---

## 4. Waves

Paper edits are partitioned by file, because the paper repo is a single
non-worktree checkout shared by every agent. Code work goes in worktrees.

**Wave 0 — serialised, ~1 session.** Retire `\rev`/`\del` (§2.1). Tag first.
Nothing else touches the paper until this lands. Freeze `response_letter.tex`.

**Wave 1 — parallel, no GPU.**
- *W1a* Equations and algorithms vs code (`sections/presley.tex` +
  `tests/test_paper_equations.py`). §2.2.
- *W1b* JND → significance/equivalence audit and rewrite
  (`sections/evaluation.tex` + `src/presley/suite.py`). §2.3.
- *W1c* Figure infrastructure: the three-artefact convention, the CI check, and
  figures 2–7 and 10–13 from committed data. §2.4.
- *W1d* Structural cut plan and the appendix skeleton. §2.5, §2.6.

**Wave 1′ — parallel with Wave 1, GPU.**
- *W1e* Stage-timing instrumentation + campaign. §3.1.
- *W1f* Resolution-ladder preprocessing, then the ladder itself. §3.2.
- *W1g* Damage predictor + restorability-aware selection arm. §3.3.

**Wave 2 — after 1 and 1′ report.** Fold the new results in (figures 8, 9, 14),
execute the evaluation cut against the real page count, rewrite the abstract,
rebalance introduction/background/method. DiffBIR (§3.4) runs here if the
weights landed.

**Wave 3 — serialised.** Rebuild `response_letter.tex` against final labels;
regenerate the referee-safe submission copy
(`tools/make_submission_copy.py` strips ~117 non-CLAIM markers — necessary,
since the `.tex` source goes to TOMM and the markers contain retraction
histories and notes addressed to agents); final render, final
`paper_metrics.py`, final page count.

## 5. Risks

- **The 23-page cut is the critical path and must not wait on GPU.** Waves 1
  and 1′ are deliberately concurrent for this reason.
- **Bucket (b) of the JND retirement is where a claim could be lost.** If TOST
  cannot support "foreground is unharmed", that is a real finding and must be
  reported as one — but I want it surfaced early, not discovered in Wave 2.
- **The corrected selection rule may lose.** Bounds are stated above; the
  fallback framing is written and does not depend on the outcome.
- **The resolution ladder can silently measure nothing** if the QP rungs are
  not recalibrated per resolution. Called out in §3.2 as the first thing to
  verify.
- **Equation fixes may invalidate prose elsewhere.** The sign-convention error
  in Equation 3 in particular is referenced by later text; W1a must grep for
  dependants rather than editing in place.
