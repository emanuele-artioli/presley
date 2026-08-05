# Plan C — the tradeoff surface, and the integrity fixes it needs first

**Supersedes `docs/PLAN_OPERATING_MAP.md`.** (Copy of the session plan, checked in so the next session finds it without the plans directory.) Written 2026-08-03 after Waves 1,
2A, 2B, 2C reported and their claims were audited.

> ## Execution status — updated 2026-08-05
>
> | | workstream | status |
> |---|---|---|
> | **W1a** | fixed-QP kvazaar baselines | ✅ **DONE** — rate claim refuted, quality claim survives calibrated |
> | **W1b** | disclosure gap (camel/dog/pigs) | ✅ **DONE** — gap closed entirely, not just disclosed |
> | **W1c** | SVT-AV1 rate deltas | ✅ **DONE** — withdrawn; quality null unaffected |
> | **W1d** | commit 2B report + post-check | ✅ **DONE** — and it found a bug in the map tool |
> | **W2** | tradeoff surface, matched pooling | ✅ **DONE** — tool + 12 tests |
> | **W3a** | record acquisition conditions | ✅ **DONE** — prerequisite only |
> | **W3b–d** | controlled timing campaign | ⬜ **NOT STARTED** — now the only thing blocking W7 |
> | **W4** | significance completion | ✅ **DONE** — `freeze+propainter` closed at n=10, p_Holm 0.0254; three significant quality comparisons, none on bitrate. Only `ac_truncate` (n=7) left, one video from the floor |
> | **W4b** | sweeps already on disk | ✅ **DONE** — graded vs uniform downsampling answered from existing runs |
> | **W5** | content axis round 2 | ✅ **DONE** — negative, as pre-registered; BG-motion fired and was withdrawn |
> | **W6** | naming / hygiene | ✅ **DONE** — documented, deliberately not renamed |
> | **W7** | paper restructure | ⬜ **NOT STARTED** — blocked on W3/W5 |
>
> **Everything is committed; nothing is pushed.** Work since 2026-08-03 is on
> `claude/after-wave-2b-cosmic-sunbeam-1330c8`, which also **merges in the two
> branches whose tools main could not see** — `claude/operating-map-implementation-34a243`
> (W1/W2) and `worktree-agent-ac71fed8395e1fd74` (Wave 2A). Both merged cleanly.
> That invisibility is why W4's significance analysis was re-derived by hand and
> never committed; it is now `tools/analyze_w4_significance.py`, and it
> reproduces the published table exactly. The paper repo is still at `6408a0c`
> with **5 commits awaiting an Overleaf push** the human has to run.
>
> ### 2026-08-05 session, in one line each
>
> * **W4 closed.** `freeze+propainter` n=6→10, 10/10, p_Holm 0.0254. Three arms
>   now lose to `downsample+realesrgan` on quality on every video tested
>   (11/11, 10/10, 10/10); **none of the three bitrate comparisons is
>   significant and their directions disagree.** All six bounds held.
> * **W5 negative.** BG-motion fired (rho 0.663, p_Holm 0.0203) and was withdrawn
>   by both pre-registered checks. It is A1 relabelled — background blocks
>   outnumber foreground ~10:1 — so the FG/BG decomposition is not one in
>   practice. Duration is not the coverage confound; FG-fraction refutes again.
> * **W4b, no GPU.** Graded downsampling beats uniform-3 on quality 8/8 and
>   loses bitrate 0/8 (p_Holm 0.031 each, post-hoc, n at the floor) — the data
>   `HOLE(sec:downsample-vs-uniform)` needs. `blur_kernel` is a bad lever;
>   `ac_keep` is a real two-JND ladder; `mask_source` moves quality by a
>   thirteenth of a JND.
> * **A sixth instance of the one pattern**, this time in a tool written the same
>   day: `None` was both a legitimate parameter value and the no-winner signal,
>   so an 8-of-8 result displayed as 1 of 8. **When absence is data, absence can
>   no longer double as the error signal.**
>
> ### What closed each item
>
> **W1a — the plan's central prediction was right, and stronger than predicted.**
> 17/17 runs citable (verified against the DB, not the runner's exit code).
> Against a like-for-like fixed-QP baseline: **Δbits median −0.1%** (range
> −15.4…+17.3), **ΔFG median +0.51 dB** (16/17 positive), **ΔBG median −0.53 dB**
> (16/17 negative). The published "24.9–40.2% fewer bits" was the VBR baseline
> overshooting its target by 30–45%. Root cause was deeper than a config error:
> `baselines.py` had fixed-QP branches for x265 and svtav1 and **none for
> kvazaar**, so the experiment could not have been specified correctly.
> `encode_video_kvazaar_qp` added. Full table and bounds:
> `docs/W1A_KVAZAAR_FIXEDQP.md`. Three bounds fired, all recorded, including one
> **mis-set by me** — the Δbits band ignored that both arms search an *integer*
> QP, and one step is worth 10–15% of bitrate, so ±15% is the search's
> granularity floor rather than a defect.
>
> **W4 — `blur+nafnet` crossed; there are now two significant comparisons where
> there was one.** n=11, 11/11, **p_Holm 0.0137** (it sat at 0.0508 — two videos
> bought it). With `blackout+propainter` (10/10, p_Holm 0.0254) the quality
> ordering is established twice over. **On bitrate nothing is significant and the
> direction is not uniform** — blackout wins the rate axis (baseline wins 1/10) at
> p_Holm 0.279, blur is a coin flip at 5/11. `docs/W4_SIGNIFICANCE.md`.
>
> **W2 — matched pooling matters more than expected.** Naive pooling puts
> `blackout+propainter` at −10.0% bits; matched puts it at **−25.5%** (3 cells, 2
> videos). Naive pooling was badly *understating* the rate arm. Also corrected my
> own conflict count: buying speed at equal quality costs bitrate in **8 of 12**
> cells (median 5.6 pp, up to 22.8), **not the 15 of 16 stated in this plan's
> Context section below** — that scan did not collapse duplicate configs and
> counted swaps from an arm to itself.
>
> ### Four bugs found in our own analysis this session, all one shape
>
> Each was a correct number, or a correct silence, carrying a meaning it had not
> earned. This is the pattern to watch for, not four unrelated slips.
>
> 1. **Control matching** used `(component, transport, block_size,
>    shrink_amount)`, which does not distinguish degradation *strength*.
>    `dancing`@QP43 has three `blur` runs at `blur_kernel` 7/15/31 collapsing to
>    one key; the dict kept the last, so arms were scored against **another run's
>    damage while keeping their own bitrate**. That is how a rate–damage ladder
>    acquires a positive slope. Corrected: **3** deployable off-ladder arms, not
>    10. Fixed by keying on the whole config minus the fill.
> 2. **Silent arm dropping.** W4's 8 runs landed clean and `n` did not move: a
>    fresh run carries only `overall` LPIPS until `--backfill-lpips`, and
>    `score_arms` skipped them without a word. A new arm does not error, it just
>    never appears — indistinguishable from the runs having failed. Now counted by
>    reason (which surfaced 34 runs with no usable baseline, previously invisible).
> 3. **A backfill loop that failed on all 8 hashes and exited 0.**
>    `python -m presley.evaluation` is a package; the entry point is the
>    `presley-evaluate` console script.
> 4. **A mis-aimed caveat of my own**, caught before it hardened: the `tennis`
>    note blamed the union-bbox FG pathology, but `foreground.psnr_mean` is
>    already a true masked metric (`_masked_psnr` indexes `ref[mask]`). Corrected
>    in the paper; the finding itself was unchanged.

---

## Context

The operating-map framing is dead. Three independent lines killed it: 18 of 19
quality-first separable cells name the same arm; no content attribute predicts
transport choice; and the one target that mattered (T1 — *which* arm wins)
could not be tested at all, because a near-constant outcome is not predictable
by anything. "For a given kind of content at a given quality there is a best
choice" is not supported.

But the audit of our own claims found something better, and it is the user's
framing: **the three axes a deployer actually cares about — quality, bitrate,
speed — genuinely conflict, and we can quantify the frontier between them.**
The evidence that they conflict is hard: in **8 of 12** cells where the
rate-first winner has a faster arm at indistinguishable quality, that faster arm
**gives up 1.2–22.8 percentage points of bitrate saving** (median 5.6). Only 4
swaps are free. There is no single winner, and that is the contribution.

*(This paragraph originally read "15 of 16, only 1 free". Corrected 2026-08-03
once the duplicate-config aggregation rule was applied consistently — the first
scan counted swaps from an arm to itself. The conclusion is unchanged; the
count was wrong.)*

Before any of that can be published, four of our own numbers have to be fixed.
The audit found them; they are not reviewer-proof as they stand.

### Corrections to the record (all verified this session)

| claim | status |
|---|---|
| "`tab:roi`: kvazaar ROI saves 24.9–40.2% bits" | **INVALIDATED.** ROI arms are `cqp` but their baselines are `vbr_1pass`, and kvazaar VBR overshoots targets 30–45% — which fully accounts for the "saving". Same defect in the SVT-AV1 rate deltas. |
| "rate-first is often replaceable for free" (mine, from 2C) | **WRONG.** 1 of 16, not 14 of 50. 2C tested quality-tie + speed and never held bitrate constant — on the axis rate-first exists to optimise. |
| "resolution is not a lever for restoration cost" (2C) | **FALSIFIED.** Real-ESRGAN: 4.85 / 1.11 / 0.49 fps at 360p / 720p / 1080p — near-linear in pixel count. Full-frame *compute* and resolution-*independent* are different claims; only the first is true. |
| "a dedicated GPU plausibly recovers ~2×" (2C) | **MISLEADING.** Runs already pin a whole RTX A6000 (`gpu_utils.preflight_gpu`). The real residual is co-tenancy jitter, unquantified. |
| "`shrink_amount` is stale" (hypothesis) | **NO — it is live and load-bearing.** Since 2026-07-20 it is the *selection budget* (fraction of blocks degraded) for **every** degradation, not a geometric shrink. Only `removal_mode: 'shrink'` is legacy. The **name** is the problem, not the parameter. |
| "under rate-first nothing is significant and the direction reverses" | **HALF RIGHT.** Reverses for `blackout+propainter` (raw p 0.022) and `blur+nafnet`; does **not** reverse for `freeze+propainter` or `ac_truncate+nafnet` — downsample wins both objectives there. So the rate rival is **blackout, not freeze**. |

---

## The claim the paper will make

> Degrade-and-restore has no single best configuration. Quality, bitrate and
> restoration cost form a frontier on which the choices genuinely conflict, and
> we characterise it: which arm is best on each axis, what each costs on the
> other two, and which arms are dominated outright.

Supporting sub-claims, in descending strength:

1. **Quality:** `downsample+realesrgan`. The only comparison surviving Holm over
   the real family (m=14): beats `blackout+propainter` 10/10 videos,
   p_Holm 0.027.
2. **Bitrate:** the `blackout` family. This is the reversal — it wins the rate
   axis where it loses the quality axis.
3. **Speed:** to be established by W3; current data cannot support it.
4. **Dominated outright:** `blackout+propainter` loses to `blur+nafnet` *and*
   `blackout+e2fgvi` on all three axes — worse quality, fewer bits saved, 2×
   slower. Stronger than the sign test alone.

---

## Workstreams

W1–W3 are prerequisites for publishing anything. W4–W6 run in parallel.

### W1 — Integrity fixes (blocking, cheap)

**W1a. Re-run the missing fixed-QP kvazaar baselines.** ~16 baseline points,
~60–90 s each, **~30 min serial, CPU-only, no GPU restoration**. Then recompute
`tab:roi`'s rate column against them. Three outcomes, all publishable: savings
survive (claim strengthens and is finally calibrated), shrink, or vanish. Add
the two 960×540 / 1280×720 points only if the 640×360 result holds.
Files: `experiments.yaml`, `sections/evaluation.tex` (`tab:roi`, lines
~1130–1168), `CLAIM(tab:roi)` marker at line ~209.

**W1b. Close the disclosure gap in the same edit.** Line ~1168 names only
`india`/`tennis` as lacking kvazaar baselines. `camel`, `dog` and `pigs` ROI
runs are equally orphaned and currently unmentioned — 10 ROI runs total.

**W1c. Fix the SVT-AV1 rate deltas the same way** (+5.0/−6.0/−18.6%; same
`crf`-arm-vs-VBR-baseline defect). The "ROI is near-inert" *quality* conclusion
survives — it is a null, independently corroborated by the qp_range 15→30 test.

**W1d. Commit Wave 2B's report.** `docs/WAVE2B_MAP_HOLES.md` is **untracked** in
the main checkout and would be lost. Also finish batch 1's outstanding map re-run
and bound check — the report ends mid-sentence on it.

### W2 — The matched-cell tradeoff surface (the core exhibit)

The per-arm medians we have are **not matched**: each arm's median is over
whatever operating points it happened to run at (147 arms for
`downsample+realesrgan` vs 8 for `blackout+e2fgvi`, different QPs and videos).
That is the same coverage confound that produced the withdrawn motion result and
the pooled-median throughput artifact. **Do not publish unmatched medians.**

Build `tools/build_tradeoff_surface.py`, reusing the arm-scoring and
JND-separability logic already in
`.claude/worktrees/operating-map-implementation-34a243/tools/build_operating_map.py`
(`score_arms`, `eligible`, `mark_cost_dominated` — do not re-derive them):

- pair arms **within** an operating point, never across
- three axes per arm: restored BG-LPIPS, Δbits vs the matched pristine baseline,
  restoration fps
- emit the 3-D Pareto frontier per cell **and** pooled over cells with an
  explicit statement of how many cells each arm appears in
- report the **conflict magnitude** — the 15-of-16 result — as a headline number

**Pin the duplicate-arm aggregation rule.** Wave 2B's counts change (9/9 → 8/9,
6/6 → 5/6) depending on whether duplicate arms at one cell are collapsed
best-of or first-of, and the rule was never stated. Choose one, justify it, and
make the tool's tests fail if it silently changes.

### W3 — Controlled timing campaign (unblocks the speed axis)

Current timings cannot support a speed claim. ProPainter at 640×360 splits into
two clusters ~10–40× apart (n=80 slow, n=81 fast) and **no recorded config
explains it** — `fp16`, `subvideo_length`, `raft_iter` are unset in both. The DB
stores **no acquisition timestamp and no GPU occupancy**, so timings cannot be
attributed to conditions at all.

1. **Record what is missing first** — add wall-clock timestamp, GPU index, and
   free/total VRAM at launch to `result.json` (`src/presley/runner.py`, near
   `preflight_gpu` at ~line 203). Cheap, and every future timing depends on it.
2. **Re-measure a fixed arm set back-to-back in one session**, ≥3 repeat trials
   per configuration, at 360p / 720p / 1080p, with co-tenancy recorded. Arms:
   the four Pareto-frontier candidates plus `blackout+propainter`.
3. **Diagnose the ProPainter split.** If it is co-tenancy, the fast cluster is
   the true figure and the slow one is contention. If it is a code change, find
   the commit. Do not publish a ProPainter speed number until this resolves.
4. Publish the Real-ESRGAN resolution scaling — it is clean, tight, and it
   **restores resolution as a deployment lever**, which the paper currently denies.

### W4 — Complete the significance picture (cheap, high value)

| arm | now | to become citable |
|---|---|---|
| `blackout+propainter` | n=10, p_Holm 0.027 | **done** |
| `blur+nafnet` | ✅ **CLOSED** — n=11, 11/11, p_Holm **0.0137** | done: `dog` + `pigs` @ svtav1 QP43 bs16, both arms added so the test measures the arm and not block size |
| `freeze+propainter` | n=6, underpowered — **still open** | n=8 is **not** enough (p_Holm ≈ 0.10); needs ~10/10 → **4 more videos**. Candidates: `color-run`, `dancing`, `drift-straight`, `drift-turn`, `motorbike`, `india` @ svtav1 QP43 |
| `ac_truncate+nafnet` | n=7, underpowered both ways — **still open** | loses quality 6/7, wins bitrate 7/7. One video reaches the n≥8 floor |

`blur+nafnet` is two runs from citable and should be done first. Note two videos
(`youtube_vos/0e4068b53f`, `b1a8a404ad`) are blocked by an **ineligible
baseline**, which no new run fixes — do not schedule them.

### W4b — NEW, no GPU: analyse the parameter sweeps we already ran but could not see

Discovered while fixing the control-matching bug. **61 of 564 arm-groups had two
or more runs collapsed into one identity**, across 16 config keys that the
analysis was blending. No re-runs are needed — the runs were executed correctly
with distinct parameters, and the defect was purely in the *analysis*. What is
newly available is a set of parameter sweeps sitting on disk, already paid for:

| key | distinct values on disk | why it matters |
|---|---|---|
| `alpha` / `beta` | 5 each | the selection-objective weights — a real ablation |
| `blur_kernel` | 7 / 15 / 31 | degradation-strength sweep |
| `ac_keep` | 1 / 2 / 4 | degradation-strength sweep |
| `mask_source` | gt / ufo / yolo | mask-sensitivity |
| `mask_morphology` | dilate / erode / jitter | mask-sensitivity |
| `downsample_uniform_level`, `downsample_levels`, `downsample_level_map` | 2–3 each | the S1 graded-downscale work |
| `inpainter_params.subvideo_length` | 20 / 45 / 80 | ProPainter speed/quality knob |

**This directly contradicts a premise of Wave 3(b)** in the old plan, which
asserted these parameters "have never been swept". Several *have* been — we
simply could not see it, because the analysis keyed on a tuple that omitted
them. Re-read that section against this table before commissioning anything.

⚠ **Already checked and negative:** `inpainter_params` does **not** explain the
ProPainter 10–40× timing split. 158 of 161 ProPainter runs at 640×360 carry an
empty params dict and still split 77 slow / 81 fast. W3's campaign is still
required.

### W4c — NEW, small: why does ROI do nothing on `tennis`?

`tennis` is the only one of 17 kvazaar points where the mechanism does not
appear — the only negative ΔFG (−0.45 dB) and the only positive ΔBG (+0.13 dB).
Both are sub-JND, so the defensible reading is **"no ROI effect"**, not "ROI
reversed", and the paper says exactly that.

**Do not repeat the mis-aimed version of this task.** An earlier note asked for a
"masked re-check" on union-bbox grounds; that was wrong, because
`foreground.psnr_mean` is already a true masked metric. There is nothing to
re-check metric-side. The open question is physical: why the removability scores
buy nothing on this clip. `tennis` has a small, scattered foreground — that is
the first hypothesis, and it is testable from the cached masks with no GPU.
Worth doing only if `tennis` is going to be discussed in the text; otherwise the
sub-JND wording already covers it.

### W5 — Content axis, round 2 (closes a question 2A left open)

2A tested four EVCA attributes and was negative. Three of the candidates named
in review were **never tested**, and two are nearly free:

- **BG-motion** — mean TC over *non*-FG blocks. **One line** in
  `tools/analyze_content_axis.py`; 2A tested global motion (A1) and FG-motion
  (A2) but never the FG/BG *decomposition*.
- **Duration / frame count** — `runs.video_frames`, 38 to 2440, a **64× spread**.
  This is plausibly the coverage/provenance confound 2A blamed, measured
  directly rather than inferred.
- **FG-percentage** — 2A did *not* re-test it, relying on a prior refutation
  that was against a **different target** (win/loss, not T2/T3). Masks are in
  `dataset/annotations/` and 33 `cache/*/ufo_masks` dirs.

Pre-register again, keep the unit = video (n=19 for T2, n=23 for T3), Holm over
the new k. **Expect negative**: T1 remains untestable regardless, so even a hit
here cannot tell a deployer which arm to pick — it would only explain *where the
map has choices to offer*. Say that in the pre-registration.

### W6 — Naming and hygiene

- **Rename `shrink_amount` → `degrade_fraction`** (or document it loudly in
  place). It is a selection budget applied to every degradation; the name says
  geometric shrink, which misled two people this session. **It is part of
  `compute_experiment_hash` and of the control-matching key**, so a rename
  invalidates every existing hash — prefer an alias read plus a documented
  deprecation, not a bare rename. Decide explicitly, do not do it casually.
- Record the two session traps already logged (YAML anchors defeat fragment
  round-trips; process-count checks match their own command string).

### W7 — Paper restructure (after W1–W3)

Retire the operating-map framing in the text; adopt the tradeoff surface.
`docs/PLAN_PRESENTATION.md` F1 becomes the **3-axis frontier scatter**, not the
ladder-residual plot it currently specifies. Keep `\rev{}` tracking; the JND
provenance paragraph already landed (paper repo commit `61b32e3`, unpushed).

---

## Verification

- `PYTHONPATH=$PWD/src /home/itec/emanuele/.conda/envs/presley/bin/python -m pytest -q`
  (`python` is not on PATH; env at `/home/itec/emanuele/.conda/envs/presley`).
- W1a: the recomputed `tab:roi` rate column, shown as a diff against the
  published numbers, with both baselines' `rate_control` printed beside it.
- W2: `build_tradeoff_surface.py` must reproduce the 15-of-16 conflict count and
  the `blackout+propainter` triple-domination, with tests pinning the
  aggregation rule.
- W3: repeat-trial CV per configuration reported; any cell with CV > 0.3 is
  published as "not measurable", not as a number.
- W4: `presley-compare --suite` output with `--candidates-tried` set to the true
  family size **including losers**.
- Every batch: **verify result count against entry count** — `presley-run` exits
  0 when every entry fails (`grep -c 'Error running experiment' <log>`).

## Not doing

- **Wave 3 (quantization/distillation).** Its trigger was a cost/quality
  decision the map would settle; the map is gone and W3 has not yet established
  whether speed is even reliably measured.
- **New transports or restorers.** Nothing in the audit points at a missing arm.
- **More coverage for its own sake.** The corpus is already confounded with
  coverage; W4's targeted runs are the exception because they close named
  significance gaps.
