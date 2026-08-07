# Handoff — rewrite the PRESLEY manuscript into a publishable TOMM paper

**Read this whole file before editing a single `.tex` line.** The science is
done and defensible; what is wrong is the *paper*. It is 45 pages against a
23-page budget, two thirds of it is one section, and it reads as a lab notebook
rather than an article.

Your job is **structural rewriting, not new experiments.** Every number you need
already exists and is committed. If you find yourself wanting to run something,
re-read §6 first — the odds are it has been run, and possibly refuted.

---

## 1. Start here: the two commands that tell you where you are

```bash
tools/render_paper.sh /tmp/presley_render && python tools/paper_metrics.py --pages 45
```

`render_paper.sh` builds the PDF locally. **This worked for the first time on
2026-08-06**; every earlier handoff said "pdflatex is not installed, Overleaf is
the first real compile". Four things had to line up, all documented in the
script's header — the one that will bite you if you touch the toolchain is that
**biber must be 2.17, not 2.21**: tectonic bundles biblatex 3.17 which writes a
v3.8 control file, biber 2.21 demands v3.11, and it fails with *empty* stdout
and stderr, surfacing only as `external tool exited with error code 2`.

`paper_metrics.py` reports words, floats and float spacing per section against
journal heuristics, splitting HARD failures from soft warnings. Re-run it after
every structural edit — it is the only feedback loop that will tell you whether
the rewrite is actually working.

**Always render before claiming a page count.** Estimating from words is a
fallback, and the estimate is calibrated to the *current* float density.

---

## 2. What the numbers say today

| section | words | % body | tables | figures | algs |
|---|---|---|---|---|---|
| Introduction | 1133 | 5.5% | 0 | 1 | 0 |
| Background and Related Work | 830 | 4.0% | 0 | 0 | 0 |
| PRESLEY (method) | 3590 | 17.5% | 1 | 3 | 10 |
| **Performance Evaluation** | **13541** | **66.0%** | **31** | 4 | 0 |
| Conclusions | 1209 | 5.9% | 0 | 0 | 0 |
| **total** | **20505** | | **32** | **8** | **10** |

Rendered: **45 pages**. Budget: **23**. Abstract: **504 words**.

**Hard failures:**
- Over budget by 22 pages — roughly **9,800 words / 49% must go**.
- Evaluation is **66% of the body**. No section should exceed 40%.
- **Six stretches longer than two pages with no float at all**, the worst
  **6.7 pages** (in `main.tex`, the abstract→introduction→conclusions run).

**Soft warnings worth acting on:**
- **Figures are 16% of floats** (8 figures vs 32 tables). Tables carry values;
  figures carry *shape*. A reader skimming 31 tables learns nothing about shape.
- Introduction (6%) and Background (4%) are both under the 8–15% a journal
  article normally gives them — they were cut hard for a referee and are now
  undersized *relative to* the evaluation, not in absolute terms.
- Method is 18% (want 20–30%).
- Conclusion is marginally longer than the introduction, which usually means it
  re-argues the paper rather than closing it.
- **The abstract is 504 words.** TOMM expects ~150–250. It currently contains a
  results section.

---

## 3. The shape to aim for

At ~467 words/page, 23 pages is **~10,700 words**. A defensible split:

| section | target % | target words | now |
|---|---|---|---|
| Introduction | 12% | ~1,300 | 1,133 ✔ |
| Background | 12% | ~1,300 | 830 (grow slightly) |
| Method (PRESLEY) | 26% | ~2,800 | 3,590 (trim ~20%) |
| Evaluation | 40% | ~4,300 | 13,541 (**cut 68%**) |
| Conclusions | 8% | ~850 | 1,209 (trim) |

**Almost the entire cut comes out of the evaluation**, and it must come out as
*whole results*, not as compressed prose. Compressing 31 tables into denser
paragraphs produces an unreadable paper that is still too long.

**Float target:** ≥1 float per 2 pages (≈12 minimum), with **≥25% figures**.
Given a 23-page paper, aim for roughly **8 figures and 10–12 tables**. That
means going from 32 tables to ~11 — which is another way of saying the same
thing: most tables must merge, become figures, or move to an appendix.

---

## 4. How to cut the evaluation without losing the science

The evaluation has 31 tables because each was landed as its own experiment. A
reader needs a *narrative*, and the paper already has one — the three goals.
Reorganize around them and most of the merging becomes obvious.

**Keep as first-class, with a float each (these are the paper):**

1. **`tab:ratematched-n13`** — PRESLEY beats ELVIS at matched rate on **13/13**
   ladders, mean **−56.4%** BD-rate BG-LPIPS, exact two-tailed sign
   **p=0.000244**, surviving a Holm family of **k≤204**. Two codecs, three
   dataset families. *This is the headline. It should be visible by page 2.*
2. **`sec:restorability` (M1)** — post-restoration damage is predictable at
   transmit time (spatial complexity, ρ **+0.506**, **120/120** runs), and it
   predicts in the **wrong direction**: the score already selects on complexity
   because complex blocks cost the most bits, so the objective **preferentially
   degrades the blocks that survive restoration worst**. This is the mechanism
   behind the α/β null and the answer to referee 2's "lack of technical insight".
3. **`tab:frontier` + `tab:speed-scaling`** — quality, bitrate and cost order the
   arms three different ways. `downsample+realesrgan` wins quality against every
   rival tested (p_Holm 0.014–0.025); on bitrate the ordering inverts and is not
   significant; on speed it sits mid-pack.
4. **`tab:av1` + `tab:av1-breadth` + `fig:regime-reversal`** — the honest
   negative: the starved-bitrate saving does **not** generalize, and the flip is
   *inverted* on the added content. Two independent mechanisms agree, so it is a
   content result. **Do not bury this.** It is the most credibility-building
   thing in the paper and referees have already seen it in the response letter.
5. **`sec:regime-stability` (R1)** — the advantage does *not* depend on operating
   point (p=0.58). A properly powered negative that **widens** the claim.

**Merge aggressively (candidate groupings):**

- The four breadth tables (`tab:breadth`, `tab:breadth-ext`,
  `tab:breadth-ext-presley`, `tab:breadth-ratematched`) → **one** dataset-breadth
  table plus one sentence each on what the intermediate stages showed.
- The restorer catalogue (`tab:inpainters`, `tab:conditioned`,
  `tab:conditioned-twins`, `tab:conditioned-stream-diffvsr`,
  `tab:instantir-kill`, `tab:goal2`, `tab:goal2-breadth`) → **one** restorer
  comparison table. The finding is singular: *no backbone separates from
  Real-ESRGAN by a perceptible margin, and conditioned restoration beats
  in-painting because in-painters discard the transmitted prior by construction.*
- The selection ablations (`tab:ablation`, `tab:graded`, `tab:graded-oracle`,
  `tab:budget-knee`) → **one** selection table + the M1 result.
- `tab:transport`, `tab:fillvariant`, `tab:ratecontrol`, `tab:priced-trade` →
  compress into a short "what we retired and why" subsection with **one** table.

**Move to an appendix or the response letter** (TOMM allows appendices beyond the
main-body limit — *verify this against the current author guide before relying on
it*): per-video hash provenance, the VBR/fixed-QP methodology ablation, the
FG-metric audit detail, `tab:parameters`.

---

## 5. Figures to draw (this is where the paper gains most)

You need ~8 figures and have 8, of which only 4 are in the evaluation. Suggested,
in value order — **all buildable from committed data, no new runs**:

1. **Pipeline overview** — exists (`Figures/Overview.pdf`), keep.
2. **`fig:regime-reversal`** — exists, built by
   `tools/plot_regime_reversal.py`. Rate–distortion ladders showing the
   reversal a BD number hides. Already validated: the script re-derives all four
   BD-rates and aborts if they disagree with the published table.
3. **The three-axis frontier** — quality vs bitrate vs restoration cost, arms as
   points. Replaces prose describing three orderings.
   Data: `tools/build_tradeoff_surface.py`, `tab:frontier`, `tab:speed-scaling`.
4. **The M1 scatter** — per-superblock complexity vs post-restoration damage,
   with the regression and the *wrong-direction* sign made visual. This is the
   paper's best idea and it currently has no picture.
   Data: `results/block_damage_s1b.npz` via `tools/analyze_m1_restorability.py`.
5. **Matched-rate BD-rate across 13 ladders** — a sorted bar chart, all bars
   negative. Makes "13/13" instantly legible.
6. **Cost scaling with resolution** — Real-ESRGAN's sub-linear curve
   (3.25× cost for 4× pixels; 7.5× across 360p→1080p).
7. **Qualitative frames** — degraded vs restored vs pristine crops. A generative
   restoration paper with no visual examples is a real gap, and referee 2 asked
   about plausibility. Frames exist under `results/<hash>/restored_frames/`.
8. **The selection diagnosis** — score terms vs the bit oracle (0.833 captured
   vs a 0.402 random null), showing the numerator is nearly saturated.

**Placement rule:** no stretch of prose longer than two pages without a float.
`paper_metrics.py` reports violations with their location.

---

## 6. Rules you must not break

These are not style preferences. Each one exists because breaking it already
produced a wrong conclusion in this project.

- **Fixed-QP/CRF only for degradation comparisons. Never VBR.** VBR launders bit
  relocation — 25/25 matched pairs project-wide, zero counterexamples.
- **Foreground claims only from true masked metrics** (`foreground.lpips_mean`,
  `dists_fg`). **FG-VMAF and FG-FVMD are banned** — they are union-bbox
  artefacts; the "foreground" box averaged 76% of frame area against a true
  foreground of 15%.
- **A sub-JND delta is "no perceptible difference", never a trend** — regardless
  of how consistent its sign or how small its p-value. JND: 0.5 dB PSNR, 0.05
  LPIPS/DISTS.
- **n≥6 for any significance claim; two-tailed always.** At n=6 the sign test
  floors at p=0.031, which admits a Holm family of exactly one. Holm families
  count *every candidate tried, losers included*.
- **A run with non-empty `invariant_failures` is never citable.**
- **Never `rm` anything under `results/`, `dataset/`, `cache/`.**
- **Every number traces to a `results/<hash>` through a `CLAIM` line.** There are
  **44 CLAIM anchors**. Moving or merging a table means moving its CLAIM with it.
  **Do not delete a CLAIM to tidy up** — it is the provenance chain, and it is
  the reason this paper can defend itself.
- **`NOTE(...)` markers are wording constraints, not commentary.** Several say
  explicitly "do not quote X" or "do not word this as a win". Read the NOTEs
  attached to any table you touch. The ones that matter most:
  - `tab:av1-breadth` — the flip is *inverted*, not absent; n=4 is descriptive.
  - `sec:restorability` — quote **ρ 0.506**, never 0.684 (the latter is selection
    leakage: complexity predicts *whether a block was degraded* at ρ 0.786).
  - `sec:regime-stability` — must never be conflated with `tab:av1`'s regime
    flip. Different goals: restoration is regime-stable; **bit relocation is
    regime- and content-dependent.**
  - `tab:chain` — banned phrasings: "the chain holds" as an ordering, "each
    stage adds", "cumulative", "compounding", "end-to-end gain of X%". Composition
    is untested *and* mechanistically doubtful.
  - `tab:roi` — the bitrate-saving claim is **retracted** (a VBR-baseline
    overshoot artefact). The relocation is real but its *visible* half is the
    loss: foreground gain sub-JND on every metric, background cost −0.54 dB at
    8/8, p=0.008.
- **Revision tracking:** reviewer-visible text you add or change goes in
  `\rev{}`; removed text goes in `\del{}` and stays visible. **A rewrite this
  large will make `\rev{}` meaningless if applied naively** — see §8.

---

## 7. What the user asked for that is NOT yet fully measured

Point 1 of the brief asks for comparisons on **speed, size and quality**. Two of
those have gaps you should either fill or scope honestly:

- **Speed by pipeline step.** Only *restoration* is timed cleanly
  (`tab:speed-scaling`, 2026-08-06 campaign, pinned GPU). `selection_time_seconds`
  was added to the runner on 2026-08-06 but **no run has it yet** — it only
  populates on new runs. Encoding and selection are still folded into one
  `encoding_time_seconds`. Either run a handful of fresh `presley_ai` runs to
  populate it, or state the breakdown as restoration-only and say why.
  **No pre-2026-08-05 timing may be quoted** — those mix two device populations
  (silent CPU fallback) and which one a run belongs to is unrecoverable.
- **Bitrate at different resolutions.** `tab:speed-scaling` covers 360p/720p/1080p
  for *cost*, but there is no systematic bitrate-vs-resolution table. Most of the
  corpus is 640×360. This is a genuine gap: either run a small resolution ladder
  or scope the claim to 360p explicitly.
- **Quality on traditional metrics.** PSNR/SSIM exist per run; the paper leads on
  LPIPS/DISTS by design. A journal reader will want both — report traditional
  metrics alongside, but keep the *verdict* on the masked perceptual metrics, per
  the rules above.

---

## 8. Practical advice for a rewrite this large

- **Work in a branch and a worktree.** This is a substantive change spanning
  every file: `git worktree add ../wt-presley/paper-rewrite -b refactor/paper-rewrite`.
  The paper repo (`68e8b6bb11d0dd9e62a67aef/`) is a **separate git repo** whose
  origin is Overleaf — commit and push it separately.
- **Decide the `\rev{}` policy first, and ask the user.** The manuscript is a
  revision with tracked changes. A 49% rewrite cannot sensibly mark every change
  as `\rev{}` — the whole paper would be blue. Options: (a) declare the revision
  complete and reset tracking, (b) keep `\rev{}` only for changes that alter
  *claims* rather than *structure*. **This is the user's call, not yours.**
- **Cut whole results, then re-render, then re-measure.** Do not micro-edit
  prose for length first; it is the slowest path and it damages the writing.
- **Keep the response letter in sync.** `response_letter.tex` (9 pages, compiles
  via tectonic) cites tables by label. If you rename or merge a table, the letter
  must follow, and **a letter that contradicts the manuscript is the single worst
  failure mode available** — it converts a defensible negative into an apparent
  concealment.
- **`git log` order is checkable.** Two pre-registrations
  (`docs/PREREG_R1_REGIME_SCOPE.md`, `docs/PREREG_M1_RESTORABILITY.md`) were
  committed *before* the runs they govern, deliberately. Do not rewrite history
  in a way that destroys that ordering.

---

## 9. State of the repos, 2026-08-06

- **Code** `main` @ `129827a`, clean, CI green, tests pass.
- **Paper** `main` @ `1bfd869`, clean, pushed to Overleaf.
- **One open `HOLE`**: `tab:instantir-kill` (DiffBIR) — a deliberate deferral,
  no referee asked for it. **ASK before wiring DiffBIR.**
- **Submission copy**: `tools/make_submission_copy.py` writes a referee-safe tree
  with all ~117 non-CLAIM markers stripped (1209 comment lines), leaving the
  source untouched. Needed because `.tex` source goes to TOMM alongside the PDF,
  and the markers contain retraction histories and notes addressed to agents.
- **Recently fixed and worth knowing**: `references.bib` had an entry with **no
  citation key** (`@inproceedings{,` with `fvmd,` on the next line). biber
  rejected the whole file. Check the rendered bibliography contains FVMD.

## 10. The one-paragraph version

PRESLEY's evidence is strong and unusually honest: a matched-rate win over its
predecessor on 13/13 ladders at p=0.000244 that survives any multiple-comparisons
objection; a mechanism explaining why its own selection score is not merely
uninformative but *pointed the wrong way*; a properly powered negative showing
the advantage is regime-independent; and a frankly reported non-replication that
most papers would have buried. None of that is legible in 45 pages of which two
thirds is one section and 31 of 40 floats are tables. **The rewrite is a
structural problem, not a scientific one — cut whole results rather than
compressing prose, draw the six figures the argument is missing, and let the
three-goal narrative carry the paper.**
