# Plan B — presentation restructure for submission

**Status:** proposed 2026-08-02, not started. Workstream 2 of 2.
**Depends on `docs/PLAN_OPERATING_MAP.md` Wave 1.** Do not begin the exhibit
rebuild before the map exists: which results survive scoping determines what
there is to present. The cleanup steps (§4) are independent and can start now.

---

## 1. The diagnosis, measured

| | count |
|---|---|
| Live tables | **26** |
| Live figures | **3** |
| Live figures **in the evaluation section** | **0** |

All three figures are method diagrams in `presley.tex`. **Every result in the
article is a table.** That is the problem in one line, and it is not "too many
tables" — it is that the article has no visual argument at all. A reader cannot
see a trend, a distribution, or a tradeoff anywhere; they can only read numbers
and be told what the numbers mean.

Four figure environments already exist in `evaluation.tex` but are commented
out, so the section was drafted with figures in mind and they were dropped
rather than replaced.

## 2. The principle to restructure by

**One exhibit per claim, and the form follows the claim type.** A table is the
right form for a small number of exact values that a reader will compare
pairwise or quote. It is the wrong form for a trend, a distribution, a
tradeoff, or a per-video split — all of which the article currently renders as
tables of numbers.

| claim shape | right form | currently |
|---|---|---|
| A tradeoff between two quantities | scatter with the frontier drawn | tables |
| An outcome that varies by content | per-video plot with a significance band | tables |
| A curve over an operating range | curve | tables of BD numbers |
| A few exact values to be quoted | **table** | table ✓ |
| A retired candidate | one sentence, detail to appendix | full table |
| A mechanism | diagram | prose |

## 3. Proposed exhibit set

### New figures (the article's visual argument)

- **F1 — the information ladder, used the way a BD-rate curve is used.**
  Transmitted bits against background damage, one point per transport per
  operating point, with the fitted ladder drawn and the **off-ladder residuals
  highlighted**.

  The ladder is not merely descriptive: it plays the role a rate-distortion
  curve plays for BD-rate. BD-rate scores a codec against a rate-quality curve;
  the **ladder residual** (Plan A, Wave 1 step 3) scores a transport against the
  rate-damage curve its peers define, giving one signed, cross-cell-comparable
  number where today we have only per-cell verdicts. F1 is that metric's
  picture, and the two should land together — the figure without the scalar is
  an illustration, the scalar without the figure is unmotivated.

  ⚠ **Plot the absolute restored quality too, not only the residual or the
  gain.** Blackout is the standing warning: largest restoration gain
  (1.70× JND) and worst absolute result (0.374 vs downsample's 0.220). A
  gain-only exhibit reproduces exactly the misreading that an earlier draft of
  Plan A made. Second panel, or a second encoding on the same axes.

  Replaces the numeric content of `tab:transport` and the ladder `NOTE` on
  `tab:priced-trade`.
- **F2 — the operating map.** (content, rate) → recommended transport+restorer,
  shaded by the winner's margin in JND, with "no separable winner" as an
  explicit category rather than a blank. This is Plan A's deliverable and the
  article's answer to "what should I deploy?".
- **F3 — replication. ⚠ CONDITIONAL — do not build before Plan A Wave 1.**
  Per-video effect with a JND band, for the three breadth results: at a glance,
  `bear`/`camel` win and `dog`/`pigs` do not, currently spread across
  `tab:av1-breadth`, `tab:conditioned-breadth` and `tab:goal2-breadth` as three
  tables the reader must cross-reference.

  **The risk is that Plan A makes this figure obsolete before it is drawn.**
  F3 asks "which videos win?", which presupposes a single fixed configuration
  per video. Plan A's whole premise is that the right configuration *varies*
  by video and rate — so under the map, a video that "loses" may simply have
  been run with the wrong transport. If the map succeeds, "dog loses" becomes
  "dog at QP 63 wants blackout, not downsample", and F2 subsumes F3.

  Three outcomes, decide after Wave 1:
  1. **Map succeeds and explains the win/loss split** → drop F3, F2 carries it.
  2. **Map succeeds but the split survives within it** (best-configuration
     still loses on some videos) → build F3, and it becomes *stronger*: the
     failure is not a bad configuration choice but a real content limit.
  3. **No separable winners (map fails)** → build F3 as specified; the breadth
     results are then the paper's main scoping evidence.

  Building F3 now would risk drawing the third-best version of it.
- **F4 — cost/quality Pareto.** fps against restoration gain, dominated arms
  marked. Makes the efficiency axis a decision variable instead of a footnote,
  which is what elevating it to a cross-cutting theme requires.

### Consolidations

| current | proposal |
|---|---|
| `tab:breadth`, `-ext`, `-ext-presley`, `-ratematched` (4) | **1 summary table + F3**; per-clip detail to appendix |
| `tab:av1-breadth`, `tab:conditioned-breadth`, `tab:goal2-breadth` (3) | **F3**, keeping one table for exact quotable values |
| `tab:priced-trade`, `tab:budget-knee`, `tab:transport` (3) | **F1** + one table for the knee's exact numbers |
| `tab:mask-sens`, `tab:mask-morph`, `tab:fillvariant` (3) | **1 compact robustness table**, likely appendix |
| `tab:roi`, `tab:hnerv` (2) | **1 external-baselines table** |
| `tab:instantir-kill` | one sentence + appendix; it is a retirement, not a result |
| `tab:throughput` | fold into **F4** |

Indicative target: **~10 tables and 4–6 result figures**, from 26 and 0. The
exact count matters less than that every surviving exhibit answers "which claim
is this, and is a table the right way to see it?"

### Keep as tables

`tab:parameters` (setup), `tab:ablation` and `tab:graded` (the selection
negative result — exact values are the point), and one table per goal holding
the quotable headline numbers.

## 4. Cleanup — independent, can start immediately

Three separable jobs. **They are not equally safe.**

1. **Goal labels — ✅ DONE 2026-08-02, in the session that wrote this plan.**
   Done first precisely because it touched every file the other workstreams
   will edit; leaving it to a parallel session would have guaranteed conflicts.

   The defect was real: the introduction promised three goals in one order
   while `evaluation.tex` numbered two in a different order, so "Goal 1" in the
   body meant the introduction's *second* goal.

   **The fix was not a renumber.** This plan originally proposed renumbering to
   1/2/3 and scoped it at "~26 marker blocks plus prose". That was wrong by
   about 5×: the real cost is **~124 sites** across the paper, the research log,
   `reviewers_comments.md` and `docs/`, and `hard-rules.md` *defines* the legacy
   labels — its rule 1 is phrased "inverting Goal 1". The failure mode is
   silent, since one stale "Goal 1" asserts the opposite of what it says.

   Instead, reviewer-visible prose now **names** the goals ("bit relocation",
   "generative restoration", "selection") and numbers none of them, so nothing
   is numbered twice and the collision cannot arise. 21 sites changed, all
   inside existing `\rev{}` blocks. The ~19 numbered references in comments are
   kept deliberately — markers key to `hard-rules.md`, whose vocabulary stays
   stable. **Do not harmonize the two vocabularies in either direction**, and
   do not reintroduce numbers into the prose.
2. **Marker sweep.** The paper's own convention already defines this: before
   final submission the discovery grep must return **only `CLAIM` lines**.
   Resolve or consciously delete every `GOAL`/`NOTE`/`HOLE`/`NEXT`. Two `HOLE`s
   remain, both unrelated to the closed wave: `HOLE(tab:instantir-kill)`
   (DiffBIR — **ASK before wiring**) and `HOLE(sec:downsample-vs-uniform)`.
3. **Research log.** `open-questions.md` (310) and `dead-ends.md` (306) are
   over the log's own 300-line ceiling. Drain the entries whose CLAIMs have
   landed.

### ⚠ Do NOT strip `\rev{}` / `\del{}`

The paper is **in revision for TOMM** with all 17 referee items closed.
`\rev{}` is how the referees see what changed and `\del{}` is text deliberately
kept visible *for them*. The paper's own `CLAUDE.md` says "do not strip these".
Removing them now would make the revision unreviewable. **That sweep happens
after acceptance, not before submission.**

## 5. Build check

`pdflatex` was unavailable on this host until 2026-08-02; a full TeX Live is
being installed to `~/opt/texlive` (no root needed — conda-forge's
`texlive-core` ships binaries with an empty package tree and a broken `tlmgr`).
Until it lands, **Overleaf is the only real build** and brace/column checks are
by hand.

Note for whoever gets the local build working: TeX Live 2026's `acmart`
conflicts with the article's `\addbibresource` setup (natbib vs biblatex).
**This is not believed to be a defect in the paper** — Overleaf pins an older
`acmart` and the article has been through a submission and a revision there.
Do not "fix" the paper to satisfy a newer local toolchain; verify against
Overleaf first.

## 6. Waves

The first draft of this plan gave a flat 6-step list. That was an oversight —
the host guideline asks for parallel-agent waves on multi-part work, or an
explicit statement of why they were skipped, and neither was given. This plan
*does* have independent pieces, so here they are properly grouped. A wave starts
only when every workstream it depends on has reported; workstreams inside a wave
launch together, each in its own worktree.

### Wave P1 — cleanup (no dependency on Plan A; start now)

| # | workstream | files |
|---|---|---|
| P1a | ~~Goal labels~~ **DONE 2026-08-02** | — |
| P1b | Marker sweep to the camera-ready standard | `main.tex`, `sections/*.tex` |
| P1c | Research-log drain (`open-questions.md` 310, `dead-ends.md` 306, both over the 300 ceiling) | `research-log/*.md` |
| P1d | Close the throughput alarm (svtav1 720p 3.2 fps vs 1080p 28.9 fps is impossible) — Plan A Wave 2C cannot report cost until this is explained | `results/`, no GPU |

P1b and P1c touch disjoint files and can run together. **P1d is here rather
than in Plan A because it is a measurement bug, not a design question**, and it
blocks an exhibit (F4).

### Wave P2 — exhibits that need only existing data

Starts after **Plan A Wave 1** reports (the map exists, or is shown not to).

| # | workstream | depends on |
|---|---|---|
| P2a | **F1** ladder + residual figure | Plan A W1 step 3 (the residual metric) |
| P2b | **F4** cost/quality Pareto | P1d (alarm closed) |
| P2c | Decide F3's fate per the three outcomes in §3; build only under outcome 2 or 3 | Plan A W1 |

### Wave P3 — the map's own exhibit

| # | workstream | depends on |
|---|---|---|
| P3a | **F2** operating map | Plan A W1 *and* W2A (content axis, which sets whether rows are named classes or bare video titles) |

F2 is deliberately last among the figures: its row labels are the content
classes, and whether those exist at all is Plan A Wave 2A's open question. Under
a negative result there F2 still ships, with videos as rows and an explicit
statement that no predictive class was found.

### Wave P4 — consolidation and narrative

Starts when every figure that will exist does.

| # | workstream |
|---|---|
| P4a | Table consolidations per §3 (breadth ×4→1, ladder ×3→1, robustness ×3→1, baselines ×2→1) |
| P4b | Appendix construction for the demoted detail |
| P4c | Full narrative read-through — the argument will have changed shape, and section ordering should follow the new argument rather than the old one |

P4a and P4b are parallel; **P4c is strictly last** and is one agent's job, since
narrative coherence is exactly what parallel editing destroys.
