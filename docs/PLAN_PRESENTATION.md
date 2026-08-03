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

**F1 — rate–quality curves, where the shape carries the argument.**
These are genuine BD-rate curves and we already have the data for them: every
four-rung QP ladder behind `tab:bdrate`, `tab:av1`, `tab:av1-breadth`,
`tab:priced-trade` and `tab:conditioned-breadth` *is* an R–D curve, currently
collapsed to a single BD number in a table.

Collapsing is the wrong call for the central result. **The regime reversal is a
curve-shape phenomenon** — the sign of the bit saving flips along the ladder,
and on `dog`/`pigs` it flips the opposite way from `bear`/`camel`. A single BD
integral hides exactly that, and hides curve crossings generally. Plot the
curves where shape is the argument; **keep the BD numbers in a table** for exact
quotable values. Figure and table are not redundant here — one shows shape, the
other gives numbers, which is the conventional division and what a reviewer
expects.

Do *not* plot an R–D curve for every table. Two panels earn their space: the
regime reversal, and the conditioned pipeline against the pristine baseline.

**F2 — the information ladder. Not a BD-rate curve; a different object.**
Transmitted bits against background damage, one point per transport per
operating point, with the fitted ladder drawn and off-ladder residuals
highlighted.

An R–D curve traces *one* method across its rate points. F2 is
cross-sectional: it compares *different transports* at a *fixed* operating
point, so it shows the reduction/restoration tradeoff itself rather than one
method's efficiency. The **ladder residual** (Plan A, Wave 1 step 3) is the
scalar that goes with it, and is *analogous to* BD-rate in the way it scores an
arm against a curve its peers define — but it is not a BD-rate and should not
be labelled as one.

⚠ **Plot absolute restored quality, not only residual or gain.** Blackout is
the standing warning: the largest restoration gain (1.70× JND) and the worst
absolute result (0.374 against downsample's 0.220). A gain-only exhibit invites
the reader to conclude the opposite of what the data supports. Second panel, or
a second encoding on the same axes.

Replaces the numeric content of `tab:transport` and the ladder `NOTE` on
`tab:priced-trade`.

**F3 — the operating map.** (content, rate) → recommended transport + restorer,
shaded by the winner's margin, with "no separable winner" an explicit category
rather than a blank. Plan A's deliverable and the article's answer to "what
should I deploy?".

**F4 — cost/quality Pareto.** Throughput against restoration gain, dominated
arms marked. Makes efficiency a decision variable rather than a footnote. If
Plan A's Wave 3 runs, model precision joins this figure as a third axis.

**F5 — replication. ⚠ CONDITIONAL — decide after Plan A Wave 1.**
Per-video effect with a significance band, for the three breadth results:
`bear`/`camel` win and `dog`/`pigs` do not, currently spread across
`tab:av1-breadth`, `tab:conditioned-breadth` and `tab:goal2-breadth`.

F5 asks "which videos win?", which presupposes one fixed configuration per
video — exactly what Plan A's map denies. Under the map, "dog loses" may become
"dog at QP 63 wants blackout, not downsample", and F3 subsumes F5.

1. **Map succeeds and explains the split** → drop F5; F3 carries it.
2. **Map succeeds but the split survives within it** → build F5, and it is
   *stronger*: the failure is a real content limit, not a bad configuration.
3. **Map fails (no separable winners)** → build F5; the breadth results are
   then the paper's main scoping evidence.

### Consolidations

| current | proposal |
|---|---|
| `tab:breadth`, `-ext`, `-ext-presley`, `-ratematched` (4) | **1 summary table + F5** (if F5 survives); per-clip detail to appendix |
| `tab:av1-breadth`, `tab:conditioned-breadth`, `tab:goal2-breadth` (3) | **F3 or F5**, keeping one table for exact quotable values |
| `tab:priced-trade`, `tab:budget-knee`, `tab:transport` (3) | **F1 + F2** + one table for the knee's exact numbers |
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

## 3b. Significance, and what the exhibits should show

The article currently expresses almost every verdict as a **JND multiple**.
That answers *is the difference perceptible?* It does not answer *is the
difference real, or sampling noise?* — and hard rule 2b's machinery (two-tailed
sign test, Holm correction over candidates, TOST for equivalence, n≥6 videos)
exists precisely to answer the second.

**The thresholds are the exposed part.** `src/presley/compare.py` calls them
"literature just-noticeable-differences" but cites nothing, and gives LPIPS and
DISTS the same 0.05 despite different scales. PSNR 0.5 dB and VMAF 6 are
defensible; the perceptual-metric constants are adopted convention. A reviewer
who pulls that thread pulls on most of the article's verdicts at once.

This is a presentation problem as much as a methodology one, so the exhibits
should be built to defuse it:

- **Error bars / significance bands, not bare JND multiples.** Where n permits
  a test, show the interval. F5's per-video band should be a confidence
  interval with the JND drawn as a *reference line*, so the reader sees both
  "is it real" and "is it perceptible" at once, and can see when the two
  disagree.
- **Draw the JND as a line, never as the axis.** A figure whose y-axis is "JND
  multiples" bakes the constant into the geometry; one with a metric axis and a
  JND rule line survives a different constant.
- **State the threshold and its status in every caption** that uses it —
  adopted convention, sensitivity-tested, not a measured perceptual study.
- **Report the sensitivity analysis** (Plan A: recompute at 0.03 / 0.05 / 0.08)
  wherever a recommendation depends on the threshold. A recommendation stable
  across all three is far stronger than one quoted at a single constant.

**No MOS study exists**, and the article already says so. That is the honest
ceiling here: with no human ratings, JND is a stand-in and should be presented
as one rather than as a measured perceptual fact.

## 4. Cleanup — independent, can start immediately

Three separable jobs. **They are not equally safe.**

1. **Goal labels — ✅ DONE 2026-08-02, in the session that wrote this plan.**
   Done first precisely because it touched every file the other workstreams
   will edit; leaving it to a parallel session would have guaranteed conflicts.

   The defect was real: the introduction promised three goals in one order
   while `evaluation.tex` numbered two in a different order, so "Goal 1" in the
   body meant the introduction's *second* goal.

   **Renumbering to 1/2/3 was considered and rejected**, with the cost measured:
   **~124 sites** across the paper, the research log, `reviewers_comments.md`
   and `docs/`, and `hard-rules.md` *defines* the legacy labels — its rule 1 is
   phrased "inverting Goal 1". The failure mode is silent, since one stale
   "Goal 1" asserts the opposite of what it says.

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

Independent pieces run in parallel. A wave starts only when every workstream it
depends on has reported; workstreams inside a wave launch together, each in its
own worktree.

### Wave P1 — cleanup (no dependency on Plan A; start now)

| # | workstream | files |
|---|---|---|
| P1a | ~~Goal labels~~ **DONE 2026-08-02** | — |
| P1b | Marker sweep to the camera-ready standard | `main.tex`, `sections/*.tex` |
| P1c | Research-log drain (`open-questions.md` 310, `dead-ends.md` 306, both over the 300 ceiling) | `research-log/*.md` |
| P1d | Close the throughput alarm (svtav1 baseline 720p 3.2 fps vs 1080p 28.9 fps is impossible) — Plan A Wave 2C cannot report cost, and F4 cannot be drawn, until this is explained | `results/`, no GPU |

P1b and P1c touch disjoint files and can run together. **P1d is here rather
than in Plan A because it is a measurement bug, not a design question**, and it
blocks F4.

A fifth, optional workstream: **P1e — threshold provenance.** Find citations
for the JND constants in `src/presley/compare.py`, or mark them as adopted
convention (§3b). No dependency, and it feeds every caption.

### Wave P2 — exhibits that need only existing data

Starts after **Plan A Wave 1** reports (the map exists, or is shown not to).

| # | workstream | depends on |
|---|---|---|
| P2a | **F1** rate–quality curves (regime reversal) | none beyond existing data |
| P2b | **F2** ladder + residual figure | Plan A W1 step 3 (the residual metric) |
| P2c | **F4** cost/quality Pareto | P1d (throughput alarm closed) |
| P2d | Decide **F5**'s fate per the three outcomes in §3; build only under outcome 2 or 3 | Plan A W1 |

### Wave P3 — the map's own exhibit

| # | workstream | depends on |
|---|---|---|
| P3a | **F3** operating map | Plan A W1 *and* W2A (content axis, which sets whether rows are named classes or bare video titles) |

F3 is deliberately last among the figures: its row labels are the content
classes, and whether those exist at all is Plan A Wave 2A's open question. Under
a negative result there F3 still ships, with videos as rows and an explicit
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
