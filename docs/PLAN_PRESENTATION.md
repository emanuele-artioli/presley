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

- **F1 — the information ladder.** Transmitted bits vs untreated background
  damage, one point per transport per operating point, with the fitted ladder
  and the **off-ladder residuals highlighted**. This is the new centerpiece:
  it shows in one image why reduction and restoration trade off, and which
  transports beat the trade. Replaces the numeric content of `tab:transport`
  and the ladder `NOTE` on `tab:priced-trade`.
- **F2 — the operating map.** (content, rate) → recommended transport+restorer,
  shaded by the winner's margin in JND, with "no separable winner" as an
  explicit category rather than a blank. This is Plan A's deliverable and the
  article's answer to "what should I deploy?".
- **F3 — replication.** Per-video effect with a JND band, for the three
  breadth results. Shows at a glance that `bear`/`camel` win and `dog`/`pigs`
  do not — currently spread across `tab:av1-breadth`,
  `tab:conditioned-breadth`, `tab:goal2-breadth` as three separate tables the
  reader must cross-reference themselves.
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

1. **Goal renumbering — do this first, it touches everything.** The article
   defines three goals (selection, reduction, restoration) but numbers only
   two, and the numbers contradict the introduction's order: `evaluation.tex`
   calls bit relocation "Goal 1" and generative restoration "Goal 2", leaving
   selection unnumbered. Renumber to **1 = selection, 2 = reduction,
   3 = restoration** throughout, matching the introduction. Mechanical but wide
   (~26 marker blocks plus prose). The current text explicitly chose not to
   ("we keep those established labels rather than renumber every result") —
   that shortcut is a reviewer trap at submission.
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

## 6. Order of work

1. §4 cleanup (renumber → markers → log). Independent of Plan A, start now.
2. Plan A Wave 1 → the map exists.
3. F1 and F3, which need only existing data.
4. F2 and F4, which need the map and the cost axis.
5. Consolidations, once the figures carry the argument the tables used to.
6. Full read-through for narrative, since the argument will have changed shape.
