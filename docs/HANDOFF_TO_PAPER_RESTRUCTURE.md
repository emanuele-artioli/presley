# Report to the paper-restructuring session

## What landed while you were restructuring

A session spun off your chip has **changed how every quality claim in the paper
is justified**, and has already edited `sections/evaluation.tex`,
`sections/presley.tex` and `RESEARCH_LOG.md`. Pull before you touch those files.

Commits: `presley@main` (code, docs) and `68e8b6bb11d0dd9e62a67aef@main`
(paper). Both pushed.

### The one-paragraph version

JND gating (hard rule 2) is right for a single comparison and too blunt for a
suite: a sub-JND effect that reproduces on every video is real even though it
stays imperceptible. There is now a **suite significance layer**
(`src/presley/suite.py`, `presley-compare --suite`) that complements JND
without ever overriding it, and hard rule 2 has been **amended in place as
2b**, not replaced. `compare.same_quality` is untouched, so **no landed
single-pair verdict changed**. Full audit: `docs/SIGNIFICANCE_AUDIT.md`.

### The finding that matters most for your text

**The exact two-tailed sign test floors at 2/2ⁿ.** Consequences you will hit
while restructuring:

- **n=2 floors at p=0.5.** Every restorer twin comparison in the paper is n=2.
- **n=5 floors at p=0.0625** — it can *never* reach α=0.05. Exact Wilcoxon has
  the same floor, so switching test is not an escape.
- **n=6** is the first size that can clear α uncorrected; **n=8** once
  corrected for the 6 restorer candidates tried against Real-ESRGAN.
- A widely-quoted figure in `docs/WAVE1_FALSIFIERS.md` (F3, on branch
  `feat/goals-wave0`, **not** fixed there) says 5/5 gives p=0.031. **That is
  the one-tailed value and is wrong for our use** — the direction was read off
  the data first. It is 0.0625. If your restructuring pulls F3 into the text,
  do not carry the 0.031.

### What changed in the paper (3 edits, all already applied)

1. **`tab:conditioned-twins`, `tab:conditioned-stream-diffvsr`, and the
   NAFNet-vs-`unsharp` row: "tie"/"near-tie" → "underpowered".** A tie asserts
   *evidence of no difference*; at n=2 the data assert nothing in either
   direction. **The conclusions are unchanged** (keep Real-ESRGAN, on
   throughput) — only the justification. Nothing was upgraded to a win.
2. **`tab:ablation`'s α/β row: strengthened.** The single claim the audit found
   *under*-stated. "No effect" is an equivalence claim, which a failed
   difference test can never support, so it now carries a TOST equivalence
   result over 18 invariant-clean runs (90% CI [+0.0165, +0.0269] dB, max|Δ|
   0.0507 dB against a 0.5 dB margin).
3. **"n>2" is now quantified as "n≥8"** in the open `HOLE`s.

### What you need to do

**Nothing is blocking.** But three things will make your restructure land clean:

- **Re-pull the three files** before editing. Your working copy is stale on all
  of them.
- **Don't reintroduce "tie", "wash" or "near-tie"** for any n=2 comparison as
  you move text around. That phrasing is now specifically prohibited by hard
  rule 2b, and it is the exact wording restructuring tends to reintroduce when
  compressing two paragraphs into one.
- **New `OPEN(sec:evaluation)` marker** in `evaluation.tex` lists every claim
  *not* yet re-run through the suite layer (`tab:bdrate`, `tab:av1`,
  `tab:goal2`, `tab:conditioned`, `tab:fillvariant`, `tab:breadth`,
  `tab:breadth-ext`, `tab:mask-sens`, `tab:mask-morph`, `tab:priced-trade`).
  None are known-wrong; they are simply unassessed. Preserve that marker —
  it is the to-do list for the next data-bearing session.

Two of those are worth flagging as genuinely promising, if your restructure
touches them: **`tab:breadth` (7/18) and `tab:breadth-ext` (5/9)** are *counts
across a suite*, which is precisely what a sign test is for — they may be the
paper's best candidates for a real significance statement. `tab:fillvariant`
(6/6 videos) is probably an equivalence claim rather than a difference one.

### For your own handoff

The useful thing to carry forward is small: **hard rule 2b in
`research-log/hard-rules.md`, plus `docs/SIGNIFICANCE_AUDIT.md`.** Both are
self-contained and neither needs this session's context. You do not need to
carry the implementation details of `suite.py` — the CLI prints the verdict, the
required n, and the mandated wording, so a fresh session can use it correctly
from the `--help` text and the skill entries alone (`results-report` and
`update-paper` skills were both updated).

If your context is long, the cheapest correct handoff is: "read
`research-log/hard-rules.md` rule 2b and `docs/SIGNIFICANCE_AUDIT.md` before
wording any quality comparison; never call an n=2 result a tie."

## Second update (2026-07-29): RESEARCH_LOG.md is now an index

A later session restructured the log for token cost. **This changes where you
read and write, not what anything says.**

- **`RESEARCH_LOG.md` is a 157-line index.** The bodies moved to
  `research-log/{hard-rules,standing-results,open-questions,bugs,dead-ends,
  operational}.md`. The move was content-only and verified byte-exact
  reassemblable, so nothing was reworded in the split itself.
- **Read the index, then open at most one file.** It carries every entry
  *title*, so "has X been tried?" is answerable without opening anything. The
  old file had grown to 1008 lines / ~17k tokens — past the point where it
  could be read in a single call at all.
- **When you add a finding:** append to the file that owns it and update its
  index line in the same commit. Files are capped at 300 lines.
- **Any path you have written down as `RESEARCH_LOG.md "Hard rules"` is now
  `research-log/hard-rules.md`.** Every pointer in `AGENTS.md`, both skills and
  three source comments was updated; if your restructure carries an old path,
  fix it rather than recreating the monolith.

### `standing-results.md` was drained — this one affects your text

The standing-results queue had never been drained, so it still listed ~20
results that are **already in the paper as `CLAIM(id)` lines**. Those entries
are now deleted. The CLAIM lines carry strictly more than the queue did
(hashes, operating point, numbers, "do not cite" caveats) and are never
deleted, so nothing was lost — but two consequences for you:

- **Do not treat the shorter `standing-results.md` as work having disappeared.**
  `grep -n '^% *CLAIM(' main.tex sections/*.tex` is now the authoritative list
  of what has landed.
- **Do not re-add a landed result to that file** as you move text around. What
  remains there is deliberately only: the not-yet-paper-ready α/β screens, the
  Goal-2 provenance archive (still the wording source of truth), the bmx-trees
  boundary case, the retired-trio retraction, and the open `HOLE`s.

The `n>2` sizing rule moved out of standing-results into `hard-rules.md` 2b,
since it constrains how you *size* an experiment rather than recording a
result. It still says n≥8 for restorer comparisons.

### Two figures that changed underneath you

- **Noise cost is +76.5…+83.0%**, not +213…+334%. Q9's matched-budget rematch
  supersedes the unbudgeted screen; the retirement conclusion is unchanged, only
  the citable magnitude. Corrected in
  `tools/noise_mode_decision_analysis.py`, `NOISE_MODE_DECISION_REPORT.md` (4
  places) and `src/presley/degradation.py:477`. If your restructure quotes the
  old range anywhere in the text, it is wrong.
- **`docs/EXPERIMENTS_QUEUED.md`** already carried the matched-budget number and
  was left alone.

### Paper-repo push state — check before you pull

The paper repo's `origin` is Overleaf. As of this handoff the research-log
split, the drain and the `CLAUDE.md` update are **committed on
`68e8b6bb11d0dd9e62a67aef@main` but may not be pushed yet** (pushing to Overleaf
is a user decision, not an agent one). If `git log origin/main..main` in that
repo is non-empty, the work is local-only — pull from the local checkout at
`/home/itec/emanuele/presley/68e8b6bb11d0dd9e62a67aef`, not from Overleaf, and
do not force anything over it.
