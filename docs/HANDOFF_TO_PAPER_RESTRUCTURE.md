# Report to the paper-restructuring session

Paste the block below into that session. It is written to be read cold: it does
not assume that session has seen any of this work, and it is ordered so the
session can stop reading once it has what it needs for its own handoff.

---

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
`RESEARCH_LOG.md`, plus `docs/SIGNIFICANCE_AUDIT.md`.** Both are self-contained
and neither needs this session's context. You do not need to carry the
implementation details of `suite.py` — the CLI prints the verdict, the required
n, and the mandated wording, so a fresh session can use it correctly from the
`--help` text and the skill entries alone (`results-report` and `update-paper`
skills were both updated).

If your context is long, the cheapest correct handoff is: "read RESEARCH_LOG
hard rule 2b and docs/SIGNIFICANCE_AUDIT.md before wording any quality
comparison; never call an n=2 result a tie."
