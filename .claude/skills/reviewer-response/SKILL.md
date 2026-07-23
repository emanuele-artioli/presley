---
name: reviewer-response
description: Work through an open item in the PRESLEY paper's reviewer-response checklist (68e8b6bb11d0dd9e62a67aef/reviewers_comments.md) — scope the required experiment/code/text change, implement it, and update the checklist. Use when the user references a referee comment, asks what's left to address in the paper revision, or wants to close out a "To Do" item.
---

# Working a reviewer-response item

`68e8b6bb11d0dd9e62a67aef/reviewers_comments.md` is the authoritative,
live checklist of TOMM referee comments, each with Validity / Effort /
Status / Resolution-or-Plan. State as of 2026-07-18 (always re-read the
checklist and the paper's `GOAL/HOLE` markers — they carry the per-item
plans now):

1. **"Lack of technical insight"** (Referee 2) — text task, no new
   experiment: the answer is the transport-diagnosis + VBR-laundering +
   starved-regime story (see `RESEARCH_LOG.md` and `GOAL(tab:transport)`,
   `GOAL(tab:ratecontrol)` in `sections/evaluation.tex`). The item's old
   Plan (addroi comparison, noise-injection analysis) predates the bridge
   pivot — the addroi/x265-AQ inability is already evidenced in the
   dead-end registry, and noise is retired.
2. **Randomized-grid / controlled ablation** — experiment DONE (2026-07-13:
   α/β near-zero impact, block size 16 optimal); paper text pending
   (`GOAL(tab:ablation)`).
3. **Only 10 DAVIS videos** — open; the breadth + explanatory-axis campaign
   (screen unused DAVIS videos, add non-DAVIS longer clips with
   segmenter-derived masks) — `HOLE(sec:evaluation)`.
4. **Only H.265 evaluated** — experiments done in both regimes (comfortable
   loses, starved wins; bear+camel); text pending (`GOAL/HOLE(tab:av1)`).
5. **Neural-codec comparison** — DONE (HNeRV 10–70× more bitrate at equal
   FG-PSNR); text pending (`GOAL(tab:hnerv)`).
6. **Segmentation-noise sensitivity** — open; runs with the breadth campaign
   (degraded UFO masks on DAVIS + segmenter masks on non-DAVIS; erosion is
   the dangerous direction, it defeats `fg_protect`).
7. **Real-time throughput claims** — reframe, don't optimize: measure
   ProPainter/E2FGVI fps once at the final operating point
   (`HOLE(fig:presley-speed)`), reframe text to starved-regime
   VOD/edge-caching proof-of-concept.
8. **No subjective/MOS study** — metric evidence rebuilt (FG-LPIPS + masked
   FG-DISTS; VMAF/FVMD excluded — see RESEARCH_LOG hard rules); a small
   pairwise study is decided after the Goal-2 probe (user decision
   2026-07-18).

## Workflow

1. Read the specific comment block in `reviewers_comments.md` (its Plan
   sub-bullets are usually a decent implementation checklist already).
2. Scope whether it needs new code (usually in `src/presley`), a new
   experiment run (see `/run-experiment` skill — check the paper's `HOLE()`
   markers first), a text-only edit in the paper, or some combination. If an
   experiment must come first, record it as a `HOLE()` marker at the target
   paper location, not only in the checklist.
3. Implement. Paper edits go through `/update-paper` (or the `paper-editor`
   agent) into `sections/*.tex`, following the paper repo's `CLAUDE.md`
   (`\rev{}` wrapping, marker rules) — that repo is a separate git repo one
   directory down.
4. Update the comment's `Status` to `[x] Done` and write a specific
   `Resolution` (what changed, where) — only after the change is actually
   in place, not from a plan alone.
