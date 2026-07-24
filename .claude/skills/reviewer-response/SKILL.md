---
name: reviewer-response
description: Work through an open item in the PRESLEY paper's reviewer-response checklist (68e8b6bb11d0dd9e62a67aef/reviewers_comments.md) — scope the required experiment/code/text change, implement it, and update the checklist. Use when the user references a referee comment, asks what's left to address in the paper revision, or wants to close out a "To Do" item.
---

Read `/home/itec/emanuele/.agent-rules/skills/reviewer-response/SKILL.md` and follow it.

## PRESLEY specifics

- **Checklist:** `68e8b6bb11d0dd9e62a67aef/reviewers_comments.md` — the
  authoritative, live TOMM referee checklist (Validity / Effort / Status /
  Resolution-or-Plan per item). Always re-read it and the paper's
  `GOAL/HOLE` markers rather than trusting a summary of what's open — it
  moves. As of 2026-07-18 the specific open/done items were: "lack of
  technical insight" (text-only, transport-diagnosis story), randomized-grid
  ablation (experiment done, text pending `GOAL(tab:ablation)`), 10-DAVIS-only
  breadth (open, `HOLE(sec:evaluation)`), H.265-only (both regimes done, text
  pending `GOAL/HOLE(tab:av1)`), neural-codec comparison (done, text pending
  `GOAL(tab:hnerv)`), segmentation-noise sensitivity (open, runs with the
  breadth campaign), real-time throughput (reframe as starved-regime
  proof-of-concept, not optimize, `HOLE(fig:presley-speed)`), no MOS study
  (metric evidence rebuilt; small pairwise study pending a decision). Verify
  current state against the checklist itself before acting on any of this.
- Paper edits go through the `/update-paper` skill or the `paper-editor`
  agent (this repo's copy, not the global one) into `sections/*.tex`,
  following `\rev{}`/`\del{}` wrapping and the marker convention — the paper
  repo is a separate git repo one directory down with its own `CLAUDE.md`.
- Experiment runs go through the `experiment-runner` agent or `presley-run` —
  check the paper's `HOLE()` markers first so only paper-needed experiments
  run.
