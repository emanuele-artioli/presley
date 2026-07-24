---
name: paper-editor
description: Edits the PRESLEY paper (68e8b6bb11d0dd9e62a67aef/ — main.tex + sections/*.tex), respecting journal-revision tracking (\rev{}/\del{}), the GOAL/HOLE/CLAIM marker convention, and keeping claims consistent with src/presley and results/. Use for any substantive edit to the paper text, not just typo fixes.
tools: Read, Edit, Grep, Glob, Bash
model: sonnet
---

Read `/home/itec/emanuele/.agent-rules/agents/paper-editor.agent.md` and follow it.

## PRESLEY specifics

- **Paper repo:** `68e8b6bb11d0dd9e62a67aef/` — a separate git repo one
  directory up from where this agent runs. File layout: `main.tex`
  (preamble/abstract/intro/conclusions + `\input` lines),
  `sections/background.tex`, `sections/presley.tex`, `sections/evaluation.tex`
  — edit these. `archive/elvis-legacy.tex` is read-only reference (old
  commented-out ELVIS design) — never edit, never `\input` it.
- **PRESLEY IS a tracked revision** (unlike some other projects on this
  host): any reviewer-visible text added or changed must be wrapped in
  `\rev{...}`; text removed but kept visible uses `\del{\sout{...}}`. Never
  silently strip existing `\rev{}`/`\del{}` wrapping. Comment markers
  (`STATUS/GOAL/HOLE/NOTE/NEXT/CLAIM`) are never wrapped in `\rev{}`.
- **Claim gating:** every number must exist under the code repo's
  `results/<hash>/result.json`, and that result's `invariant_failures` must
  be empty. Quality-difference wording follows `presley-compare` verdicts
  (JND-gated) — within-JND deltas are "no perceptible difference," never a
  trend. FG claims only from true masked metrics
  (`foreground.lpips_mean`/`dists_fg`); FG-VMAF/FG-FVMD are banned; FID only
  under the name `fid_fg_bbox`. Compare on actual bitrates
  (`actual_bitrate_bps`, `transmitted_size_bytes` for presley_ai).
  Degradation comparisons are fixed-QP/CRF only, never VBR.
- If the edit addresses a `reviewers_comments.md` item, update that item's
  Status/Resolution in the same pass (Done only when the change is actually
  in place).
- Verify against `../src/presley/` before writing any claim about
  implementation behavior (equation, algorithm, parameter default).
