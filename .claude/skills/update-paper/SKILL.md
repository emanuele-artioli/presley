---
name: update-paper
description: Fold new PRESLEY findings (experiment results, diagnoses, retractions) into the paper (68e8b6bb11d0dd9e62a67aef/sections/*.tex), guided by its GOAL/HOLE/CLAIM markers. Use after experiments complete and results are committed/tested, or when a conclusion changes. Replaces the retired update-reports workflow.
---

Read `/home/itec/emanuele/.agent-rules/skills/update-paper/SKILL.md` and follow it.

## PRESLEY specifics

- **Paper repo:** `68e8b6bb11d0dd9e62a67aef/` (separate git repo, Overleaf
  sync, own `CLAUDE.md`). Discovery grep:
  `grep -n '^% *\(STATUS\|GOAL\|HOLE\|NOTE\|NEXT\|CLAIM\)(' main.tex sections/*.tex`.
- **Citability backfill:** a result with no `invariant_failures` key predates
  the check — run `python -m presley.invariants results/` before citing it.
  A non-empty `invariant_failures` list makes the run uncitable regardless of
  how good the numbers look.
- **FG-citability rules:** cite `foreground.lpips_mean` (spatial-mode LPIPS
  over the true UFO mask) and `foreground.dists_fg` (mask-weighted DISTS,
  paired with `background.dists_bg`) — these are true region metrics.
  `foreground.fid_fg_bbox` is a bbox crop, not a FG metric — cite only as a
  corroborating signal, always by its full name. `foreground.vmaf_fg_bbox` /
  `vmaf_neg_fg_bbox` and `foreground.fvmd` are **banned** for the FG claim
  (bbox-crop based). Always compare on `actual_bitrate_bps`
  (`transmitted_size_bytes` for presley_ai), never `file_size_bytes`.
- **Quality-difference gating:** `presley-compare` (JND table in
  `src/presley/compare.py`) decides whether a delta is real — within-JND is
  "no perceptible difference," never a trend. Degradation comparisons must be
  fixed-QP/CRF, never VBR (`rate_control` field).
- **Revision tracking (presley IS a tracked revision):** reviewer-visible
  text added/changed goes in `\rev{...}`; removals use `\del{\sout{...}}`.
  Never silently strip existing `\rev{}`/`\del{}` wrapping. Comment markers
  are never wrapped.
- **Reviewer checklist:** `68e8b6bb11d0dd9e62a67aef/reviewers_comments.md` —
  update Status/Resolution when an item advances; Done only when the text or
  experiment is actually in place.
- Never edit `archive/elvis-legacy.tex` (read-only reference).
