---
name: update-paper
description: Fold new PRESLEY findings (experiment results, diagnoses, retractions) into the paper (68e8b6bb11d0dd9e62a67aef/sections/*.tex), guided by its GOAL/HOLE/CLAIM markers. Use after experiments complete and results are committed/tested, or when a conclusion changes. Replaces the retired update-reports workflow.
---

# Folding findings into the paper

The paper is the primary living document. Its comment markers
(`STATUS/GOAL/HOLE/NOTE/NEXT/CLAIM(anchor):` — full spec in the paper repo's
`CLAUDE.md`) record each element's goal, missing data, and provenance.
`RESEARCH_LOG.md` (paper repo) is the secondary store for non-paper knowledge.

## Procedure

1. **Locate.** Run the discovery grep in the paper repo
   (`68e8b6bb11d0dd9e62a67aef/`):
   ```
   grep -n '^% *\(STATUS\|GOAL\|HOLE\|NOTE\|NEXT\|CLAIM\)(' main.tex sections/*.tex
   ```
   Find the anchors this finding touches; read the surrounding text and the
   owning file's `STATUS` header.
2. **Classify the impact:**
   - **Fills a `HOLE`** → proceed to write (step 3–4).
   - **Contradicts a rendered claim** → find its `CLAIM` provenance line;
     decide rewrite vs reframe-as-ablation vs cut (`\del{}`); never leave a
     stale number standing.
   - **Not paper-relevant** → one entry in `RESEARCH_LOG.md` (Open questions,
     Dead-end registry, or the append-only Log), stop.
3. **Gate the claim** before wording anything:
   - `presley-compare` decides whether a quality difference is real (JND
     table in `src/presley/compare.py`). Within-JND deltas are "no
     perceptible difference" — never a trend.
   - Degradation comparisons must be fixed-QP/CRF (`rate_control` field),
     never VBR. FG claims only from `foreground.lpips_mean` / `dists_fg`;
     FG-VMAF/FG-FVMD banned; FID only as `fid_fg_bbox`. Compare on
     `actual_bitrate_bps` (`transmitted_size_bytes` for presley_ai). Report
     the FG/BG split, never overall-only.
4. **Edit via the `paper-editor` agent** (it knows the file layout and
   macros). Hand it *verified* numbers and the exact result hashes — verify
   against `results/<hash>/result.json` yourself first (past reports have
   mis-summarized their own numbers; trust data over docs). Reviewer-visible
   text in `\rev{}`, removals in `\del{}`; **clear
   the `HOLE` and write/update the `CLAIM(id): src=<result hashes> date=`
   line in the same edit** — a HOLE may never be cleared without its data
   landing in the text.
5. **Cross-update:** if a referee item advanced, update
   `reviewers_comments.md` Status/Resolution (Done only when the text or
   experiment is actually in place). If a dead end or hard rule emerged, add
   it to `RESEARCH_LOG.md`; if a Standing-results entry just landed in the
   text, delete it from the log's queue.
6. **Verify:** no local TeX on this host — check balanced braces/environments
   in the edited file (unclosed `\rev{`/`\del{` especially); push and let
   Overleaf compile.
7. **Commit** the paper repo with a message naming the anchor ids and result
   hashes.

## Guardrails

- No number without a `results/<hash>` path (or `results/_superseded/`,
  explicitly flagged as superseded — see the log's registry before citing
  anything old).
- Never edit `archive/elvis-legacy.tex`; never strip `\rev{}`/`\del{}`
  wrapping; markers are comments and never wrapped in `\rev{}`.
- Camera-ready sweep rule: before final submission the discovery grep must
  return only `CLAIM` lines.
