---
name: update-reports
description: Fold new PRESLEY findings (experiment results, diagnoses, fixes, dead ends, next steps) into the technical reports in the paper repo, keeping the chain-status dashboard current. Use after running experiments, fixing a pipeline/encoder issue, or discovering anything a future session or the paper should know.
---

# Updating the PRESLEY technical reports

The reports live in the paper repo (`68e8b6bb11d0dd9e62a67aef/`) and are the
research source of truth. Entry point: `TECHNICAL_REPORTS.md` (index), which
also contains the canonical report template.

## Which report owns what

| Finding about… | Report |
|---|---|
| Codec ROI (kvazaar/svtav1/x265), QP mapping, rate control, presley_* degradation-ROI, RD sweeps of those | `TECHNICAL_REPORT_ROI_ENCODING.md` |
| ELVIS block removal, packing, in-painters (ProPainter/E2FGVI) | `TECHNICAL_REPORT_ELVIS_INPAINTING.md` |
| presley_ai degrade+restore, Real-ESRGAN/InstantIR, GPU ops, side-channel accounting | `TECHNICAL_REPORT_PRESLEY_AI.md` |
| Runner/hash model, evaluation metrics, masks, bitrate discipline, infra bugs that fake research results | `TECHNICAL_REPORT_PIPELINE_INFRA.md` |

If a finding genuinely fits none of these, create a new
`TECHNICAL_REPORT_<TOPIC>.md` from the template in the index and add it to
the index catalog — don't shoehorn.

## Update procedure

1. **Append a findings-log entry** to the owning report:
   `### YYYY-MM-DD — <title>` with **Problem/Question**,
   **Diagnosis/Evidence**, **Resolution**, **Paper impact**.
   Evidence means numbers, commands, hashes, file paths — enough that a
   skeptical reader could re-derive the conclusion. No entry without evidence.
2. **Refresh that report's "Current state (TL;DR)"** and its
   "Open questions & next steps" so they stay true after the new entry.
3. **Refresh the index** (`TECHNICAL_REPORTS.md`): the chain-status table if
   any link's evidence changed, the prioritized next steps, and the
   *Last updated* dates (index + touched reports).
4. If a new result **supersedes an old conclusion**: revise the TL;DR, but
   keep the old finding in the log and add a
   `**Superseded YYYY-MM-DD:** …` line pointing at the new entry. Never
   silently rewrite history — dead ends are paper material.
5. If results were invalidated by a code fix: move the affected
   `results/<hash>/` dirs to `results/_superseded/<hash>_<reason>/` (mv, not
   rm) before re-running, and mention the reason suffix in the log entry.

## Rules of evidence

- Chain-status claims cite FG metrics at **actual** bitrates
  (`actual_bitrate_bps` / `transmitted_size_bytes`), never targets.
- Single-bitrate-point results are marked as such; only sweeps (and
  eventually BD-rate) upgrade a link's status.
- Numbers must exist under `results/` (or `_superseded/`); if you quote a
  scratchpad-only diagnostic, say so explicitly in the entry.

## Reminders

- The paper repo is a separate git repo with its own CLAUDE.md conventions
  (`\rev{}`/`\del{}` etc. apply to `main.tex`, not these .md reports).
- If the finding also closes/advances a reviewer-comment item, follow up via
  the `/reviewer-response` skill so `reviewers_comments.md` stays in sync.
