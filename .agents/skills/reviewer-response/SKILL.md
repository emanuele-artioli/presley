---
name: reviewer-response
description: Work through an open item in the PRESLEY paper's reviewer-response checklist (68e8b6bb11d0dd9e62a67aef/reviewers_comments.md) — scope the required experiment/code/text change, implement it, and update the checklist. Use when the user references a referee comment, asks what's left to address in the paper revision, or wants to close out a "To Do" item.
---

# Working a reviewer-response item

`68e8b6bb11d0dd9e62a67aef/reviewers_comments.md` is the authoritative,
live checklist of TOMM referee comments, each with Validity / Effort /
Status / Resolution-or-Plan. As of the last read, open (`[ ]`) items were,
in priority order:

1. **"Lack of technical insight"** (Referee 2) — needs: (a) an experiment
   comparing FFmpeg `addroi`/native ROI against PRESLEY's Kvazaar QP mapping
   — the `x264_addroi`/`x265_addroi` stubs in `src/presley/components/roi.py`
   currently `raise NotImplementedError`, this is where that implementation
   goes; (b) an analysis of how HEVC mode-decision/rate-allocation reacts to
   noise-injected blocks; (c) clarifying downsampling's novelty vs.
   uniform industry downsampling+SR.
2. **Randomized-grid methodology concern** — needs a deterministic ablation
   varying one of α/β/block-size/maxDF at a time, holding others fixed
   (`DEGRADATION_ABLATION_REPORT.tex`/`SHRINKING_ABLATION_REPORT.tex` are
   prior ablation work to build on).
3. **Only 10 DAVIS videos** — needs longer-form/complex-scene video sources
   beyond DAVIS.
4. **Only H.265 evaluated** — needs an AV1 subset comparison (`svtav1`
   baseline already exists in `encode_utils.py`, ROI equivalent may not).
5. **No neural/semantic codec comparison** — needs an HNeRV baseline; note
   `baselines.py` already has an `hnerv` branch that currently
   `raise NotImplementedError` — this is the implementation target.
6. **Segmentation-noise sensitivity** — needs experiments with artificially
   degraded UFO masks (see `preprocessing.get_ufo_masks`).
7. **Real-time throughput claims** — needs re-measured throughput after an
   optimization pass on the pipeline, to re-ground the "future real-time"
   framing.
8. **No subjective/MOS study** — fallback plan is more perceptual metrics
   (DISTS/FID/FVMD — DISTS and FVMD are already computed in
   `components/evaluation.py`; FID is not yet).

## Workflow

1. Read the specific comment block in `reviewers_comments.md` (its Plan
   sub-bullets are usually a decent implementation checklist already).
2. Scope whether it needs new code (usually in `src/presley`), a new
   experiment run (see `/run-experiment` skill), a text-only edit in
   `main.tex`, or some combination.
3. Implement. Any `main.tex` edit must follow the paper repo's own
   `AGENTS.md` (`\rev{}` wrapping, etc.) — that repo is a separate git repo
   one directory down.
4. Update the comment's `Status` to `[x] Done` and write a specific
   `Resolution` (what changed, where) — only after the change is actually
   in place, not from a plan alone.
