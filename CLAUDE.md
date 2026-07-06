# PRESLEY

Research pipeline for perceptual video compression: degrade less-important video
regions server-side (QP mapping, downsampling, or noise injection) and restore
them client-side with generative models (Real-ESRGAN, InstantIR, ProPainter,
E2FGVI). Extends prior work ELVIS (block removal + in-painting). Companion
paper lives in [68e8b6bb11d0dd9e62a67aef/](68e8b6bb11d0dd9e62a67aef/) — a
separate git repo (Overleaf sync, gitignored here) with its own CLAUDE.md and
its own conventions. Don't apply this file's rules there.

## Entry points — use these, not cli.py

```
presley-run experiments.yaml [--filter component=X] [--filter video=Y] [--dry-run]
presley-evaluate results/
```

`presley-run` (`src/presley/runner.py`) dispatches each entry in
`experiments.yaml` to `src/presley/components/{baselines,roi,elvis,presley_ai}.py`
based on its `component` field, then calls evaluation automatically unless
`--dry-run`. Always try `--dry-run` first when adding or editing experiments —
GPU runs are slow (ProPainter/InstantIR can take hours) and there is no
cheap way to cancel mid-run cleanly.

**`src/presley/cli.py` and `src/presley/pipeline_legacy.py` are dead code**
left over from before the `components/` refactor — they import
`presley.config`, a module that no longer exists, so they are already broken.
Do not use them as a reference for "how the pipeline works" and do not try to
fix their imports; if asked to clean up, deleting them is correct.

## Experiment/result model

Each experiment dict in `experiments.yaml` is hashed
(`compute_experiment_hash`) into `results/<hash>/result.json`. The runner
skips any hash that already has a `result.json` — so re-running after editing
`experiments.yaml` is always safe and never silently overwrites a prior
result. If a result looks stale, delete the specific `results/<hash>/`
directory rather than the whole `results/` tree.

Every `presley-run` invocation (including `--dry-run`) refreshes a `# hash:
<id>` comment above each entry in `experiments.yaml` so you can map an entry to
its `results/<id>/` dir without guessing; `presley-run experiments.yaml
--annotate-only` just refreshes those comments and exits. The hash is computed
excluding any `hash`/`_`-prefixed keys, so the annotation never perturbs it.

## Evaluation methodology — every experiment has a comparison target

The claim under test is the chain **presley_ai > elvis > roi > baseline** on
*foreground* quality at matched bitrate. Never report only overall metrics —
the `metrics.foreground`/`metrics.background` split is the point. Analyze each
component against its designated target:

- **Codec ROI methods** (`kvazaar`, `x265_aq`, `svtav1`) vs the **same codec's
  baseline** at comparable bitrate. Expected signature: FG quality ↑, BG
  quality ↓. If it's absent, assume our usage is wrong before blaming the
  codec — "codec X doesn't implement ROI correctly" is a strong claim needing
  evidence beyond reasonable doubt (see TECHNICAL_REPORT for past false alarms).
- **presley_* ROI methods** (mask-driven degradation before encoding) vs the
  codec ROI methods: does direct block-level control buy more FG quality, and
  at what BG cost?
- **elvis** vs baselines, same analysis as ROI: did dropping removable blocks
  leave more bits for FG blocks at the same bitrate?
- **presley_ai** vs all of the above: FG quality must be best-in-class at
  matched bitrate, and the bitrate accounting must use
  `transmitted_size_bytes` (video + side-channel strength maps), not just the
  video file.

Exact bitrate matches are rare: compare at *similar actual* bitrates
(`actual_bitrate_bps`, not `target_bitrate`) for preliminary conclusions, and
use BD-rate curves (multiple target bitrates per method) for paper-grade
claims. If a link in the chain breaks, first search for regimes where it holds
(video subsets, bitrate ranges, codecs, parameters) before concluding the
method is worse — and only after that, re-examine the implementation.

**Fast iteration:** `presley-run … --fast-metrics` / `presley-evaluate
results/ --fast-metrics` compute only FG/BG/overall PSNR/SSIM/MSE and skip the
slow metrics (LPIPS/DISTS/VMAF/FVMD and block-level maps). Fast-only results
are tagged `metrics.fast_only` and get upgraded in place by a later full
`presley-evaluate results/`.

## Environment

Conda-managed (`environment.yaml` + `install_openmmlab.sh`), Python 3.10,
CUDA-pinned PyTorch. Do not `pip install` ad hoc into it — dependency
versions here are pinned tightly on purpose (see pinned versions in
`pyproject.toml`) because several forked third-party models
(ProPainter/E2FGVI/Real-ESRGAN/InstantIR) are version-sensitive.

**Host:** work runs on a shared remote Linux **GPU server, no root/sudo**.
Never reach for `apt` or other system installs — install any extra tooling
with conda (Miniconda is at `/usr/local/miniconda3`) into a *separate* env, not
the pinned `presley` env. Home is `/home/itec/emanuele`. `git push` already
works via a stored credential helper, so GitHub PRs/connectors/`gh` are not
needed for this solo workflow.

## No test suite

There is currently no automated test suite in this repo. Never report that
"tests pass" — verify by running a real (small/dry-run) experiment instead,
and **show the evidence**: the exact command and its output, not an assertion
that it worked. After a non-trivial change under `src/presley/`, run
`/code-review` before treating it as done — it reviews the working diff in a
fresh subagent that never saw your reasoning, which is the closest thing to a
test this repo has.

## `results/` is gitignored — deletion is unrecoverable

`results/` (and `cache/`) are in `.gitignore`, so they are **not** in git
history — a wholesale `rm` cannot be undone with git. The expensive
preprocessing (`cache/`: reference frames, EVCA scores, UFO masks) is
regenerable but slow; the GPU restoration outputs in `results/` cost hours to
recompute. A `.claude/hooks/guard-rm.py` PreToolUse hook blocks `rm` against
the whole `results/`/`dataset/`/`cache/` tree for this reason. Never test a
destructive command against these real directories.

## This tooling is meant to evolve

The `.claude/` directory (this file, `skills/`, `agents/`, `hooks/`,
`settings.json`) is part of the working setup, not frozen. If during work we
find a convention Claude gets wrong twice, a skill that would help, a hook
worth adding, or a rule that's stale — add/update/remove it in `.claude/`
right then. Note: edits to `settings.json`/hooks only take effect on the next
session (open `/hooks` or restart to reload); skills and CLAUDE.md load fresh
each session too, so prefer those for anything you want to rely on immediately.

## Technical reports are the research source of truth

`68e8b6bb11d0dd9e62a67aef/TECHNICAL_REPORTS.md` (paper repo) is the dashboard
for the whole experimental effort: chain-status table, prioritized next
steps, and the catalog of topic reports (ROI encoding, ELVIS in-painting,
PRESLEY AI restoration, pipeline/evaluation infra), each with a standardized
findings log. **Read the relevant topic report's TL;DR before working on that
area** — it records what's already been tried, fixed, and disproven (e.g. why
ROI must run in fixed-QP/CRF mode, why addroi/x265-AQ can't do semantic ROI).
After any experiment run or diagnosis that produces new knowledge, fold it
back in with the `/update-reports` skill.

## Where to look for more

- Experiment workflow, filters, and reading back results → `/run-experiment` skill
- Summarizing/comparing results → `/results-report` skill
- Folding new findings into the technical reports → `/update-reports` skill
- Reviewer-response checklist workflow → see the paper repo's own CLAUDE.md
- Degradation/restoration algorithm details, past debugging dead-ends →
  topic reports catalogued in `68e8b6bb11d0dd9e62a67aef/TECHNICAL_REPORTS.md`
