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

## Environment

Conda-managed (`environment.yaml` + `install_openmmlab.sh`), Python 3.10,
CUDA-pinned PyTorch. Do not `pip install` ad hoc into it — dependency
versions here are pinned tightly on purpose (see pinned versions in
`pyproject.toml`) because several forked third-party models
(ProPainter/E2FGVI/Real-ESRGAN/InstantIR) are version-sensitive.

## No test suite

There is currently no automated test suite in this repo. Never report that
"tests pass" — verify by running a real (small/dry-run) experiment instead.

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

## Where to look for more

- Experiment workflow, filters, and reading back results → `/run-experiment` skill
- Reviewer-response checklist workflow → see the paper repo's own CLAUDE.md
- Degradation/restoration algorithm details, past debugging dead-ends
  (e.g. why FFmpeg `addroi` and x265 AQ modes don't work for per-block ROI,
  and why Kvazaar `--roi` does) → `68e8b6bb11d0dd9e62a67aef/TECHNICAL_REPORT*.md`
