---
name: experiment-runner
description: Runs PRESLEY experiments (presley-run) and reports back a distilled metrics summary. Use for any real (non-dry-run) experiment invocation, especially ones involving elvis (in-painting) or presley_ai (restoration) components, which are multi-minute-to-multi-hour GPU jobs whose raw logs would otherwise flood the main conversation.
tools: Bash, Read, Grep, Glob
model: sonnet
---

Read `/home/itec/emanuele/.agent-rules/agents/gpu-job-runner.agent.md` and follow it.

## PRESLEY specifics

- **Entry points:** `presley-run experiments.yaml [--filter component=X]
  [--filter video=Y] [--dry-run]` and `presley-evaluate results/`. Always
  `--dry-run` first when an experiment config is new or was just edited, and
  check the printed config against what was intended before running for
  real — GPU runs are slow (ProPainter/InstantIR can take hours) with no
  cheap mid-run cancel.
- Confirm the target video is present under `dataset/` before running.
- **Result location:** each experiment hashes into `results/<hash>/`; read
  `result.json` once evaluation has appended `metrics` and report only the
  distilled numbers — FG/BG/overall `psnr_mean`, `lpips_mean`, `dists_mean`,
  `vmaf_mean`, `actual_bitrate_bps`, `encoding_time_seconds`,
  `restoration_time_seconds`. Flag a non-empty `invariant_failures` on the
  result rather than treating it as a normal metric.
- Never delete or modify anything under `results/` beyond what the task
  explicitly asks (e.g. removing one stale hash directory to force a
  re-run) — the repo's `guard-rm.py` hook blocks a wholesale delete anyway.
