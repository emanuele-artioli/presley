---
paths:
  - "src/**"
  - "scripts/**"
  - "config/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Experiment/result model' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

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
