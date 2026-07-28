---
paths:
  - "pyproject.toml"
  - "environment.yaml"
  - "setup.py"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Environment' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

## Environment

Conda-managed (`environment.yaml` + `install_openmmlab.sh`), Python 3.10,
CUDA-pinned PyTorch. Do not `pip install` ad hoc into it — dependency
versions here are pinned tightly on purpose (see pinned versions in
`pyproject.toml`) because several forked third-party models
(ProPainter/E2FGVI/Real-ESRGAN/InstantIR) are version-sensitive.

**Exception — YOLOE masks:** `mask_source: yolo` needs `ultralytics` in the
same `presley` env. Install only via the optional `[yolo]` extra with
`--no-deps` control (see `pyproject.toml`) so torch/diffusers/transformers
are not upgraded. Checkpoint: `/home/itec/emanuele/Models/YOLO/yoloe-11l-seg.pt`.
Cache warmer: `conda run -n presley python tools/generate_yolo_masks.py`.

**Host:** work runs on a shared remote Linux **GPU server, no root/sudo**.
Never reach for `apt` or other system installs — install any extra tooling
with conda (Miniconda is at `/usr/local/miniconda3`) into a *separate* env, not
the pinned `presley` env (YOLO above is the documented exception). Home is
`/home/itec/emanuele`. `git push` already works via a stored credential
helper, so GitHub PRs/connectors/`gh` are not needed for this solo workflow.
