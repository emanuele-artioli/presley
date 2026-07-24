---
name: test-design
description: Propose and then write the tests for a component you added or changed in PRESLEY — behaviour cases, plausible-misuse cases, and an explicit list of what is deliberately not tested. Use after writing or modifying anything under src/presley/, and as the test step of a refactor PR. Surfaces the proposed list for approval before writing any test code.
---

Read `/home/itec/emanuele/.agent-rules/skills/test-design/SKILL.md` and follow it.

## PRESLEY specifics

- **Test command:** `pytest` (fast tier, no marker — CPU only, no GPU/data);
  `pytest -m invariants` for goal checks against real `results/<hash>/` dirs.
  `pytest.ini` excludes `gpu`, `slow`, `integration`, `invariants` by default.
- **Coverage gate:** defined in `pyproject.toml`'s `[tool.coverage.report]`
  omit list — holds only GPU-bound modules. Ratchet up as real tests land,
  never down to accommodate untested code.
- **Tiers in this repo:**
  1. Unit — pure logic: experiment hashing, config dispatch, masked metrics,
     encode helpers, the JND comparison (`tests/`, no marker).
  2. Stage-contract — each `src/presley/components/*.py` checks its own
     output as it produces it (dimensions/duration match source, mask
     coverage sane, a promised size reduction actually happened).
  3. Goal-invariant (`-m invariants`, `src/presley/invariants.py`) — checks
     the paper's claims on real runs: Goal 1 (fixed-QP degradation frees bits
     without hurting FG), Goal 2 (restoration improves BG toward the
     original), and the structural check that no degradation experiment used
     VBR. Violations land in that run's own `result.json` under
     `invariant_failures`; a non-empty list makes the run uncitable. Add its
     unit tests in `tests/test_invariants.py`.
- **The silent-wrong-answer bar (this repo's real examples):**
  `annotate_experiments_yaml` once matched zero list items and rewrote the
  file unchanged while reporting success; an all-false mask makes the masked
  metrics score the whole frame instead of erroring; `derive_rate_control`
  misreading `qp` as constant-QP for `presley_*` methods would let a VBR run
  pass the fixed-QP invariant check.
- Run `/code-review` after non-trivial changes under `src/presley/`, and
  follow it with a real small experiment showing actual command + output —
  tests alone are not sufficient evidence a pipeline change works.
