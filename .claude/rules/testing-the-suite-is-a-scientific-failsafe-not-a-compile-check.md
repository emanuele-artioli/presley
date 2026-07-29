---
paths:
  - "tests/**"
  - "pytest.ini"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Testing — the suite is a scientific failsafe, not a compile check' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

## Testing — the suite is a scientific failsafe, not a compile check

```
pytest                       # fast tier: pure logic, CPU only, no GPU/data
pytest -m invariants         # goal checks against real results/<hash>/ dirs
```

`pytest.ini` excludes the `gpu`, `slow`, `integration` and `invariants`
markers by default, so the bare command stays fast enough to run on every
change. CI runs the fast tier plus a coverage gate; the gate's omit list
holds only the GPU-bound modules, so the number means what it says. Ratchet
the threshold up as real tests land — never down to accommodate new untested
code.

Three tiers, each catching a different kind of wrong:

1. **Unit tier** — behavior and misuse of pure logic: experiment hashing,
   config dispatch, the masked metrics, encode helpers, the JND comparison.
2. **Stage-contract tier** — each component checks its own output as it
   produces it (degraded video matches the source's dimensions and duration,
   mask coverage lands in a sane range, a promised size reduction actually
   happened), so a broken stage fails there rather than surfacing as a
   strange metric hours later.
3. **Goal-invariant tier** (`-m invariants`) — checks the *paper's* claims on
   real runs: Goal 1 (at fixed QP, degradation frees bits without hurting FG)
   and Goal 2 (restoration improves BG toward the original), plus the
   structural check that no degradation experiment used VBR. Violations are
   written into that run's own `result.json` under `invariant_failures`, and
   **a run with a non-empty `invariant_failures` is never citable** — re-check
   it before it reaches a report or the paper.

Tests are necessary but not sufficient: after a pipeline change also run a
real small experiment and **show the evidence** — the exact command and its
output, not an assertion that it worked. Run `/code-review` after non-trivial
changes under `src/presley/`.

**Every diagnosed bug or newly imagined edge case gets a test in the same
session it is diagnosed** — the `research-log/dead-ends.md` entry and the
regression test are written together. Deleting a test requires saying why its
failure mode is now impossible.

Research code, so keep tests honest and thin: cover envisioned behavior and
plausible misuse of code we own, not unreachable branches, third-party
library behavior, or errors a caller cannot produce. **A test that exists
only to raise the coverage number is a defect.** The `/test-design` skill
proposes a test list for review before writing any of it.
