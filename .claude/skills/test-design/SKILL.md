---
name: test-design
description: Propose and then write the tests for a component you added or changed in PRESLEY — behaviour cases, plausible-misuse cases, and an explicit list of what is deliberately not tested. Use after writing or modifying anything under src/presley/, and as the test step of a refactor PR. Surfaces the proposed list for approval before writing any test code.
---

# Designing tests for a PRESLEY component

The goal is a small number of tests that would actually fail if the code broke,
not coverage. This is research code: a test that cannot fail is worse than no
test, because it makes the suite look like it is watching something it is not.

## Workflow

1. **Read the change.** `git diff` for uncommitted work, or read the module.
   Identify what the component promises: its inputs, its outputs, and the
   invariants a caller relies on.

2. **Draft the list, in three groups.** Keep it short — most components deserve
   3–8 tests total.

   - **Behaviour** — the envisioned cases. What does this compute, and what is
     the answer for an input where you can state the expected value by hand?
     A metric with a closed form gets a hand-computed case; a mapping gets its
     boundary values.
   - **Plausible misuse** — what a caller in *this* repo could realistically do
     wrong: an empty mask, a mask at the wrong resolution, a config key that
     means something different per codec, a frame list shorter than expected.
     Prefer the mistakes that produce a *plausible number* over the ones that
     raise, because only the first kind survives to be cited.
   - **Deliberately not testing** — say what you are leaving out and why.
     Unreachable branches, third-party library behaviour, errors a caller
     cannot produce, and anything whose only effect would be to move the
     coverage number.

3. **Show the list to the user before writing any test code.** Number the items
   so they can say "drop 3, and add one for X". Do not skip this step even when
   the list looks obvious — the user knows failure modes the code does not show.

4. **Write only the approved tests.** Then run `pytest` and report the result
   plus the coverage delta.

## What makes a test worth writing here

Ask: *what would have to break for this test to fail, and is that a thing that
could plausibly happen?* If the answer is "nothing realistic", drop it.

The tests that have earned their place in this repo are the ones guarding
against a wrong answer that looks right:

- `annotate_experiments_yaml` once matched zero list items and rewrote the file
  unchanged while reporting success — the indent-style tests exist for that.
- An all-false mask makes the masked metrics score the *whole frame*, so an
  absent foreground mask yields a plausible "FG" number rather than an error.
- `derive_rate_control` reading `qp` as constant-QP for the presley_* methods
  would label VBR runs as fixed-QP and let the invariant check pass on exactly
  the configuration it exists to reject.

Each of those is a silent wrong answer. That is the category to hunt for.

## Where the test belongs

- Pure logic, no GPU or dataset → `tests/`, no marker; runs on every change and
  in CI.
- Needs a GPU, weights, or the dataset → mark `@pytest.mark.gpu` or
  `@pytest.mark.slow`; excluded by default.
- Checks a *paper claim* against real results rather than the code's behaviour →
  that is an invariant, not a unit test. Add it to `src/presley/invariants.py`
  so it runs on every experiment and gets recorded in `invariant_failures`,
  with its unit tests in `tests/test_invariants.py`.

## The living-test rule

Every diagnosed bug and every newly imagined edge case gets a test in the same
session it is diagnosed — the RESEARCH_LOG dead-end entry and the regression
test are written together. When a test is deleted, say why its failure mode is
now impossible.

## Coverage

The gate exists to stop untested code arriving unnoticed, not as a target. If a
module is omitted in `pyproject.toml`'s `[tool.coverage.report]` list, that is a
debt to repay: when you make part of it testable (usually by splitting the
GPU-bound half away from the pure half, as the evaluation split did), remove the
entry in the same PR. Never add an entry to make a number go up, and never write
a test whose only purpose is to raise one.
