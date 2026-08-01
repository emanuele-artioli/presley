# Handoff — resubmission push: R2 answered, two jobs still running (2026-08-01)

Supersedes the 2026-07-31 handoff. Its open items are done or carried below.
Working plan: `~/.claude/plans/prepare-a-plan-to-quiet-crown.md` (approved;
wave structure W0/W1/W2).

## State (verified at write time)

- Code `main` @ **`274767c`** plus later commits, pushed, CI green (CI caught a
  real `ARCHITECTURE.md` gap — trust it over local green).
- Paper pushed to Overleaf; **every edit is `\rev{}`/`\del{}`-tracked**.
- **Two jobs running.** Neither is finished; see "Running" below.
- Results DB ~955 runs.

## What landed this session

| | outcome |
|---|---|
| **Saturation invariant** | new-runs-only, `metrics.saturation` + `_check_output_not_saturated`; NAFNet defect closed in `bugs.md` |
| **Baselines QP 42/47** | 18 runs, monotone 9/9, no invariant failures; `tab:breadth-ratematched` unchanged (it never needed them) |
| **Claim (b)** | `probe_block_damage`, 8 runs → `CLAIM(block-damage)`; **`NEXT(sec:implementation)` cleared in full** |
| **0c** | narrowed, still open: ρ's CI includes zero; three explanations refuted |
| **12.4 dB audit** | was pooled; re-derived within-run to **11.6 dB**, R = 6.96 so "some 7 times" stands |
| **R2 "technical insights"** | **Done** — `sec:insight` landed; contribution bullets 1–2 + abstract reframed |
| **No-MOS statement** | landed; `background.tex` stops promising VMAF/FVMD |

### Two numbers changed, both the same class of error

`6.2/8.2 dB` → **4.9/8.4 dB** and `12.4 dB` → **11.6 dB**. Both were pooled
across runs while worded as within-run. **The generalizable lesson is in
`dead-ends.md`:** pooling cost 1.07× in the second case and much more in the
first, so *"it was pooled" is not itself proof a figure is wrong* — the
inflation has to be measured. If you find a third dispersion figure, measure it
before rewriting it.

## Running — check artifacts, not notifications

1. **HOLE wave A** — `config/holes_wave_a.yaml`, 22 runs (H4 shrink_amount arm +
   H1 stage-1 QP calibration). Log `/tmp/claude-1041/holes_wave_a.log`.
   At write time 4/22 evaluated.
2. **FG-metric corpus audit** — `tools/audit_fg_metric_bbox.py`, log
   `/tmp/claude-1041/fg_audit.log`. ~102 of ~899 rows; ~5 h total. Read-only,
   writes nothing into `results/`.

`!! 1 INVARIANT FAILURE — metrics block is missing or empty` in a runner log is
the documented pre-evaluation sticky verdict, **not** a failure. It clears when
evaluation writes metrics.

## Next, in order

### 1. When the audit finishes → complete `CLAIM(fg-metric-audit)`

It is marked **PARTIAL** in `sections/evaluation.tex` and carries only the camel
figures (masked 0.1639 vs bbox 0.1404; bbox area 66.8% vs true FG 14.4%, 6
hashes). Fill in the corpus-wide per-video table and the winner-flip count, then
drop the PARTIAL. **The old "13 of 43 groups, 9 away from baselines" figures may
not be used until re-derived** — they predate `drop_unionbbox_keys` deleting the
values they came from, and the marker says so.

### 2. W0-D — the Real-ESRGAN speedup benchmark (not started)

**Deliberately not started: the GPU has two jobs on it, and a throughput
benchmark under contention measures the contention.** Wait for a quiet GPU.
Pre-registered: reportable only if **≥1.2×** *and* BG-LPIPS stays within JND of
the fp32 result. A null closes the item honestly. Then rewrite the unmeasured
"quantization, distillation, NPUs" sentence in the Conclusions with whatever was
actually measured, and clear the matching clause of `NEXT(sec:conclusions)`.
This is the **last blocking piece of W1**.

### 3. W0-A — the remaining three HOLE sets

Design and pre-registered bounds for all four: **`docs/HOLE_CLOSURE_WAVE.md`**.
Wave A covers H4 + H1 stage 1. Still to append and launch (generator at
`/tmp/claude-1041/append_holes.py`, sets `h3`, `h1b`, `h2`):

- **H1 stage 2** needs QPs *chosen from* stage 1's baseline sweep — fill `QPS`
  in the generator. Do not copy bear's ladder; recalibration is the point.
- H3 (`tab:conditioned`, 32 runs, BD-rate curve) and H2 (`tab:goal2`, 32 runs,
  8 of them ProPainter ≈ 7 h).
- ⚠ **Never run `--filter component=presley_ai`.** `experiments.yaml` holds 36
  unrelated pending entries, including VBR-era ones hard rule 1 forbids and the
  `dc_vsr` stub that raises by design. Use a scoped run-file, as wave A does.

### 4. W2 — close-out

Fold H1–H4 into their `HOLE`s (a HOLE may only be cleared by the edit that lands
its data); throughput + perceptual items → Done in `reviewers_comments.md`;
final sweep that **every dispersion figure is within-run or labelled pooled**.

## Gotchas

**New this session:**

- **The `Bash` tool's own `timeout` is the one that matters** (ms, max 600000);
  a foreground command that exceeds it is moved to background, not killed —
  which is how the claim-(b) smoke test became the full 8-run job.
- **A wrapper shell's "completed" notification is not the job.** `nohup … &`
  inside a harness `Bash` call returns as soon as the wrapper exits; the runner
  lives on. Check result dirs and `pgrep` — and note `pgrep -f <pattern>`
  matches the wrapper whose command line *contains* the pattern.
- **The editable install points at the main checkout**, so tests run from a
  worktree import the *other* tree's `presley` unless you set
  `PYTHONPATH=<worktree>/src`. This silently made new-code tests pass against
  old code once already.
- Two edits went to the main checkout instead of the worktree; a tool importing
  a module that exists only on a branch is the symptom.

**Carried forward:** filter fixed-QP with `codec=svtav1`; NAFNet / Real-HAT-GAN
reject `fp32=False`; Stream-DiffVSR has its own env; Q10 morph A/B is UFO none;
`fg_protect=True`; BG-PSNR is never the Goal-2 verdict; `experiments.yaml` is
appended as text and never re-dumped; never `rm` `results/`/`dataset/`/`cache/`.

## Still open, unchanged

- **Q6 DiffBIR — ASK before any wire** (no mechanism argument → leave deferred).
- **Q8 DC-VSR — blocked upstream** (weights-only; the stub's `RuntimeError` is
  correct).
- **0c** — `drift-straight` unexplained; the cost model may not be called
  uniformly adequate. `tools/analyze_drift_straight_0c.py`, no GPU.
- 36 stale pending `presley_ai` entries in `experiments.yaml` want triage.
- `open-questions.md` (310) and `dead-ends.md` (306+) are over the log's own
  300-line ceiling; drained what this work made drainable, the rest needs a
  judgement call on older entries.

## Landmarks

| Path | Why |
|---|---|
| `~/.claude/plans/prepare-a-plan-to-quiet-crown.md` | the approved plan, W0/W1/W2 |
| `docs/HOLE_CLOSURE_WAVE.md` | bounds for all four HOLEs, pre-registered |
| `docs/CLAIM_B_BLOCK_DAMAGE.md` | claim (b) design, result, and the two alarms that fired on my own bounds |
| `sections/evaluation.tex` → `sec:insight` | the R2 answer |
| `research-log/dead-ends.md` | both superseded dispersion figures, with the lesson |
| `tools/analyze_within_run_spread.py` | within-run vs pooled, reusable |
