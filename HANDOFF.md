# Handoff — three HOLE sets left to run; everything else is closed (2026-08-02)

Supersedes the 2026-08-01 handoff. **Your job is W0-A's remaining three sets and
the W2 close-out that follows them.** Nothing else is outstanding: all reviewer
items are `[x] Done`, the writing pass has landed, and no job is running.

Approved plan: `~/.claude/plans/prepare-a-plan-to-quiet-crown.md`.
Bounds for every set below: **`docs/HOLE_CLOSURE_WAVE.md`** — read it first;
it is pre-registered and two of its bounds have already needed revising *with a
stated reason*, which is the convention to follow if yours fire.

## State (verified at write time)

- Code `main` clean and pushed, **CI green**. ~342 tests.
- Paper pushed to Overleaf. Every edit `\rev{}`/`\del{}`-tracked; comment-only
  edits verified as such.
- **Nothing running.** `nvidia-smi` may show another user's process — check PIDs.
- Results ~955 runs; `experiments.yaml` 800 entries, 20 of them `_retired`.

## Your task: H1 stage 2, H2, H3

Generator: `/tmp/claude-1041/append_holes.py` (sets `h1b`, `h2`, `h3`). If that
scratch file is gone, it is small — regenerate from the configs in
`docs/HOLE_CLOSURE_WAVE.md`. **Fill in `QPS` first:**

```python
QPS = {"dog": [50, 55, 60, 63], "pigs": [50, 55, 60, 63]}
```

Those come from the stage-1 calibration that already ran (table in the design
doc). They were **chosen to bracket the incumbents' PSNR range, not copied from
bear's ladder** — recalibration is the point of the exercise.

⚠ **`pigs` cannot be starved as hard as bear/camel.** At QP 63, the codec
maximum, its baseline is still ~1 dB above where the incumbents end. Keep the
rungs (there is no QP left) but **do not read a weaker regime effect on `pigs`
as content-dependence** without saying that first.

| set | what | runs | cost |
|---|---|---|---|
| H3 | `tab:conditioned`: 2 videos × 4 rungs × 4 arms, BD-rate curve | 32 | ~1 h |
| H1b | `tab:av1`: baselines + elvis{blackout,freeze} bs16 | 24 | ~13 h (16 ProPainter) |
| H2 | `tab:goal2`: 2 new videos + a 2nd QP on bear/camel | 32 | ~7 h (8 ProPainter) |

Run cheapest-first (H3 → H1b → H2) so a broken config surfaces early.

### How to run them without stepping on a landmine

- **Use a scoped run-file**, as `config/holes_wave_a.yaml` did. Do not run
  `--filter component=presley_ai` — 16 legitimate but unrelated fixed-QP
  controls are still pending and would come along.
- Append to `experiments.yaml` **as text**; verify entry-count delta, prior
  hashes byte-identical, all distinct. Never re-dump the file.
- Launch detached. **`nohup … &` inside a harness `Bash` call returns as soon as
  the wrapper exits — that notification is not the job finishing.**
- ⚠ **`pgrep -f <pattern>` matches your own checking command.** This bit me
  twice this session, most recently reporting a finished LPIPS backfill as still
  running for half an hour. Check log mtimes and result dirs, not `pgrep`.
- `!! 1 INVARIANT FAILURE — metrics block is missing or empty` in a runner log
  is the documented pre-evaluation sticky verdict, not a failure.
- **Region LPIPS is not written by the standard evaluation.** After the runs,
  `presley-evaluate results/ --backfill-lpips` before any BG-LPIPS bounds check,
  or you will find the field missing and think something broke.

### Then W2

Fold each set into its `HOLE` — **a `HOLE` may only be cleared by the edit that
lands its data.** `HOLE(tab:priced-trade)` can be cleared now from the H4 result
already in the design doc. Finish with the sweep that every dispersion figure in
the manuscript is within-run or explicitly labelled pooled; that class of error
has now been found twice.

## What landed since the last handoff

| | outcome |
|---|---|
| **R2 "technical insights"** | `sec:insight` — concedes Eqs 1–3 are a heuristic, relocates the contribution to the objective. Item **Done** |
| **Throughput** | benchmarked: fp16 is already the shipped default, fp32 is **1.5× slower**, retiling worse. Null result, item **Done** |
| **Perceptual** | corpus audit + first explicit "no MOS study" statement. Item **Done** |
| **Stale queue** | 20 entries `_retired` in place (14 VBR/rule-1, 4 noise, 2 dc_vsr); pending `presley_ai` 48 → 16 |
| **H4** | done, in the design doc; the budget lever is **inert below sa≈0.25** and `tab:priced-trade` sits exactly at that knee |

### Three numbers changed under audit — expect a fourth

- `6.2/8.2 dB` → **4.9/8.4 dB** (pooled, worded as within-run)
- `12.4 dB` → **11.6 dB** (same defect; `R = 6.96` so "some 7 times" survived)
- FG-metric flips: `13 of 43, 9 away from baselines` → **34 of 142, 16 away and
  12 toward**. The old figure is **not re-derivable** and must not be quoted.

**The lesson, in `dead-ends.md`:** pooling inflated one figure by 1.07× and
another badly, so *"it was pooled" is not itself proof a number is wrong* — it
has to be measured. The rebuttal now leads on the bounding box not being a
foreground region (76% of frame vs 15% true FG), not on "the honest metric
favours us", which the corpus data does not support.

## Still open, unchanged

- **Q6 DiffBIR — ASK before any wire.** No mechanism argument → stay deferred.
- **Q8 DC-VSR — blocked upstream**; now `_retired` in the queue.
- **0c** — `drift-straight` unexplained, ρ's CI includes zero; the cost model may
  not be called uniformly adequate. `tools/analyze_drift_straight_0c.py`, no GPU.
- `open-questions.md` / `dead-ends.md` are over the log's own 300-line ceiling.
- 16 pending fixed-QP `none`-restorer controls at comfortable QPs: legitimate,
  never run, nobody has decided whether they are wanted.

## Gotchas

- **The editable install points at the main checkout.** Tests run from a
  worktree import the *other* tree's `presley` unless `PYTHONPATH=<wt>/src`.
- **Never `import torch` before `presley`/`sqlite3`** (CXXABI_1.3.15). It fails
  at *runtime*, in-function, so a tool can look fine until the one path runs.
- The `Bash` tool's own `timeout` (ms, max 600000) is what applies; exceeding it
  backgrounds the command rather than killing it.
- Carried: filter fixed-QP with `codec=svtav1`; NAFNet / Real-HAT-GAN reject
  `fp32=False`; `fg_protect=True`; BG-PSNR is never the Goal-2 verdict; never
  `rm` `results/`/`dataset/`/`cache/`.

## Landmarks

| Path | Why |
|---|---|
| `docs/HOLE_CLOSURE_WAVE.md` | **start here** — bounds, calibration, H4 result |
| `~/.claude/plans/prepare-a-plan-to-quiet-crown.md` | the approved plan |
| `config/holes_wave_a.yaml` | the scoped-run-file pattern to copy |
| `sections/evaluation.tex` → `sec:insight` | the R2 answer |
| `research-log/dead-ends.md` | every superseded figure, with its lesson |
