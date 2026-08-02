# Handoff — H3 run, H1b and H2 chained and running unattended (2026-08-02)

Supersedes the earlier 2026-08-02 handoff. **A detached job chain is in flight
and will be for roughly 20 more hours.** Do not launch a second GPU job against
this checkout until it drains; check the driver log before you do anything.

Approved plan: `~/.claude/plans/prepare-a-plan-to-quiet-crown.md`.
Bounds for every set: **`docs/HOLE_CLOSURE_WAVE.md`** — read it first.

## What is running right now

```
/home/itec/emanuele/presley/logs/holes_chain_driver.log
```

That is the driver. It waits on the H3 `presley-run` PID, then runs **H1b**
(~13 h, 16 ProPainter), then **H2** (~7 h, 8 ProPainter), then
`presley-evaluate results/ --backfill-lpips`. Each set writes its own
`logs/holes_<set>_<stamp>.log`.

- **Check log mtimes and `results/<hash>/result.json`, never `pgrep`** — a
  `pgrep -f` pattern matches your own checking command. This has cost this
  project hours more than once.
- The runner **skips any hash that already has a `result.json`**, so an
  interrupted set resumes by being re-run against its own run-file. Nothing is
  lost if the chain dies; relaunch `config/holes_wave_h1b.yaml` (or `h2`).
- Scoped run-files exist precisely so you never need
  `--filter component=presley_ai`, which would drag in the 16 unrelated
  pending controls.

### The trap that will waste your time if you skip this

A fresh run's `result.json` has **`invariant_failures: ["metrics block is
missing or empty"]` and no metrics at all** until `presley-run`'s evaluation
pass finishes, which happens *after* the whole run loop. That is the documented
pre-evaluation sticky verdict, not a broken run. Both analysis scripts below
correctly refuse to quote such a run, so "incomplete or not citable" for a
whole set means *wait*, not *debug*. **Region LPIPS for `elvis` runs
additionally needs `--backfill-lpips`**, which the chain does at the very end.

## State

- Code `main` clean and pushed, **CI green** (run 30738506530).
- Paper pushed to Overleaf. Everything `\rev{}`-tracked.
- `experiments.yaml` 880 entries (800 + 80 appended as text; prior hashes
  byte-identical, verified by a pure-addition diff). 20 still `_retired`.
- Results 985+ and climbing as the chain runs.
- **`pdflatex` is not installed on this host** — the paper cannot be compiled
  here. Brace balance and tabular column counts were checked by hand instead.

## Done this session

| | outcome |
|---|---|
| **H3** | 32 runs complete, evaluation pass still finishing at write time — **run `tools/analyze_holes_h3.py` first thing**; no number has been read from it yet |
| **H1b / H2** | appended, run-files written and dry-run-clean, chained and queued |
| **H4 → `tab:budget-knee`** | **landed in the paper**, `HOLE(tab:priced-trade)` cleared by the edit that landed its data |
| **Dispersion sweep** | **clean** — all nine figures already within-run or explicitly scoped; recorded in `operational.md` with the grep that reproduces it |

### The H4 finding, since it sharpened under JND gating

The budget lever is **inert below sa 0.25** (bitrate flat within ±1.2%,
encoder noise at fixed QP) and the restoration gain there is **sub-JND**
(0.57× bear, 0.83× camel) — so the smallest budget both fails to free bits and
produces a repair nobody would see. Above 0.25 the budget buys bits (−31.6% /
−23.5% vs pristine) and the gain grows to 2.5–3.7× JND without cliffing.
`tab:priced-trade`'s fixed 0.25 sits exactly at that knee. Every JND multiple
came from `presley-compare`, not hand arithmetic.

The pre-registered monotonicity bound **fired on bear and was revised with a
stated reason** rather than dropped — that is the convention to follow if one
of yours fires.

## Your task, in order

1. **Read H3** — `python tools/analyze_holes_h3.py`. Bounds are encoded in the
   script and checked in its output. If a bound fires, investigate
   implementation / eval / data *before* reporting it as a finding.
2. **Fold H3 into `HOLE(tab:conditioned)`** — a `HOLE` may only be cleared by
   the edit that lands its data. Follow the `tab:budget-knee` commit as the
   worked example (GOAL + CLAIM with hashes + NOTEs for the caveats).
3. **H1b** — `python tools/analyze_holes_h1b.py` once it lands, then
   `HOLE(tab:av1)`. ⚠ `pigs` cannot be starved as hard as bear/camel; the
   script prints that caveat next to the numbers, keep it attached.
4. **H2** — `HOLE(tab:goal2)`. n=4 videos is still under the n≥6 hard rule 2b
   requires, so this **cannot** produce a "significant" verdict; it can only
   widen or narrow the descriptive claim, and the text must say so.

## Still open, unchanged

- **Q6 DiffBIR — ASK before any wire.** No mechanism argument → stay deferred.
- **Q8 DC-VSR — blocked upstream**; `_retired` in the queue.
- **0c** — `drift-straight` unexplained, ρ's CI includes zero.
  `tools/analyze_drift_straight_0c.py`, no GPU.
- `open-questions.md` / `dead-ends.md` are over the log's 300-line ceiling.
- 16 pending fixed-QP `none`-restorer controls: legitimate, never run, nobody
  has decided whether they are wanted.

## Gotchas

- **The editable install points at the main checkout.** Tests run from a
  worktree import the *other* tree's `presley` unless `PYTHONPATH=<wt>/src`.
- **Never `import torch` before `presley`/`sqlite3`** (CXXABI_1.3.15) — it
  fails at *runtime*, in-function.
- `nohup … &` inside a harness `Bash` call returns when the wrapper exits; that
  notification is not the job finishing.
- The `Bash` tool's own `timeout` (ms, max 600000) is what applies.
- Carried: filter fixed-QP with `codec=svtav1`; NAFNet / Real-HAT-GAN reject
  `fp32=False`; `fg_protect=True`; BG-PSNR is never the Goal-2 verdict; never
  `rm` `results/`/`dataset/`/`cache/`.

## Landmarks

| Path | Why |
|---|---|
| `logs/holes_chain_driver.log` | **check this first** — what is running |
| `docs/HOLE_CLOSURE_WAVE.md` | bounds, stage-1 calibration, H4 result |
| `tools/analyze_holes_h3.py`, `tools/analyze_holes_h1b.py` | the analyses, bounds encoded |
| `config/holes_wave_{h3,h1b,h2}.yaml` | scoped run-files; re-runnable, resume-safe |
| `sections/evaluation.tex` → `tab:budget-knee` | worked example of clearing a HOLE |
| `research-log/operational.md` | the dispersion sweep and how to redo it |
| `research-log/dead-ends.md` | every superseded figure, with its lesson |
