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

Every `presley-run` invocation (including `--dry-run`) refreshes a `# hash:
<id>` comment above each entry in `experiments.yaml` so you can map an entry to
its `results/<id>/` dir without guessing; `presley-run experiments.yaml
--annotate-only` just refreshes those comments and exits. The hash is computed
excluding any `hash`/`_`-prefixed keys, so the annotation never perturbs it.

## Evaluation methodology — every experiment has a comparison target

PRESLEY has **two co-equal goals**. Every experiment tests one of them, and a
result is only complete when it says something about both:

- **Goal 1 — bit relocation.** Degradation moves encoding bits **BG→FG**, so at
  the same bitrate FG is *better*, respecting the chain
  `baseline < roi < elvis < presley_ai` (elvis and presley_ai may legitimately
  **tie** — see the FG-flatness finding). Lower BG quality is an *accepted cost*.
  **Metric:** FG-PSNR/FG-LPIPS at matched *actual* bitrate; BD-rate for
  paper-grade claims. Expected signature: FG ↑, BG ↓.
- **Goal 2 — generative restoration.** The client-side model restores the BG as
  close as possible to the **original**, without hurting FG; ideally exceeding
  original BG and/or FG. **Metric (perceptual primary): BG-LPIPS / BG-DISTS of
  the restored output vs the ORIGINAL**, compared against the pristine
  baseline's BG at matched bitrate. **BG-PSNR is reported alongside but is never
  the verdict** — `mean_fill` scores the *highest* BG-PSNR while being
  perceptually the *worst* (flat DC blocks are mathematically "closer" than
  hallucinated detail), so a PSNR-primary Goal 2 rewards a fill for **not**
  hallucinating, i.e. punishes the generative model for doing its job. The
  restoration *gain* (`metrics.background` − `metrics.transmitted.background`)
  is the mechanism; the **headline is restored-vs-original**, not
  restored-vs-degraded.

Goal 1 is not evidence for Goal 2 or vice versa. A method can free bits and
still fail to restore (that is the current standing — see the reports).

### ⛔ Hard rule: degradation experiments MUST use fixed-QP/CRF

Under **VBR the encoder spends the bitrate target regardless of source
complexity**, so degradation *cannot* free bits — it only makes the content
harder to code at that target, and the holes steal bits *from* FG, inverting
Goal 1. This is not a hypothesis: **25/25 matched VBR pairs, across every
degradation method ever run (freeze, downsample, blur, shrink), encode to MORE
bits than the pristine baseline. Zero counterexamples.** Under fixed QP the same
methods free bits (elvis_blackout −8.6% avg, elvis_freeze −9.7%, mean_fill
−6.8%).

**A VBR degradation curve is not evidence about the method — do not commission
one, and do not accept a spec that asks for one** (a 2026-07-16 TOP-PRIORITY
spec did exactly this and burned hours of GPU time re-measuring VBR laundering).
This is the same mechanism that already bit the codec-ROI work; see
TECHNICAL_REPORT_ROI_ENCODING's fixed-QP finding.

### Reporting rule: never dress up imperceptible deltas

Imperceptible deltas are not a result or a trend. **Run `presley-compare` to
decide whether a quality difference is real** — don't eyeball deltas. Its JND
table (`src/presley/compare.py`) is the single source of truth and is
deliberately not restated here. `presley-compare results/ --hash-a <h1>
--hash-b <h2>` for a pair; `presley-compare results/ --group-by
component,video,codec_params.qp --baseline-component baselines` for a
matched-QP sweep, which reports each group's quality verdict and its bitrate
winner. At matched QP this is the *whole* analysis: FG differences are small
by construction, so the question is never "who wins FG" but "who encodes
fewest bits at indistinguishable FG quality." State it the way it lands: *"at
FG quality that is indistinguishable, method X costs N% fewer bits than the
baseline, and BG-LPIPS is Y vs the baseline's Z."*

Never report only overall metrics — the `metrics.foreground`/`metrics.background`
split is the point (and for bridge runs `overall` is actively misleading, since
the collapsed BG dominates it). Analyze each component against its designated
target:

- **Codec ROI methods** (`kvazaar`, `x265_aq`, `svtav1`) vs the **same codec's
  baseline** at comparable bitrate. Expected signature: FG quality ↑, BG
  quality ↓. If it's absent, assume our usage is wrong before blaming the
  codec — "codec X doesn't implement ROI correctly" is a strong claim needing
  evidence beyond reasonable doubt (see TECHNICAL_REPORT for past false alarms).
- **presley_* ROI methods** (mask-driven degradation before encoding) vs the
  codec ROI methods: does direct block-level control buy more FG quality, and
  at what BG cost?
- **elvis** vs baselines, same analysis as ROI: did dropping removable blocks
  leave more bits for FG blocks at the same bitrate?
- **presley_ai** vs all of the above: FG quality must be best-in-class at
  matched bitrate, and the bitrate accounting must use
  `transmitted_size_bytes` (video + side-channel strength maps), not just the
  video file.

Exact bitrate matches are rare: compare at *similar actual* bitrates
(`actual_bitrate_bps`, not `target_bitrate`) for preliminary conclusions, and
use BD-rate curves (multiple target bitrates per method) for paper-grade
claims. If a link in the chain breaks, first search for regimes where it holds
(video subsets, bitrate ranges, codecs, parameters) before concluding the
method is worse — and only after that, re-examine the implementation.

**Fast iteration:** `presley-run … --fast-metrics` / `presley-evaluate
results/ --fast-metrics` compute only FG/BG/overall **PSNR+MSE** (SSIM,
LPIPS/DISTS/VMAF/FVMD and block-level maps are deferred to the full pass).
Fast-only results are tagged `metrics.fast_only` and get upgraded in place by a
later full `presley-evaluate results/`. The eval bottleneck is *not* the
metrics (~7% of time) — it's loading reference frames/masks from NFS, so
`evaluate_all` memoizes them across experiments in one pass (load once, not
per-experiment).

**FG-perceptual backfill:** the paper argues *foreground* perceptual quality,
but the base metrics are PSNR/SSIM. `presley-evaluate results/ --backfill-lpips`
appends region-restricted **LPIPS** (`foreground`/`background`/`overall`
`lpips_mean`) to every existing `result.json` *in place* — a metric-only pass
that re-reads the on-disk output videos and needs **no re-encoding** and no
rerun of experiments. It works on `fast_only` results too and is re-entrant
(skips ones that already have FG-LPIPS; use `--force` to recompute). LPIPS is
computed in spatial mode (per-pixel map averaged over the UFO mask), so FG/BG
are true region metrics, not bbox crops. LPIPS-alex is the fastest perceptual
metric (~0.76 s / 82 frames); DISTS/VMAF stay in the full pass.

**Starved-bitrate rule:** generative methods (elvis, presley_ai) only pay off
where the codec is bit-starved — hallucinating detail is only cheaper than
coding it when the codec can't afford the detail. Run their experiments at
bitrates low enough that the *baseline* is visibly quality-limited; a
comfortable-bitrate result understates them. The claim to pursue is "presley
wins in the starved regime," not "at every bitrate."

## Environment

Conda-managed (`environment.yaml` + `install_openmmlab.sh`), Python 3.10,
CUDA-pinned PyTorch. Do not `pip install` ad hoc into it — dependency
versions here are pinned tightly on purpose (see pinned versions in
`pyproject.toml`) because several forked third-party models
(ProPainter/E2FGVI/Real-ESRGAN/InstantIR) are version-sensitive.

**Host:** work runs on a shared remote Linux **GPU server, no root/sudo**.
Never reach for `apt` or other system installs — install any extra tooling
with conda (Miniconda is at `/usr/local/miniconda3`) into a *separate* env, not
the pinned `presley` env. Home is `/home/itec/emanuele`. `git push` already
works via a stored credential helper, so GitHub PRs/connectors/`gh` are not
needed for this solo workflow.

## No test suite

There is currently no automated test suite in this repo. Never report that
"tests pass" — verify by running a real (small/dry-run) experiment instead,
and **show the evidence**: the exact command and its output, not an assertion
that it worked. After a non-trivial change under `src/presley/`, run
`/code-review` before treating it as done — it reviews the working diff in a
fresh subagent that never saw your reasoning, which is the closest thing to a
test this repo has.

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

## Technical reports are the research source of truth

`68e8b6bb11d0dd9e62a67aef/TECHNICAL_REPORTS.md` (paper repo) is the dashboard
for the whole experimental effort: chain-status table, prioritized next
steps, and the catalog of topic reports (ROI encoding, ELVIS in-painting,
PRESLEY AI restoration, pipeline/evaluation infra), each with a standardized
findings log. **Read the relevant topic report's TL;DR before working on that
area** — it records what's already been tried, fixed, and disproven (e.g. why
ROI must run in fixed-QP/CRF mode, why addroi/x265-AQ can't do semantic ROI).
After any experiment run or diagnosis that produces new knowledge, fold it
back in with the `/update-reports` skill.

## Where to look for more

- Experiment workflow, filters, and reading back results → `/run-experiment` skill
- Summarizing/comparing results → `/results-report` skill
- Folding new findings into the technical reports → `/update-reports` skill
- Reviewer-response checklist workflow → see the paper repo's own CLAUDE.md
- Degradation/restoration algorithm details, past debugging dead-ends →
  topic reports catalogued in `68e8b6bb11d0dd9e62a67aef/TECHNICAL_REPORTS.md`
