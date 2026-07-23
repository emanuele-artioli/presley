---
applyTo: '**'
---

<!-- GENERATED from CLAUDE.md by tools/sync_agent_rules.py — DO NOT EDIT.
     Edit CLAUDE.md and re-run the script; a pre-commit hook checks this. -->

# PRESLEY

Research pipeline for perceptual video compression: degrade less-important video
regions server-side (QP mapping, downsampling, or noise injection) and restore
them client-side with generative models (Real-ESRGAN, InstantIR, ProPainter,
E2FGVI). Extends prior work ELVIS (block removal + in-painting). Companion
paper lives in [68e8b6bb11d0dd9e62a67aef/](68e8b6bb11d0dd9e62a67aef/) — a
separate git repo (Overleaf sync, gitignored here) with its own CLAUDE.md and
its own conventions. Don't apply this file's rules there.

**This file is the only rule file to edit by hand.** `AGENTS.md`,
`.agents/rules/presley.md` (Antigravity) and
`.github/instructions/presley.instructions.md` (Copilot) are *generated* from
it by `tools/sync_agent_rules.py`, which also inlines the host-wide
`~/.claude/CLAUDE.md` that only Claude loads automatically. Edit CLAUDE.md,
then re-run the script — CI fails if the generated files are stale.

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

Because they take hours, run them either through the `experiment-runner`
subagent or as a background `Bash` (`run_in_background: true`), which
notifies on exit. Never poll for completion with a hand-written
`until ! pgrep …` loop — see the global CLAUDE.md's waiting rule for why
that loop can never terminate.

`src/presley/cli.py`, `src/presley/pipeline_legacy.py` and the 1602-line root
`utils.py` were deleted on 2026-07-22 — all three were pre-`components/`
refactor dead code (the first two imported the long-gone `presley.config`;
nothing imported the third). The live helper module is `src/presley/utils.py`.
If an old doc or a stale worktree still refers to them, that reference is the
thing to fix.

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
the RESEARCH_LOG's fixed-QP hard rule.

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
  evidence beyond reasonable doubt (see RESEARCH_LOG.md for past false alarms).
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
session it is diagnosed** — the RESEARCH_LOG dead-end entry and the
regression test are written together. Deleting a test requires saying why its
failure mode is now impossible.

Research code, so keep tests honest and thin: cover envisioned behavior and
plausible misuse of code we own, not unreachable branches, third-party
library behavior, or errors a caller cannot produce. **A test that exists
only to raise the coverage number is a defect.** The `/test-design` skill
proposes a test list for review before writing any of it.

## Concurrent sessions & which branch to work on

This repo gets worked on by several agents at once (Claude sessions,
Antigravity, Codex, Copilot). A branch only isolates a session if it also has
its own working directory — two agents sharing this checkout share one HEAD,
so "make a branch" alone does not prevent one session clobbering another's
uncommitted edit through an ordinary read-modify-write race.

- **Substantive code change** (a refactor, a new component, anything spanning
  more than a file or two): take a worktree *and* a branch —
  `git worktree add ../wt-presley/<slug> -b <type>/<slug>` with a `refactor/`,
  `feat/`, `fix/` or `exp/` prefix. Push after every logical commit so the
  work survives an SSH drop. When it is green, **suggest** merging and let the
  human do it — never self-merge.
- **Small fix, doc edit, or a session whose real work is running
  experiments:** stay in this checkout and commit as soon as fast checks pass.
  Worktrees carry no `results/`, `cache/` or `weights/`, so run-only sessions
  belong here anyway.
- Commit a fix as soon as it passes — never leave it uncommitted while a long
  GPU run is in flight.
- Deleting a branch requires reading it first (`git log main..<branch>`); if it
  is not empty, `git tag archive/<branch>` and push the tag before deleting. A
  worktree with uncommitted changes gets those changes committed onto its own
  branch before removal — never `--force` them away.

## Long jobs must checkpoint at least hourly

SSH to this host drops a couple of times a day, and restoration runs take
hours. Any job expected to exceed an hour checkpoints at least every 60
minutes of wall clock, independent of its epoch/step cadence, and its resume
path is verified *before* it is relied on. Long scripts append a progress line
(step, metric, timestamp) to their log at least every 10 minutes, so a silent
hang is visible in minutes and `Monitor` always has something fresh to match
on. Launch detached — never attached to a shell an SSH drop takes with it.

## `results/` is gitignored — deletion is unrecoverable

`results/` (and `cache/`) are in `.gitignore`, so they are **not** in git
history — a wholesale `rm` cannot be undone with git. The expensive
preprocessing (`cache/`: reference frames, EVCA scores, UFO masks) is
regenerable but slow; the GPU restoration outputs in `results/` cost hours to
recompute. A PreToolUse hook (`~/.agent-rules/scripts/guard-rm.py`,
centralized and shared with other repos on this host, configured in this
repo's `.claude/settings.json`) blocks `rm` against the whole
`results/`/`dataset/`/`cache/` tree for this reason. Never test a destructive
command against these real directories.

## This tooling is meant to evolve

The `.claude/` directory (this file, `skills/`, `agents/`, `hooks/`,
`settings.json`) is part of the working setup, not frozen. If during work we
find a convention Claude gets wrong twice, a skill that would help, a hook
worth adding, or a rule that's stale — add/update/remove it in `.claude/`
right then. Note: edits to `settings.json`/hooks only take effect on the next
session (open `/hooks` or restart to reload); skills and CLAUDE.md load fresh
each session too, so prefer those for anything you want to rely on immediately.

## The paper is the primary living document

The manuscript (`68e8b6bb11d0dd9e62a67aef/main.tex` + `sections/*.tex`)
carries the research plan as machine-readable comment markers
(`STATUS/GOAL/HOLE/NOTE/NEXT/CLAIM(anchor):` — spec in the paper repo's
CLAUDE.md). **Before planning any experiment, grep the paper's `HOLE()`
markers** — run only the experiments the paper needs. After a session
produces committed, tested results, fold them in with the `/update-paper`
skill. `68e8b6bb11d0dd9e62a67aef/RESEARCH_LOG.md` is the secondary store:
hard methodology rules (fixed-QP-only, JND gating, FG-metric citability),
the dead-end registry (what's been tried and disproven — read it before
re-attempting anything), and the queue of results not yet in the text. The
old technical reports were consolidated into it on 2026-07-18 (full history
via `git log --follow` in the paper repo).

## Where to look for more

- Experiment workflow, filters, and reading back results → `/run-experiment` skill
- Summarizing/comparing results → `/results-report` skill
- Folding new findings into the paper → `/update-paper` skill
- Choosing and writing tests for a component → `/test-design` skill
- Reviewer-response checklist workflow → see the paper repo's own CLAUDE.md
- Algorithm details, hard rules, past dead-ends →
  `68e8b6bb11d0dd9e62a67aef/RESEARCH_LOG.md`

---

# Host-wide rules

These apply to every project on this host. Claude Code loads them
automatically; they are inlined here for agents that do not.

## Global environment notes

These apply across all projects/sessions on this host, not just one repo's
CLAUDE.md. **This file is the register of things that have gone wrong more
than once** — if a mistake happens twice, it belongs here, phrased as the rule
that prevents it rather than the story of the failure.

## Shared agent rules — single source of truth

Imported by reference (`@` syntax) from each coding agent's own rules file —
currently `~/.claude/CLAUDE.md` and `~/.gemini/GEMINI.md`. Edit **only this
file** for anything that should apply to every agent on this host. Put
agent-specific mechanics (tool names, invocation syntax, that agent's own
conventions) in the importing file instead, not here — this file stays
tool-agnostic so every importer can use it as-is.

### The host

Shared remote Linux **GPU server, no root/sudo/apt**, headless. Home is
`/home/itec/emanuele`. Install extra tooling with conda (Miniconda at
`/usr/local/miniconda3`) into a *separate* env — never into a project's
pinned env, because several forked third-party models are version-sensitive
and a stray `pip install` silently breaks them. Being headless, save media
and plots to disk; `cv2.imshow()`/`plt.show()` never works here.

### Python dependency management

Manage Python packages through `pyproject.toml`, not ad-hoc `pip install` in
the terminal. `environment.yaml` is reserved for bootstrapping heavy
CUDA/GPU binaries only (drivers, PyTorch wheels, compiled packages) — never
fall back to a `requirements.txt` file.

### GitHub CLI (gh)

`gh` is installed at `~/emanuele/bin/gh` (on `PATH` in every shell on this
host) and authenticated as `emanuele-artioli` via `gh auth login`
(credentials in `~/.config/gh/hosts.yml`, not tied to any one project).
Available in every project on this host — install/auth doesn't need
repeating.

**Use it proactively after every push to a repo with GitHub Actions (or
any CI):** don't assume a push landed cleanly or guess at failures from
job/step names alone.

- `gh run list --branch <branch> --limit 3` — find the run a push triggered
- `gh run view <run-id> --json status,conclusion -q '.status'` (poll) or
  `gh run watch <run-id>` — wait for it to finish (`gh run watch` can
  itself flake with a transient "Bad credentials" on the annotations
  call; a `gh run view <run-id>` after that still shows the real job
  status, so don't treat a `run watch` crash as the run having failed)
- `gh run view <run-id> --log-failed` — **the real fix for CI debugging.**
  The unauthenticated GitHub REST API only exposes job/step names and
  conclusions, never log content (log downloads 403 "Must have admin
  rights" even on public repos without an authenticated token) — that
  API alone means guessing at root causes from symptoms. Authenticated
  `gh` gives the exact failing line immediately.

Also usable the same way for `gh pr view`, `gh issue view`, `gh pr create`,
etc. wherever a GitHub-authenticated operation is needed — this isn't
CI-specific.

### Git — never destroy work you have not read

These repos get worked on by several agents at once (Claude sessions,
Antigravity, Codex, Copilot), and unmerged work has genuinely been lost here
before: a complete HNeRV baseline once sat in a forgotten worktree.

- **Read a branch before deleting it.** `git log main..<branch>` and
  `git diff main...<branch> --stat`. If it is not empty,
  `git tag archive/<branch> <branch>` and push the tag *before* deleting.
  Tags are free and make a triage mistake recoverable.
- **A worktree with uncommitted changes never gets `--force`d away.**
  Commit the changes onto that worktree's own branch, tag it, then remove
  the worktree. `git worktree remove` refusing is a warning, not an obstacle
  to route around.
- **"Superseded" needs proof, not a guess.** Compare with `git patch-id`, or
  diff the files against `main` — a branch whose commit message matches one
  on main may still hold changes main never got.
- **A branch alone does not isolate a session** — two agents in one checkout
  share one HEAD. Isolation needs a worktree *and* a branch.

### Research code — tests are a failsafe, not a formality

Cover envisioned behavior and plausible misuse of code we own. Skip tests for
unreachable branches, third-party library behavior, and errors a caller
cannot produce; this is research code and boilerplate slows the iteration
that actually matters. **A test that exists only to raise a coverage number
is a defect** — it makes the gate lie about what is verified. If deleting
padding drops the gate, lower the gate to the honest number and ratchet it
back up as real tests land.

The tests that pay for themselves here are the ones that check *the paper's
claim*, not just that the code runs: an experiment whose result violates the
thing the paper asserts should fail loudly and be marked uncitable, rather
than being caught later by a careful human reading a table.

### Long jobs must checkpoint at least hourly

SSH to this host drops a couple of times a day. Any job expected to run over
an hour checkpoints at least every 60 minutes of wall clock — independent of
its epoch/step cadence — and its resume path is verified *before* it is
relied on. Long scripts also append a progress line to their log at least
every 10 minutes, so a silent hang is visible in minutes rather than hours.
Launch detached; never attached to a shell an SSH drop takes with it.

### Plan mode: split complex plans into parallel-agent waves

When a plan has multiple pieces of work that don't share state, don't execute
it as one linear sequence. Split it into workstreams and hand each to a
subagent working in its own git worktree (a shared checkout with only a
different branch is not isolation — two sessions in one worktree share a
single HEAD). Group workstreams into **waves** ordered by dependency: a wave
starts only once every workstream it depends on has reported results back,
and every workstream within a wave launches together, not one at a time.

**Why:** validated on a multi-part refactor — this surfaced cross-workstream
issues at each wave boundary instead of at the end, and kept parallel agents
from clobbering each other's changes.

**How to apply:** worth it for genuinely multi-part, multi-file tasks where
pieces are largely independent. Skip it for small or sequential tasks — one
file, one clear order of steps — where waves are pure coordination overhead.

### Waiting for long-running commands — never hand-roll a waiter

⛔ **Never write `until ! pgrep -f <pattern>; do sleep N; done` (or any
self-written poll loop) to wait for a job.** The harness runs the loop via
`bash -c "<the whole command string>"`, and that string *contains* the
pattern — so `pgrep -f` matches the watcher's own process and the condition
can never become true. The job finishes, the watcher spins until timeout, and
the completion goes unnoticed. This has already burned >1h of wall clock.
Escaping tricks (`[p]attern`, `pgrep -P`) technically work but are still the
wrong answer: the harness already reports completion, so there is nothing to
poll for.

Pick by duration, not by habit:

- **Finishes in < 10 min** → foreground `Bash` with an explicit `timeout`
  (ms, max 600000). Output arrives in one piece and the harness kills it at
  the deadline, so it cannot hang forever.
- **Longer than that** (GPU restoration, full evaluation passes, big
  backfills) → `Bash` with `run_in_background: true`. It detaches, survives
  across turns, and **re-invokes Claude on exit** with the path to its
  output file. Read that file; do not poll for it.
- **Need progress while it runs** → `Monitor`, with a filter that matches
  failure signatures too (`Traceback|Error|FAILED|Killed|OOM`), not just the
  success marker — a success-only filter stays silent through a crash, and
  silence is indistinguishable from "still running."

`conda run -n <env> …` is not a solution to this. It is still a foreground
command subject to the same 10-minute cap, and without
`--no-capture-output` it buffers all output until exit — so on a long job it
shows nothing and then gets killed. Use it for env activation if convenient,
never as a completion-waiting strategy.

Note: `Monitor`'s progress-matching depends on the logging cadence described
in the shared "Long jobs must checkpoint" rule above — a job that goes quiet
for more than ~10 minutes gives Monitor nothing fresh to match, which looks
identical to a hang.

Same trap, different tool: **`ScheduleWakeup` is not a wait-for-completion
mechanism.** It exists solely to self-pace `/loop` dynamic-mode iterations.
A background agent or background `Bash` job already triggers a notification
the moment it finishes — there is nothing to poll for. Don't call
`ScheduleWakeup` "just to wait" for one; it also fails outright when used
this way (it requires a `prompt` unless `stop: true`), so the mistake
surfaces immediately rather than silently wasting a turn — still worth not
repeating.
