# Handoff — three-goal restructure: Wave 0 + Wave 1 COMPLETE, builds begin (2026-07-30)

> ## UPDATE 2026-07-30 (later) — waves A1 and A3 ran; read this block first
>
> Both were run as parallel subagents in their own worktrees. **A1 was killed
> mid-task by a monthly spend limit**, not by a technical failure; the main
> session finished its decisive step by hand.
>
> **A3 — paper restructure: DONE.** Paper repo `@0738d16`, **2 commits
> unpushed** (origin is Overleaf — pushing is the user's call). Reframed around
> selection/reduction/restoration keeping every table and number; α/β is now a
> mis-specification ablation; new `tab:graded`. The retracted "+6 VMAF" claim is
> struck with `\del{\sout{}}` in abstract, intro and conclusions.
> **Caveat that matters:** F1's "93–99%" and W0.2's "6.2–8.2 dB" appear in NO
> reviewer-visible sentence, because `NEXT(sec:implementation)` requires CLAIM
> grade first and **F1 still has no `results/<hash>`** (ad-hoc, n=3, all-intra).
> Promoting F1 through `presley-run` under inter coding is now a real todo.
> Also unresolved: mean ΔBG-LPIPS is +0.0177 recomputed from `result.json` vs
> +0.0152 in the suite pass — no mean is quoted anywhere until that is
> reconciled.
>
> **A1 — S1b, stage 1: the gate PASSED** (superseded by stage 2 below, which
> then failed the ceiling test — read both). Branch
> `exp/s1b-damage-ceiling` in worktree `.claude/worktrees/agent-adc96e2b4b84b5357`,
> `@d2d6110`, 5 commits, **not merged to main**. All 7 pristine baselines and
> all 16 uniform-level probes are on disk, verified clean (empty
> `invariant_failures`, zero NaN across 42 metrics each). Mined 143 pairs →
> `results/block_damage_s1b.npz`.
>
> | quantity | value | kill threshold |
> |---|---|---|
> | between-level cost, mean Δ(k=3)−Δ(k=2) | +1.664 dB | must be ≥1 dB |
> | within-level spread, p90−p10 at k=3 | 12.36 dB | kill if <2.0 |
> | **R = spread / cost** | **7.43** | kill if <1.0 |
>
> Expected band for R was 3–10, so this passed cleanly — meaning only that the
> 8 Arm-B runs were worth spending, not that grading works. **Stage 2 then
> spent them and grading lost; S2 is now dead. See below.**
>
> **Three things A1 found that change how earlier results read:**
> 1. **S1's "naive graded" arm was barely graded on 7 of 8 videos** — share of
>    the footprint promoted past level 1 is 57.9% on bear but 8.2% on dogs-jump.
>    S1's negative is therefore weaker evidence against grading than it reads
>    as. **The paper text was corrected for this** (commit `0738d16`).
> 2. **S1's "more compression, more bits" is a MIXING cost, not a property of
>    downsampling.** At a fixed footprint and one uniform level, the more
>    aggressive level costs FEWER bytes on 7/8 videos (mean −2.3%). Paper
>    corrected for this too.
> 3. Mean damage at k=2 came in at 12.10 dB against a predicted 2–8 dB — the
>    prediction was simply wrong (the error runs in the conservative direction
>    for R). `dogs-jump` is non-monotonic (−0.08 dB, inside noise). Both
>    recorded in `docs/WAVE1_FALSIFIERS.md`, not glossed.
>
> **S1b IS NOW COMPLETE — and the ceiling test FAILED.** Arm B ran (8 oracle
> runs, `@0d29638`), LPIPS/DISTS backfilled (48/48, 0 failures). Against plain
> binary the oracle costs **+2.09% bits on 8/8** (p=0.0078) *and* is worse on
> **all four** background metrics on **8/8** (BG-LPIPS +0.0110, DISTS +0.0039,
> PSNR −0.48 dB, SSIM −0.0100; every one `sub_jnd_significant` in the worse
> direction). Foreground untouched. The pre-registered rule required a win on
> either axis; it got neither. **Grading is CLOSED. Do NOT build S2's
> structure-tensor proxy** — its ceiling is measured and sits below binary.
>
> Against Arm A at a matched histogram (bits held fixed at −0.44%, p=0.7266)
> the oracle leans better on 7/8 for every metric but not unanimously →
> `no_consistent_direction`, **no claim may be made** about oracle-vs-naive.
>
> **Scope limit that must not be dropped:** the oracle is *greedy* and
> *superblock-resolution*. It upper-bounds any predictor of the same quantity
> at that granularity — which licenses the transport conclusion — but it is not
> a proven constrained optimum. Never write "no assignment can win".
>
> **The mechanism, and it reframes S1:** at a uniform level, more downsampling
> costs *fewer* bits. Every graded arm's penalty comes from **mixing** levels,
> not from strength. Arm B mixes as much as Arm A and pays nearly the same
> penalty. So **S1's negative was never evidence about the removability score**
> — graded multi-level downscale is simply the wrong transport, whatever ranks
> the levels. The paper's mis-specification argument must rest on the *form* of
> the score plus `tab:ablation`, and was corrected to do so (`56017c6`).
>
> **Known defect hit while doing this:** `runner.py` ends by sweeping the whole
> results tree with `invariants.backfill(force=True)`, which writes a fixed
> `result.json.tmp` per dir. Two concurrent runners race and the loser dies with
> `FileNotFoundError` *after its own experiments succeeded* — a completed wave
> looks failed. Data was not corrupted here, but fix before the next parallel
> launch (unique tmp name; scope the sweep to the run's own hashes).
>
> **A2 (the O2 re-test) is RUNNING** as a subagent in its own worktree as of
> 2026-07-30 (later), on branch `exp/o2-operator-strength`. Result not yet in;
> scope unchanged from the original approval.
>
> **New gotchas, both of which cost a wasted 23-run pass:**
> - `--dataset-dir` and `--cache-dir` are relative and must be pointed at the
>   main checkout **exactly like `--results-dir`** when running from a worktree.
> - `presley-run` had ~37 *other* waves' unrun entries queued in
>   `experiments.yaml`; it must be driven from a filtered file or it runs them.
> - A wall of `[av1] Missing Sequence Header` / `Failed to get pixel format` in
>   the runner log is **benign ffmpeg noise**, not the silent-NaN AV1 trap.
>   Confirm by reading the metrics, never by reading the log.
> - **A subagent does not receive background-job completion notifications.** If
>   you delegate a long run, tell the agent to poll with bounded foreground
>   calls; ending its turn to "wait" just stops it. This cost a full stall here.

> Supersedes the 2026-07-29 version of this file (Wave 1 was half done then;
> all seven falsifiers are closed now). Unrelated workstream: `HANDOFF.md`
> at this same root covers the Q6/Q8 restorer queue — different work, don't
> conflate them.

## What triggered this handoff

**Not a failure, an outage, or a rate limit.** A natural boundary: Wave 1 is
finished, the first Goal-1 build (S1) has been landed *and* screened, and
everything is committed, pushed and green. The session was long and its
remaining items (S1b / O2 / paper restructure) share almost no state, so they
are better run as parallel workstreams from fresh sessions than sequentially
from a warm one.

**GPU:** `nvidia-smi` shows one process — `nasrinke/medsam3`, 5998 MiB.
**That is another user's, not ours.** Do not kill it, and do not read it as a
stalled job of ours. Nothing of ours is running or queued.

## One-paragraph summary

PRESLEY is a perceptual video-compression research pipeline: degrade
less-important regions server-side, restore them client-side with generative
models. The user is restructuring the paper around three goals — (1) pick
which blocks to degrade, (2) reduce size by degrading them, (3) restore well.
Approved plan:
[`~/.claude/plans/ok-that-is-a-sleepy-lark.md`](/home/itec/emanuele/.claude/plans/ok-that-is-a-sleepy-lark.md).
Its core claim is that the block-selection axis (`alpha`/`beta`) was
**mis-specified**: it models only how many *bits* a block costs and never how
well the block *comes back after restoration* — the objective only ever had a
numerator. Wave 0 built the measurement tooling, Wave 1 was seven cheap
falsifiers each designed to kill its own workstream, and the builds have now
started.

## Current state — verified while writing this, not recalled

### Code — `presley@main` @ `4de46eb`, clean, pushed, CI green

- `git status`: clean, `main...origin/main` in sync.
- Full suite green (`PYTHONPATH=$PWD/src <python> -m pytest tests/ -q
  --ignore=tests/invariants`), both CI gates pass locally
  (`tools/check_architecture.py`, `tools/sync_agent_rules.py --check`).
- **`feat/goals-wave0` is fully merged into main** (`git log feat/goals-wave0
  --not main` is empty). Its worktree at `.claude/worktrees/goal-rework` is
  clean and now redundant — safe to `git worktree remove` and delete the
  branch, but that is housekeeping, not required.

### Paper — `68e8b6bb11d0dd9e62a67aef@main` @ `97b9852`, clean, pushed to Overleaf

- 0 commits unpushed. `origin` IS Overleaf; pushing there is a user decision,
  and the user approved each push this session.

### Done and verified this session

| | Result |
|---|---|
| **Merge** | Wave 0 tooling merged to main. **It broke CI** — `test_shortfall_in_cluster_count_is_reported` asserted scipy's tie-breaking on an all-zero distance matrix (1.10 local → 1 cluster, CI's newer scipy → 4). Fixed by stubbing `pick_medoids`; added a version-independent contract test. |
| **F4** film-grain | **O4 CLOSED.** `film-grain=N` is a *denoise strength*: level 1 is a bitwise no-op, level 8 *costs* bits, level 50 saves 12.8% but is unanimous 8/8 worse at matched rate (`perceptual_loss`). Scoped to DAVIS-like clean source. |
| **F7** chroma | **O3 CLOSED on its oracle ceiling.** All chroma is only 5.07% of the bitstream; a *perfect* colorizer wins a sub-JND 0.0305 LPIPS. No model built. |
| **F2** alignment | **Signaling-overhead hypothesis REFUTED for 16→64.** At matched area, snapping saves bits on 1/8 videos. The bs8 penalty is a **cliff between 8 and 16**, not a gradient. Corollary: 16×16 selection is free vs 64×64. |
| **`suite.py` defect** | `assess_metric` chose between its two above-JND verdicts on `clears_jnd` alone, ignoring direction — so a perceptible **degradation** came back as `perceptual_win` with wording "may be worded as an improvement". Added `perceptual_loss`. **No landed verdict changed** (F4 was the first suite to reach that branch in the worse direction). |
| **Paper fold** | F2 + F3 folded in (`NOTE`/`NEXT` markers; softened an unconfirmed causal clause in the block-size prose; cross-updated `reviewers_comments.md`). F4–F7 deliberately stayed in the research log — no `HOLE()` exists for them in the current text. |
| **S1** | Graded multi-level downscale landed (`downsample_levels`, byte-identical at the default). **First time the 2^k pyramid path in `restoration.py:624` ever ran** — every strength map in project history was binary. |
| **S1 experiment** | **Clean NEGATIVE.** Naive score-based grading costs bits on 7/8 videos (mean +2.55%) *and* BG quality on 8/8 (PSNR −0.91 dB clears JND; LPIPS +0.0152 `sub_jnd_significant`; DISTS agrees). Alarm investigated before reporting — selection verified identical from `strength_maps.npz`. Real effect, not a bug. |

**Wave 1 scoreboard — all seven closed:** F1 (EVCA captures 93–99%, direction
closed), F2, F3 (`tune` confound real but sub-JND), F4, F5 (O2 mechanism
refuted), F6 (gate validated, parked), F7. **Nothing in the wave produced a
win**, which is what a well-designed falsifier wave looks like.

## What's running right now

**Nothing.** No background jobs, no GPU processes of ours, nothing queued.
Verify with `nvidia-smi` and `pgrep -af presley.runner` (expect only the
other user's medsam3 process).

## Open questions / decisions

1. **Does the graded direction survive at all?** S1b (scoped, below) answers
   it. If the *oracle* damage-aware assignment cannot beat plain binary, close
   the graded direction entirely and do **not** build S2's structure-tensor
   proxy — its ceiling would already be known worthless.
2. **Is the paper restructure ready to start now, or should it wait for S1b?**
   Argument for now: F1 + W0.2 + S1 already form the complete "the axis was
   mis-specified" argument the restructure needs (numerator solved, denominator
   varies 6.2–8.2 dB within a run, naive grading on the numerator fails).
   Argument for waiting: S1b might add a fourth, sharper data point.
   **Recommend starting now** — the argument does not depend on S1b's outcome.
3. **Clean up the stale worktrees?** `git worktree list` shows 11, several
   from long-finished sessions. `goal-rework` in particular is clean and fully
   merged. Housekeeping only; ask before deleting anything (host rule: read a
   branch before deleting it).

## Immediate next steps, in order

**Recommended split into parallel waves** (host rule — these share almost no
state, so run them in separate worktrees rather than sequentially):

### Wave A1 — S1b damage-aware ceiling test (fully scoped, ready to run)

Full design, bounds and decision rule: **`docs/WAVE1_FALSIFIERS.md`, section
"S1b"**. Read that before touching anything. Two blockers found while scoping
that will silently waste a run if missed:

1. **Add 7 pristine baselines first.** `tools/mine_block_damage.py` joins each
   restored run to a matched pristine baseline (same video/resolution/codec/QP).
   At 640×360 svtav1 QP43 **only `bear` has one** (`a07560c409dc38ce`).
   `motorbike, drift-straight, drift-turn, color-run, dancing, dogs-jump,
   bike-packing` have none — without them the miner yields nothing, silently.
   These are `component: baselines`, plain encodes, no restoration — cheap.
2. **LPIPS/DISTS need a backfill pass.** Runs come back PSNR/SSIM-only.
   Use `presley.evaluation.backfill.{backfill_lpips,backfill_dists}`.
   Note the CLI's `--only` takes exactly one `--backfill-*` flag at a time,
   so a two-metric backfill needs two calls or a small driver script.

Then: 16 uniform-level probe runs (levels 2 and 3 × 8 videos; level 1 is
already on disk as the S1 binary arm) → mine per-SB `delta_psnr` → **check the
within-level damage spread first** (if it's small there is nothing to exploit
and the direction dies with zero further runs) → only then the 8 Arm-B
confirmation runs. ~31 runs, ~1–1.5 h GPU.

### Wave A2 — the O2 re-test (approved long ago, never done)

The user approved this at the start of the session and it was never run. F5
refuted O2's stated mechanism but swept **QP only, pre-restoration**, leaving a
real 0.053 LPIPS advantage that DISTS contradicts in sign. The approved test is
to sweep the **operator strength** and measure **post-restoration**, because
the Goal-2 family is defined as (operator, prior) pairs and a pre-restoration
measurement cannot settle it. Same n=8 probe suite and bounds discipline.

### Wave A3 — the actual paper restructure (highest value, untouched)

The plan's "Paper restructuring" section: reframe around Goal 1/2/3 **keeping
every existing table and result**; α/β moves from "robustness" to an ablation
showing the axis was mis-specified, which *motivates* the new work instead of
reporting a null. Fold via `/update-paper`. This is the original point of the
whole plan and has not been started — this session only added `NOTE`/`NEXT`
markers for F2/F3.

### Wave B — conditional on A1

S2 (structure-tensor coherence as a cheap damage proxy) **only if** S1b's
ceiling passes. If it fails, close the graded direction and redirect to Goal 3
/ codec-conditioned restoration.

## Things not to redo

- **F6 is PARKED by explicit user decision**, pending a codec-conditioned
  restorer (converges with the Q6 MoE-DiffIR argument). It is a decision, not
  outstanding work — don't reopen it.
- **`src/presley/bitcost.py` in the plan's Files table is OBSOLETE.** F1
  explicitly concluded "do not build `bitcost.py`" — EVCA already captures
  93–99% of the achievable bit saving. The plan file predates that result.
- **Don't re-derive the `perceptual_loss` fix** — it's landed and tested.
- **Don't quote `p=0.031` for 5/5** — it's the one-tailed value; correct is
  0.0625, which is the floor at n=5. Hard rule 2b names this explicitly.

## Landmarks

| Path | Why |
|---|---|
| `~/.claude/plans/ok-that-is-a-sleepy-lark.md` | The approved three-goal plan |
| `docs/WAVE1_FALSIFIERS.md` | Every falsifier + S1/S1b: bounds, method, verdict, limitations. **The single most useful file here.** |
| `68e8b6bb11d0dd9e62a67aef/RESEARCH_LOG.md` | Index only — open **one** file under `research-log/`, not all (~17k tokens if you read everything) |
| `research-log/hard-rules.md` rule 2b | Wording rules for any comparison; n≥6 to be significant at all, n≥8 for restorer comparisons |
| `research-log/dead-ends.md` | Read before re-attempting anything — F2/F4/F7/S1-naive all live here |
| `src/presley/suite.py` | `assess_metric` returns the verdict *and* the mandated wording — quote it, don't paraphrase |
| `tools/{index_results,mine_block_damage,select_probe_videos}.py` | Wave 0 output; the miner is S1b's engine |

**Probe suite (pre-registered, use it):** `tools/select_probe_videos.py -k 8` →
motorbike, drift-straight, drift-turn, color-run, dancing, dogs-jump,
bike-packing, bear. n=8 is chosen so the exact two-tailed sign test can reach
p=0.0078 and survive Holm correction; the k=4 set floors at 0.125 and can
never be significant.

## Gotchas that already cost time

- **The conda env's editable install resolves `presley` to the MAIN checkout,
  not your worktree.** Any `src/` change needs `PYTHONPATH=$PWD/src` or the
  tests silently exercise the wrong code. Python:
  `/home/itec/emanuele/.conda/envs/presley/bin/python`.
- **`results/` and `cache/` do not exist in worktrees** (gitignored, per
  checkout). Tools take an explicit `--results-dir` pointing at
  `/home/itec/emanuele/presley/results`.
- **A grain/degradation arm can decode *smoother* than its control even when
  the effect is correctly applied** — the intuitive A/B gives the wrong answer.
  Only a same-file toggle settles it (F4 used dav1d `--filmgrain 0/1`;
  `av1tools` conda env has dav1d 1.5.3).
- **Rate-matching must pick its search direction from the data.** A QP search
  that only scans one way silently leaves one arm spending more bits —
  it bit this session in F4 before being caught.
- **Comparing decoded frames against PNG references costs ~29 dB to a
  colorspace/upsampling mismatch.** Put every arm *and* the reference through
  one identical conversion (F7).
- **`strength_maps.npz` is bit-plane packed** — read it with
  `sidechannel.load_level_masks`, not `np.load`.
- **OpenCV cannot decode AV1 here** — `cv2.VideoCapture` returns an empty frame
  list and metrics come out `NaN` with no error. Decode with `ffmpeg`.
- **Single-frame ffmpeg encodes cost ~0.55 s each**, dominated by process
  startup; a large sweep exceeds a 10-minute foreground timeout. Run detached.
- **Don't launch a long job with `nohup ... &` *and* `run_in_background`** —
  the wrapper exits immediately and the harness reports a completion that
  hasn't happened. Use `run_in_background` alone.

**Bound before believing.** Write plausible best/worst cases for each
measurement *before* reading its number, and commit them. This session's
bounds caught three things that would otherwise have been reported as
findings: F4's LPIPS breach, F7's <2% chroma alarm, and S1's "more bits from
more compression". Two of the three turned out to be real effects and one
prediction was simply wrong — the point is that each was *investigated* before
being written down as a result.
