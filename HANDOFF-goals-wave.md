# Handoff — three-goal restructure: Wave 0 done, Wave 1 half done (2026-07-29)

## What triggered this handoff

Not a failure or an outage. **The repo changed underneath this session.** While
this branch was being worked, `main` gained the suite-significance layer (from a
chip this session spawned), the `research-log/` split, and a noise-figure
supersession. This session's context predates all of it, and its worktree was
stale on every one of those files.

That divergence is now **resolved**: `origin/main` is merged into this branch,
the F3 finding has been re-verdicted under the new hard rule 2b, tests and both
CI gates pass. The handoff is because the session is long and the remaining work
is cleaner from a session that starts with the new tooling in context, not
because anything is broken or half-finished.

Nothing of ours is running. `nvidia-smi` shows one process
(`nasrinke/medsam3`, 5998 MiB) — **another user's, not ours**; do not kill it and
do not read it as a stalled job of ours.

## One-paragraph summary

PRESLEY is a perceptual video-compression research pipeline: degrade
less-important regions server-side, restore them client-side with generative
models. The user is restructuring the paper around three goals — (1) pick which
blocks to degrade, (2) reduce size by degrading them, (3) restore well. The
approved plan is
[`~/.claude/plans/ok-that-is-a-sleepy-lark.md`](/home/itec/emanuele/.claude/plans/ok-that-is-a-sleepy-lark.md).
Its core claim, now backed by two independent measurements, is that the
block-selection axis (`alpha`/`beta`) was **mis-specified**: it models only how
many *bits* a block costs and never how well the block *comes back after
restoration*, so the objective only ever had a numerator. Wave 0 built the
tooling to measure the missing denominator; Wave 1 is a set of cheap falsifiers,
each designed to kill its own workstream before anyone spends days on it.

## Current state (verified while writing this, not recalled)

### Code — branch `feat/goals-wave0`, worktree `.claude/worktrees/goal-rework`

- **Clean, pushed, `HEAD == origin/feat/goals-wave0` @ `776d0f2`, 11 commits
  ahead of `main`. UNMERGED.**
- Full suite green: `PYTHONPATH=$PWD/src <python> -m pytest tests/ -q --ignore=tests/invariants`
- Both CI gates pass locally: `tools/check_architecture.py` ("covers all 33
  modules") and `tools/sync_agent_rules.py --check` ("up to date").
- **Merging is the first thing to raise with the user.** Project rule: suggest,
  never self-merge.

### Paper — `68e8b6bb11d0dd9e62a67aef/`

- Clean, `main` @ `d044d44`, **0 commits unpushed to Overleaf**.
- This session's contribution (`2490cf3`) is on `main` and pushed, and survived
  the `research-log/` split intact — verified present in
  `research-log/dead-ends.md`, `research-log/standing-results.md` (2 refs), and
  the `NOTE(sec:implementation)` in `sections/presley.tex`.
- Nothing further is owed to the paper from this session's work. Everything
  landed is a **SCREEN**, not CLAIM-grade, and says so.

### Done and verified

| | Result |
|---|---|
| **W0.1** `tools/index_results.py` | SQLite index over 694 results, 152 cols. Cross-checked against hand-computed figures. |
| **W0.2** `tools/mine_block_damage.py` | 111 run/baseline pairs → 1.66M superblock observations. **Damage spread 6.2 dB (realesrgan) / 8.2 dB (propainter) *within a single run*.** |
| **W0.3** `tools/select_probe_videos.py` | Probe set = **camel, motorbike, drift-straight, dancing**. **bear never separates from camel at any k from 2–8** (0.92 sd vs 4.15 sd median) — bear is redundant, don't run both. |
| **F1** | **Direction CLOSED.** EVCA already captures **93.0–99.4%** of achievable bit saving (ρ +0.754…+0.958). **Do not build `bitcost.py`.** |
| **F3** | Confound real (SVT-AV1 defaults to `tune=1`=PSNR, verified bitwise). Now **`sub_jnd_significant` at n=7, p=0.0156**. |
| **F5** | **My stated mechanism REFUTED** — AC truncation costs +36% MORE bits at equal QP. At matched rate LPIPS and DISTS **disagree in sign at 100% of rate points**. No win claimed. |
| **F6** | Gate mechanism validated (FG never worse than transmitted, by construction) but gain is 10–50× below JND. Nothing worth gating with current restorers. |

### Not done

**F2, F4, F7** — see next steps.

## Corrections this session made to its own work (don't re-introduce)

- **`p=0.031` for 5/5 is wrong.** It is the *one-tailed* value; the direction was
  read off the data first. Correct value is **0.0625**, which is the floor at
  n=5 — it can never reach α=0.05. This error originated in this session's chip
  brief. It is now fixed everywhere; hard rule 2b names it explicitly.
- **F6's first result (+0.00 dB, 0.0% block wins) was a tautology, not a
  finding.** `_adaptive_block_pyramid_upscale` writes *only* degraded blocks, so
  restored output is bit-identical to transmitted on every protected FG pixel —
  the measurement compared a frame against itself. **Never read it as
  "restoration never helps FG."** Pre-stated bounds caught this.
- **`docs/HANDOFF_TO_PAPER_RESTRUCTURE.md` says the `0.031` figure is in
  `docs/WAVE1_FALSIFIERS.md`. It is not** — it was only ever in the chip brief.
  The correction itself is right; only its stated location is wrong.

## Open questions for the user

1. **Merge `feat/goals-wave0` into `main`?** Green, pushed, 11 commits, CI gates
   pass locally. Recommend yes; the tooling (index, damage miner, probe
   selector) is what everything downstream reads.
2. **Is O2 (AC truncation) worth another round?** F5 refuted the *mechanism* but
   left a real 0.053 LPIPS advantage at 100% of rate points, contradicted by
   DISTS. Options: (a) re-test sweeping operator **strength** rather than only
   QP, and **post-restoration** — the honest next test; (b) drop O2 and move to
   O3/O4; (c) leave it catalogued as a metric-disagreement case. Recommend (a),
   because the current test was pre-restoration and the Goal-2 family is defined
   as (operator, prior) pairs.
3. **Does F6 stay parked until a codec-conditioned restorer exists?** F6 and the
   Q6 mechanism argument converge on MoE-DiffIR. Recommend parking it.

## Immediate next steps, in order

1. **Read** `docs/HANDOFF_TO_PAPER_RESTRUCTURE.md` and
   `research-log/hard-rules.md` rule 2b before wording any comparison. Cheapest
   correct summary: *never call an n=2 result a tie; n≥6 to be significant at
   all, n≥8 for restorer comparisons.*
2. **Ask the user about merging** (open question 1). Do not self-merge.
3. **F4 (~1 h, cheapest remaining).** `--film-grain` on/off. SVT-AV1 v1.8.0 here
   exposes `--film-grain 1..50` and `--film-grain-denoise`. Encode the probe set
   with/without at fixed QP, rate-match, run through
   `presley-compare --suite`. **Size the suite to n≥6 before running** — at n=4
   nothing can be significant. Expect grain removal to save bits on textured
   clips and do nothing on flat ones.
4. **F7 (~0.5 d).** Chroma-first degradation. The only axis orthogonal to
   everything tried; every operator to date is luma-structure focused. Degrade
   BG chroma hard, keep luma; restore with a colorization prior.
5. **F2 (~1 d).** 64×64-snapped vs scattered 16×16 selection at matched degraded
   *area*. Tests the signaling-overhead hypothesis that the existing ablation
   already hints at (bs=8 *increases* bitrate vs 16–64).
6. **Then S1**, the highest value-per-effort item in the whole plan: emit a
   genuine multi-level downscale map. **Verified: every strength map in project
   history is binary 0/1 across all 13 (restorer, degradation) combinations**, so
   the `2^k` pyramid in `restoration.py:624` has never once run.

**Bound before believing.** Write plausible best/worst case for each falsifier
*before* reading its number. This session's F6 alarm is the worked example of
why — it caught a tautology that would otherwise have killed a workstream.

## Landmarks and gotchas

| Path | Why |
|---|---|
| `~/.claude/plans/ok-that-is-a-sleepy-lark.md` | The approved three-goal plan |
| `docs/WAVE1_FALSIFIERS.md` | Every falsifier: bounds, method, verdict, limitations |
| `docs/HANDOFF_TO_PAPER_RESTRUCTURE.md` | What the significance session changed |
| `docs/SIGNIFICANCE_AUDIT.md` + `research-log/hard-rules.md` 2b | Wording rules for any comparison |
| `src/presley/suite.py` | `assess_metric` returns the verdict *and* mandated wording — quote it, don't paraphrase |
| `tools/{index_results,mine_block_damage,select_probe_videos}.py` | Wave 0 output |

**Gotchas that already cost this session time:**

- **The conda env's editable install resolves `presley` to the MAIN checkout,
  not your worktree.** Any `src/` change needs `PYTHONPATH=$PWD/src` or the
  tests silently exercise the wrong code. Python:
  `/home/itec/emanuele/.conda/envs/presley/bin/python`.
- **OpenCV cannot decode AV1 here.** `cv2.VideoCapture` returns an empty frame
  list and metrics come out `NaN` with no error. Decode with `ffmpeg` to PNG.
- **`results/` and `cache/` do not exist in worktrees** (gitignored, per
  checkout). Tools take an explicit `--results-dir` pointing at
  `/home/itec/emanuele/presley/results`.
- **Single-frame ffmpeg encodes cost ~0.55 s each**, dominated by process
  startup — an F1-style sweep of 3 videos × 6 frames × 61 encodes exceeds a
  10-minute foreground timeout. Budget accordingly or run detached.
- `strength_maps.npz` is **bit-plane packed**; read it with
  `sidechannel.load_level_masks`, not `np.load`.
