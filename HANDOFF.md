# Handoff — breadth chain confirmed, F1 closed, NAFNet defect found (2026-07-31)

Supersedes the 2026-07-29 Q7/Q9/Q10 handoff. **Its still-open items (Q6, Q8,
standing HOLEs) are carried forward below — they were not dropped.**

## State (verified at write time)

- Code `main` @ **`eeedf98`**, clean, pushed, CI green.
- Paper @ **`9f8848c`**, pushed to Overleaf.
- **Nothing running.** No PRESLEY-owned jobs. (`nvidia-smi` may show other
  users' — check PIDs before assuming a free GPU.)
- Worktrees: only this session's
  (`.claude/worktrees/presley-breadth-database-aa9a5f`), branch fully merged —
  safe to remove.
- Results DB: **899 runs, 893 citable.** The 6 failures are pre-existing and
  correct (VBR degradation caught by hard rule 1; 2 restoration-hurt-background).

## What landed this session

| | outcome |
|---|---|
| **Rate-matched breadth** | BD-rate BG-LPIPS **−51.4%**, 9/9 clips → chain `presley_ai > elvis` **confirmed at matched rate** outside DAVIS. `tab:breadth-ratematched` |
| **F1 oracle-bits** | ρ +0.669, capture 0.833 **against a 0.402 random null**, n=8 → `NEXT(sec:implementation)` cleared for **claim (a) only**. `CLAIM(f1-oracle-bits)` |
| **bike-packing NAFNet** | Reclassified: a **numerical defect** (output diverges, ~8.6% of pixels clipped to 0/255), not a content effect |
| **B3 fallout** | Fixed a runtime-only `torch`-before-`sqlite3` bug I introduced; `tests/test_import_order.py` pins it |
| **Worktrees** | 15 → 1 |

The two breadth framings are consistent: at fixed QP PRESLEY spends +23%/+15%
more bits and returns disproportionately more background quality, so **per bit**
it is far ahead. The premium was a fixed-QP artefact, not a cost of the method.

---

## Open items, in the order I would take them

### 1. Two decisions that need the human

**(a) The saturation invariant.** NAFNet clipped 8.6% of output pixels to 0/255
with `invariant_failures` **empty** — nothing checks a restorer's output for
saturation. A check like *"restoration clipped more than N% of pixels to 0/255
that its input did not"* catches it on the first run. **Deliberately not
implemented:** it would mark existing runs uncitable, changing what the paper may
cite. Decide the threshold and whether to retro-apply. Analysis:
`research-log/bugs.md`, top entry.

**(b) `baselines` at QP 42/47.** The pre-registered optional second wave, gated
on the first 36 landing clean — they did. 18 runs, ~35 min. **Not needed** for
the chain result. Only for keeping the baseline/ELVIS/PRESLEY triple intact at
the new rungs. Entries are **not** yet in `experiments.yaml`.

### 2. Claim (b) — the last thing blocking `NEXT(sec:implementation)`

The 6.2/8.2 dB **denominator** spread is still computed by
`tools/mine_block_damage.py` **outside the runner**, so it has no
`results/<hash>` and may not appear in any reviewer-visible sentence. F1 fixed
claim (a) only.

Same shape as F1's fix: a `probe_*` component through `presley-run`. **Read
`src/presley/components/probe_oracle_bits.py` first** — it solves the non-obvious
blocker: `invariants._check_metrics_present` unconditionally demands FG/BG/overall
PSNR, and a measurement probe has no reconstructed video. The answer is to publish
a real encode as `output_video`, **not** to exempt the component.

### 3. Open question 0c — why EVCA is near-chance on `drift-straight`

ρ +0.082, capture 0.510 against a 0.436 null, vs ρ 0.61–0.90 on the other seven.
**The obvious explanation is already refuted:** its marginal-bit distribution is
not flat (CV 0.93, mid-pack), and ρ vs CV across the eight correlates at 0.031.
Unexplained. Resolve before anyone claims the cost model is uniformly adequate.
Data is in the 8 probe runs — no new GPU time to start.

### 4. Carried forward from 2026-07-29 — still open

- **Q6 DiffBIR — ASK before any wire.** Needs a written mechanism argument for
  why DiffBIR's prior would beat NAFNet/unsharp on spatial-Gaussian+CRF blur.
  NAFNet already ties unsharp with no Goal-2 gain. **No argument → leave
  deferred; do not commission.**
- **Q8 DC-VSR — blocked upstream.** Re-probed 2026-07-29: HF `Janghyeok/dc-vsr`
  is weights-only (no `model_index.json`, VAE, scheduler or pipeline code); no
  official code release. The stub's `RuntimeError` is correct. Only wire if
  upstream publishes real inference. Evidence in `docs/EXPERIMENTS_QUEUED.md`.
- **Standing paper HOLEs:** `HOLE(tab:av1)` n>2 videos,
  `HOLE(tab:goal2)`/`HOLE(tab:conditioned)` n>2, `HOLE(tab:priced-trade)`
  shrink_amount arm. Fixed QP only.

### 5. Referee 2 "lack of technical insights" — still `[ ] To Do`

Untouched here and much larger than anything above: a narrative/framing job, not
an experiment. See `reviewers_comments.md`.

---

## Gotchas that will cost you time if rediscovered

**New this session:**

- **`import torch` then `import sqlite3` is fatal on this host** (CXXABI_1.3.15).
  Fixed at `src/presley/__init__.py` — do not "tidy" that import away. It fails
  at **runtime**, not import time, for deferred in-function imports.
- **The Bash tool caps at 120 s regardless of a shell `timeout 900`** — pass the
  tool's own `timeout` parameter (ms, max 600 000) for anything longer.
- **A "stopped" background-job notification does not mean the work failed.**
  Check artifacts (result dirs, mtimes), not the notification. `pgrep -f` matches
  your own checking command and will lie to you.
- **`experiments.yaml` is the fragile artifact.** Append as text; never re-dump
  (destroys `# hash:` comments, risks mid-entry corruption). Verify every time:
  entry-count delta, first-N hashes byte-identical, all distinct.
- CI flake seen twice: a run can die at 20 min on a hung `apt-get install ffmpeg`
  **without reaching the tests**. Infrastructure, not code — the next run is the
  real verdict.

**Carried forward:**

- Filter fixed-QP with **`codec=svtav1`**; bare `restorer=` rematches VBR dirs.
- NAFNet / Real-HAT-GAN reject `fp32=False`.
- Stream-DiffVSR uses an isolated env — never pip into `presley`.
- Q10 morph fair A/B is **UFO none**, not gt.
- `fg_protect=True`; BG-PSNR is never the Goal-2 verdict.
- Benign log noise: `[av1] Missing Sequence Header`, `Failed to get pixel
  format`, SVT-AV1 banner.
- Substantive code → worktree + branch. Run-only stays on this checkout.
- Never `rm` wholesale `results/` / `dataset/` / `cache/`.

## Tools written this session

- `tools/analyze_ratematched.py` — BD-rate over the 4 rungs. **Refuses to run on
  incomplete curves**; forbids quoting a BD number when quality ranges are
  disjoint.
- `tools/analyze_f1_oracle.py` — F1; prints the random null and flags
  near-chance videos.

Both apply their pre-registered decision rules mechanically and label breaches
`*** BREACH ***` rather than leaving it to the reader.

## Prompt for the next session

```
Pick up HANDOFF.md in /home/itec/emanuele/presley.

Done and pushed: rate-matched breadth (chain presley_ai > elvis CONFIRMED
outside DAVIS, BD-rate -51.4% on BG-LPIPS, 9/9); F1 oracle-bits (claim (a) now
CLAIM-grade, capture 0.833 vs a 0.402 random null); bike-packing NAFNet
reclassified as a numerical defect.

Two decisions need you before agent work:
  (a) the saturation invariant — would mark existing runs uncitable
  (b) whether to run baselines at QP 42/47 (optional second wave)

Then, in order: claim (b) via a probe_* component (read
src/presley/components/probe_oracle_bits.py first — the citability trap is
solved there); open question 0c (drift-straight near-chance, unexplained);
carried-forward Q6 (ASK first) / Q8 (blocked upstream) / standing HOLEs.

Gotcha: never import torch before sqlite3 on this host.
```

## Landmarks

| Path | Why |
|---|---|
| `docs/RATEMATCHED_BREADTH.md` | Rate-matched design, bounds, result |
| `docs/F1_ORACLE_BITS.md` | F1 design, bounds, corrections, result |
| `src/presley/components/probe_oracle_bits.py` | Template for any future measurement probe |
| `src/presley/__init__.py` | The sqlite3/torch ordering fix — load-bearing |
| `research-log/bugs.md` | NAFNet defect (top entry) |
| `research-log/open-questions.md` | Items 0/0b/0c |
| `docs/EXPERIMENTS_QUEUED.md` | Q6/Q8 queue status |
