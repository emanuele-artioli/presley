# Pre-registration — overnight wave, 2026-08-17

Written **before** any run in this wave executed. Bounds and alarms are stated
first, per `research-log/hard-rules.md`. A result outside its band is an alarm:
investigate implementation/eval/data before reporting it, and never cite it until
the alarm is closed or the band is revised **with a stated reason**.

Wave plan and rationale: `~/.claude/plans/if-you-can-t-see-transient-pine.md`,
Part F. Every experiment here needs **no code change** — that is the selection
criterion for this wave.

---

## W1-A — Breadth on a common four-rung ladder (plan F13, absorbs F10)

**Why.** `fig:breadth` currently plots a *matched-QP bitrate delta* and reports it
as a saving. A saving is only a saving at matched quality, which needs a BD-rate,
which needs a real ladder. The ELVIS arm has two rungs (QP 32/37) and the PRESLEY
arm four (32/37/42/47) on a partly different clip set, so the figure also compares
unequal numbers of clips.

**Design.** Harmonize both arms onto **QP {32, 37, 42, 47}** — the ladder the
PRESLEY arm already uses, so no new rung is invented — over the union of the
breadth clips, plus the matching pristine baselines. Only missing cells are run;
`presley-run` skips any hash that already has a `result.json`.

Fixed: `x265`, `preset medium`, 640×360, `block_size 8`, `alpha=beta=0.5`,
`shrink_amount 0.25`, `fg_protect true`, `composite_output true`.
ELVIS arm `removal_mode blackout`, `inpainter none` (transport-only, as landed).
PRESLEY arm `degradation downsample`, `restorer realesrgan`.

**Bounds.**

| quantity | plausible band | basis |
|---|---|---|
| ELVIS-arm BD-rate vs baseline, per clip | −60% … +60% | the existing 2-rung deltas span −80%…+45%; BD-rate integrates so should be tighter |
| PRESLEY-arm BD-rate vs baseline, per clip | −60% … +60% | same |
| fraction of clips saving bits at matched quality | 0.30 … 0.80 | the matched-QP version is 22/33 ≈ 0.67; matched-quality should be **lower**, since some of that saving was bought with quality |
| FG-LPIPS delta, either arm | ≤ 0.05 on ≥ 80% of clips | `fg_protect` is on; the existing breadth run holds under 0.02 on most clips |

**Alarms.**
- Any clip whose BD-rate is outside ±60% → check ladder overlap before reporting;
  a non-overlapping ladder produces meaningless extrapolated BD-rate.
- **Overlap fraction < 0.50 on any clip → that clip is not reportable**, same gate
  `tools/analyze_ratematched_n13.py` applies.
- Matched-quality saving fraction **higher** than the matched-QP 0.67 → alarm.
  Holding quality fixed should cost, not gain; if it gains, suspect the BD
  direction convention.
- Any run with non-empty `invariant_failures` is dropped and re-run, never
  reported with a disclosure.

**What it settles.** Whether "the bridge saves bits on most clips" survives being
asked at equal quality. It may not — that is an acceptable outcome and would
replace a wrong claim with a correct one.

---

## W1-B — Graded λ at a block size that can carry it (plan F7, closes D3)

**Why.** `CLAIM(tab:graded)` retired graded multi-level downsampling, but it was
measured at `block_size 16`, where the method section's own bound
λ ≤ log2(b/8) admits **one** level — so its deepest rung carried each block at
2×2 samples. The retirement was measured where grading cannot work.

**Design.** `block_size 64` at 1920×1080, where the bound admits three levels, so
the deepest rung leaves 8×8 samples. Two arms differing **only** in
`downsample_levels` (absent = binary, 3 = graded), on the eight probe clips, four
fixed-QP rungs, plus pristine baselines. Fixed: `svtav1 preset 8`,
`shrink_amount 0.25`, `fg_protect`, `realesrgan`, `composite_output`.

**Bounds.**

| quantity | plausible band | basis |
|---|---|---|
| graded-vs-binary BD-rate on BG-LPIPS | −25% … +25% | at b=16 grading cost bits on 7/8; at b=64 the degeneracy is removed, so the honest prior is centred on zero |
| clips where graded beats binary | 2 … 6 of 8 | anything at the extremes is suspicious at this n |
| FG-LPIPS delta between arms | ≤ 0.02 | both arms hard-exclude the foreground, so the FG must be near-identical |

**Alarms.**
- FG differs by more than 0.02 between arms → the exclusion is not holding and the
  comparison is invalid.
- Graded better on 8/8 → too clean for a knob that failed at b=16; check that
  `downsample_levels` actually took effect (the level map should have ≥3 distinct
  non-zero values).
- Either arm's realized removal rate departing from the 0.25 budget → the level
  floor is misbehaving.

**What it settles.** Whether the graded transport is genuinely retired or was
retired out of regime. **Either outcome is publishable**, and the current
`sec:ablation` wording is unsafe until one of them exists.

---

## Reporting rule for this wave

Numbers enter the paper only through a `CLAIM(anchor)` with `src=` hashes and an
empty `invariant_failures` on every cited run. Nothing from this wave may be
worded as a win without passing `presley-compare` (JND gate) and, for N>1, the
suite layer with its two-tailed test and Holm correction over candidates tried.

---

## W1-C — Placement control, 2×2 factorial (plan F1; F2 retired)

Added 2026-08-17, **before any W1-C run executed**. F2 ("matched-footprint
uniform") is retired: uniformity means covering the whole frame, so constraining
it to 25% coverage makes it a different placement of the same budget rather than
a uniform arm, and at λ=1 there is no milder per-block strength to match with.

**Why.** PRESLEY asserts that *placement* matters — it ranks blocks by
removability and hard-excludes the foreground — and the article contains no
measurement of either. `f1-oracle-bits` measures the ranking against a *bit*
oracle (a proxy, not delivered quality); `sec:downsample-vs-uniform` varies
placement and coverage together and isolates neither.

**Design.** Budget (0.25) and strength (λ=1) fixed. Two binary factors:

| arm | `selection` | `fg_protect` | contrast against A isolates |
|---|---|---|---|
| A | score | true | incumbent |
| B | random | true | **the ranking** |
| C | score | false | **the exclusion** |
| D | random | false | both; a floor for the pair |

6 videos × 4 rungs × 4 arms, `svtav1 preset 8`, 640×360, `block_size 8`,
`downsample` + `realesrgan`, `composite_output`, seed fixed at 1.

**Validity condition, enforced in code and tested.** The random map is
substituted *before* the clustering blur, so arms B and D get contiguous patches
like the incumbent. An unblurred random map scatters into singletons, which this
project already measures as expensive to code — a random arm without the blur
would lose on **fragmentation** rather than on placement and would answer a
different question. `tests/test_random_selection_control.py` asserts this, plus
equal budget across arms and that the exclusion still binds on the random arm.

**Bounds.**

| quantity | plausible band | basis |
|---|---|---|
| A vs B, BD-rate on BG-LPIPS | −20% … +5% | the score captures 0.833 of oracle bits against a 0.402 random null, so score placement should help; but the whole cost-axis headroom is only ~5% of bitrate, so a large win is not expected |
| A vs C, FG-LPIPS delta | **≥ 0.03 on most videos** | dropping the exclusion degraded 12.8% of foreground blocks on `bmx-trees` and cost 3 dB FG-PSNR |
| A vs D | worst of the four on FG | both protections removed |
| realized removal rate, every arm | 0.25 ± 0.01 | budget is matched by construction |

**Alarms.**
- **B beating A** on background BD-rate → the ranking is worse than chance;
  check the blur is applied to the random map and that the seed varies per frame.
- **A vs C showing no foreground difference** → the exclusion is not binding;
  invalidates the arm, not the design.
- Removal rate departing from 0.25 on any arm → budgets are not matched and no
  contrast is interpretable.
- Any arm's FG-LPIPS better than A's → suspect the evaluation is inheriting the
  method's mask (hard rule 7).

**What it settles.** Whether the two placement choices the architecture is built
on do measurable work at fixed budget and strength. A null on the ranking is a
publishable result and would say the budget, not the ordering, is what matters.
