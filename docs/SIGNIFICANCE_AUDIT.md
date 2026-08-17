# Significance audit — re-examining "within JND, therefore a tie"

Companion to the hard-rule 2b amendment (`research-log/hard-rules.md`) and
`src/presley/suite.py`. Re-runs every landed claim currently worded as a
within-JND tie through the suite layer, to answer one question: **does any of
them now read differently?**

**Headline: nothing upgrades. Not one tie becomes a win.** No human decision is
needed on any claim's *direction*. What does change is the *reason* several
claims are ties, and that changes what to do next — which is a decision, and
it is flagged as one at the bottom.

Method: exact two-tailed sign test + exact Wilcoxon + seeded bootstrap CI,
Holm-corrected over the candidate family, plus the equivalence (TOST) mirror.
Reproduce any row with `presley-compare --suite`; the restorer rows below came
from the real `results/` tree, the F3/F6 rows from the tables in
`docs/WAVE1_FALSIFIERS.md`.

## The finding that drives everything else

The exact two-tailed sign test bottoms out at **2/2ⁿ**. So:

| n | smallest attainable two-tailed p | can it ever reach α=0.05? |
|---|---|---|
| 2 | 0.5000 | no |
| 5 | 0.0625 | **no** |
| 6 | 0.0312 | yes (uncorrected) |
| 9 | 0.0039 | yes, up to k=12 candidates |

Exact Wilcoxon has the **same** floor at these n — switching test is not an
escape hatch. Two consequences:

1. **The motivating case does not clear α.** F3's 5/5 LPIPS result is
   **p=0.0625**, not 0.031. The 0.031 figure is the *one-tailed* value, and
   one-tailed is indefensible here because the direction was read off the data
   before the test. F3 is therefore `underpowered` — an unfinished experiment,
   not a negative result. **A sixth video would settle it**, and that is the
   cheapest open item in this audit.
2. **Every landed restorer twin comparison is n=2**, where the floor is 0.5.
   None of them can be moved by any amount of statistics. They need videos,
   not tests.

## Claim-by-claim

| Claim | metric | n | direction | sign p (corrected) | verdict | change? |
|---|---|---|---|---|---|---|
| F3 `tune=0` vs `tune=1` | FG-LPIPS | 5 | 5/5 better | 0.0625 | `underpowered` | **reason changes** |
| F3 `tune=0` vs `tune=1` | FG-DISTS | 5 | 5/5 better | 0.0625 | `underpowered` | corroborating only |
| F3 `tune=0` vs `tune=1` | PSNR | 5 | 5/5 worse | 0.0625 | `underpowered` | unchanged |
| `CLAIM(tab:conditioned-twins)` Real-HAT-GAN | BG-LPIPS | 2 | 2/2 better | 1.0000 (k=6) | `underpowered` | **reason changes** |
| `CLAIM(tab:conditioned-twins)` BSRGAN | BG-LPIPS | 2 | 2/2 worse | 1.0000 (k=6) | `underpowered` | **reason changes** |
| `CLAIM(tab:conditioned-stream-diffvsr)` | BG-LPIPS | 2 | 2/2 worse | 1.0000 (k=6) | `underpowered` | **reason changes** |
| Q5 NAFNet vs `unsharp` | BG-PSNR | 2 | — | 1.0000 (k=6) | `underpowered` | **reason changes** |
| F6 NAFNet on FG | FG-PSNR | 5 | 5/5 worse | 0.0625 | `underpowered`, TOST **equivalent** | **strengthened** |
| `CLAIM(tab:ablation)` α/β grid | FG-PSNR | 18 | n/a (equivalence) | n/a | TOST **equivalent** | **under-stated today** |

### "Reason changes" — ties that were never ties

Six claims are currently worded as *ties* ("both tie Real-ESRGAN within JND",
"a wash", "near-tie"). A tie asserts **evidence of no difference**. What the
data actually support at n=2 (and n=5) is **no evidence either way**. Those are
different statements with different consequences: a tie closes a question, an
underpowered result leaves it open.

This is not a licence to reopen them as potential wins — the magnitudes are all
comfortably sub-JND, so even if a larger suite reached significance, the
ceiling is `sub_jnd_significant`, which is explicitly not a perceptual win. The
practical conclusions (**keep Real-ESRGAN**, on throughput) are untouched, and
they were never resting on the tie in the first place.

Suggested rewording, which is a claim-strength change and therefore the human's
call: replace "ties Real-ESRGAN within JND" with "is within JND of Real-ESRGAN
on n=2 videos, which is too few to distinguish equivalence from a small effect;
Real-ESRGAN is kept on throughput."

The TOST mirror does return `equivalent` on these rows, but at n=2 it fires its
own not-credible warning: a bootstrap over two points reproduces the observed
range rather than a sampling distribution, so it will certify almost any tight
pair. **Do not cite those as demonstrated equivalence.**

### F6 NAFNet-on-FG — strengthened, not weakened

The 5/5 direction is underpowered exactly like F3. But F6's conclusion
("NAFNet is the wrong FG restorer; nothing here is worth gating") is a
*negative* claim resting on magnitude — deltas 10–50× below the 0.5 dB JND —
and that is an equivalence argument, which now has a test. TOST returns
`equivalent` with n=5. **F6's conclusion is better supported than it was**, and
for the right reason.

### `CLAIM(tab:ablation)` — currently under-stated

The α/β robustness claim is the one place the current framework is too
*conservative*. FG-PSNR spans 30.46–30.49 (bear) and 31.96–32.01 (camel) — a
0.05 dB spread against a 0.5 dB JND, 18 configs. TOST: **equivalent**, 90% CI
[+0.017, +0.028], every individual delta inside the bound.

Today the paper says these deltas are "within JND", which reads as "we could
not detect a difference". The data support the stronger and more useful
"α and β are demonstrably without perceptible effect over their whole range" —
which is exactly the robustness/no-per-content-tuning claim
`GOAL(tab:ablation)` set out to make. Upgrading this wording is the human's
call; it is the only *strengthening* this audit found.

## Multiple comparisons

The restorer matrix (`docs/EXPERIMENTS_QUEUED.md`) is explicitly a search:
Real-ESRGAN is the incumbent and **six** candidates have been tried against it
(Real-HAT-GAN, BSRGAN, NAFNet, InstantIR, Stream-DiffVSR, DC-VSR — DC-VSR
blocked, still counted, because a candidate that could have won belongs in the
family). At an uncorrected α=0.05 that is a ~26% chance of at least one
spurious winner.

Two rules follow, both now enforced in code:

- `--candidates-tried` takes the total **including losers**, and Holm
  correction is applied to it. Corrected, k=6 raises the minimum suite size
  from 6 pairs to **8**.
- The family is over **candidates**, never over **metrics**. LPIPS and DISTS
  agreeing 5/5 is close to one piece of evidence, not two — they are deep
  perceptual metrics on the same frames. One primary metric per goal (hard
  rule 3) carries the claim; the rest corroborate.

Note that `underpowered` verdicts do not consume error budget in the way a
tested-and-rejected hypothesis does, but the *pre-registration* obligation is
unchanged: the candidate count has to be declared, not reconstructed after a
winner appears.

## What to do next

1. **Add a 6th video to F3** (one cheap fixed-QP encode pair). At 6/6 the
   uncorrected p is 0.031 and F3 becomes the project's first
   `sub_jnd_significant` result — a clean demonstration of the new layer, on a
   claim that costs nothing to make and misleads nobody.
2. **Decide on the two rewordings above** (twins: tie → underpowered;
   `tab:ablation`: within-JND → demonstrably equivalent). Both are claim-
   strength changes, so neither was applied.
3. **`HOLE(tab:goal2)` / `HOLE(tab:conditioned)` n>2 are now quantified**: with
   k=6 candidates the target is **n≥8 paired videos**, not "more than 2".

---

# Extension — every CLAIM anchor, 2026-08-17

The 2026-07-29 audit above covered the restorer twins, the Wave-1 F3/F6 rows and
the α/β ablation. This extension sweeps **all 45 CLAIM anchors** in the
manuscript and classifies each, because "have we tested significance everywhere"
had never been answered for the anchors that landed after July.

## Method

For each `% CLAIM(...)` block: collect its inline `src=` hashes, resolve each to a
video via `results/<hash>/result.json`, and classify the claim as **comparative**
(asserts a direction or ordering between arms — needs a test) or **descriptive**
(reports a magnitude, a timing, or a dispersion — owes no test). The relevant
floor is the exact two-tailed sign test's 2/2ⁿ, so n≥6 uncorrected and n≥8 once
Holm-corrected over a candidate family of 6.

## Result

| | count |
|---|---|
| CLAIM anchors with on-disk provenance | 34 |
| — comparative (a test is owed) | **11** |
| — descriptive (no test owed) | 23 |
| comparative **and** below the n≥6 floor | **7, of which ~5 are real** |

The seven, with a judgement on each:

| claim | n (inline) | verdict |
|---|---|---|
| `sec:downsample-vs-uniform` | 1 | **artefact of the count** — genuinely n=6; only `bear`'s hashes are inline, the rest are in `docs/PLAN_DOWNSAMPLE_VS_UNIFORM.md` |
| `tab:ratecontrol` | 3 | **artefact** — the VBR finding is 25/25 matched *pairs*, paired at (video, method), not at video |
| `block-damage` | 4 | **descriptive, misclassified** — a within-run dispersion, not a directional comparison |
| `tab:budget-knee` | 2 | borderline; it asserts the lever is "inert below 0.25", which is an equivalence claim and wants TOST rather than more videos |
| `tab:conditioned-twins` | 2 | **real and already known** — the sign test floors at p=0.5 |
| `tab:conditioned-stream-diffvsr` | 2 | **real and already known** — same family |
| `sec:restoration-comparison` | 2 | **real** — the same twins claim, restated in `presley.tex` |

## What this changes

**The picture is much better than a raw count of claims-below-n=6 suggests.** Most
CLAIM anchors are descriptive — timings, throughput, dispersions, side-channel
cost — and owe no significance test at all. The genuine shortfall is concentrated
in **one family**: the conditioned-restorer twins, which is exactly what the July
audit identified and what `HOLE(tab:goal2)`/`HOLE(tab:conditioned)` already
record. Plan item F17 (extend to n≥8) closes all of it.

Two smaller actions fall out:

1. **`tab:budget-knee` is an equivalence claim wearing a difference claim's
   clothes.** "Inert below 0.25" asserts no effect; a failed difference test
   cannot support that. Run `suite.equivalence_tost` against the JND margin, the
   same fix the July audit applied to `tab:ablation`.
2. **Inline `src=` lists that carry only a representative subset are a provenance
   hazard**, because any future audit undercounts them exactly as this one did
   twice. Either inline every hash or state the count in the CLAIM line.

## Standing rule this establishes

A CLAIM block should say which kind it is. Adding `kind=comparative` or
`kind=descriptive` to the anchor line would make this audit mechanical instead of
a judgement call over 45 blocks, and would stop a descriptive claim from being
read as an untested comparison.
