# Wave 2A — the content axis

**Status:** pre-registered 2026-08-03, before any correlation was computed.
Wave 2A of `docs/PLAN_OPERATING_MAP.md`, building on
`docs/WAVE1_OPERATING_MAP.md`. No GPU time, read-only against
`/home/itec/emanuele/presley/results/` and `.../cache/`.

The question: **does any content attribute predict transport choice?** This is
a different question from the two attribute tests that already came back
negative — those tested prediction of *win/loss* (does the pipeline work on
this clip). FG area is specifically refuted for that question (dog 0.114 ≈
bear 0.111, opposite outcomes) and is therefore **not** among the candidates
below.

**Pre-registered expectation: this fails again.** Two attribute tests have
already returned negative. A third negative is a legitimate result — it turns
the operating map into an empirical lookup table rather than a predictive
rule, which is a weaker but honest contribution. This section was written and
committed before any attribute–outcome number was computed.

---

## Part 1 — Pre-registration

### The three targets

Wave 1 partitions 84 operating points, under the quality-first objective:

| verdict | cells |
|---|---|
| `separable` (a JND-separable winner) | 19 |
| `tie_within_threshold` | 27 |
| `single_eligible_arm` | 14 |
| `no_eligible_arm` | 24 |

- **T1 — which arm wins**, over the 19 separable cells.
- **T2 — the ties**: among the 46 *contested* cells (19 separable + 27 tie),
  what predicts whether anything separates?
- **T3 — the dead cells**: among all 84, what predicts `no_eligible_arm` —
  no arm both saves bits and holds FG within JND?

**T1 is declared untestable by construction, before any attribute is
computed.** Wave 1 already published the winner distribution: 18 of 19
separable cells name `downsample+realesrgan` and 1 names
`blackout+propainter`. The outcome variable has essentially no variance, so no
attribute can be shown to predict it and none will be tested against it — an
attribute that "predicted" an 18/1 split would be predicting a constant. This
is recorded as a structural negative, not as a test that failed. It also means
**the strongest possible result available to Wave 2A is a predictor of T2
and/or T3**, never of "which transport to pick".

### Candidate attributes (k = 4, fixed here, nothing added later)

All from EVCA scores already cached under `cache/<video>_<WxH>_bs8/`
(`evca_SC_blocks.csv`, `evca_TC_blocks.csv`, `evca_EVCA_reference_raw.csv`),
which give per-block-per-frame spatial complexity (SC) and temporal complexity
(TC):

| id | attribute | definition |
|---|---|---|
| A1 | motion magnitude | mean TC over all blocks and frames |
| A2 | temporal (in)stability of the hole region | mean TC over foreground blocks |
| A3 | background texture energy | mean SC over non-foreground blocks |
| A4 | residual-information proxy | mean of the `B` column of EVCA's per-frame reference summary |

Foreground blocks are defined uniformly for every video as blocks whose
coverage by the **first annotated frame** of `dataset/annotations/<video>/`
exceeds 50%, the mask nearest-resized onto the EVCA block grid. First frame
only, because the MOSEv2 and YouTube-VOS clips carry exactly one annotated
frame; a richer per-frame definition would be available for the DAVIS clips
only and would make the attribute mean something different for different
videos. Uniform beats richer-but-uneven here.

Attribute source directory: `<basename>_640x360_bs8` where it exists, else the
video's only available `_bs8` directory (`ptq7rtia` is 360x640, `0e4068b53f`
is 640x480). Block size is fixed at 8 for all videos.

### Unit of analysis and test

**The unit is the video, not the cell.** Attributes are constant within a
video, so the 84 cells are not 84 independent observations — treating them as
such would inflate n roughly four-fold and is the most likely way to
manufacture a false positive here. Per video we compute an outcome *rate*:

- T2: separable / (separable + tie), over that video's contested cells.
- T3: no_eligible_arm / all that video's cells.

n = 23 videos for T3; for T2, the videos with ≥1 contested cell. Hard rule 2b
requires **n≥6 videos** before any significance claim; if a target falls below
6 it is reported descriptively and no claim is made.

Statistic: **Spearman rho** between attribute and per-video outcome rate,
two-tailed p by permutation (10,000 permutations, seed 0). **Holm correction
over the 4 attributes within each target**, as hard rule 2b requires. Both
targets are reported whether or not anything fires; k=4 is stated here and the
correction uses 4 regardless of how many attributes turn out to be computable.

### Decision rule (fixed before looking)

- **Predictive** — Holm-adjusted p < 0.05 **and** |rho| ≥ 0.6, on T2 or T3,
  with n ≥ 6 videos.
- **Suggestive only** — |rho| ≥ 0.6 but Holm p ≥ 0.05: reported as
  underpowered, explicitly *not* a finding.
- **Negative** — anything else. The map stays an empirical lookup table.
- A predictor that fires on T2 while T3 looks random (T1 being degenerate) is
  **a weak result and will be labelled as such**, not headlined.

### Bounds, stated before reading any number

| quantity | plausible | alarm |
|---|---|---|
| \|rho\| for a null attribute, n≈20 | 0.00–0.45 | — (0.45 ≈ 1.96/√19) |
| any single \|rho\| | ≤ 0.9 | **> 0.9** — investigate confounding or a bug before reporting |
| attributes surviving Holm across both targets (8 tests) | 0–1 | **≥ 3** — suspect a shared confound, not four independent discoveries |

A fired bound is recorded as revised-with-a-reason, never dropped.

### Confounds to check regardless of the outcome

1. **Dataset provenance.** 9 of the 23 videos are MOSEv2/YouTube-VOS clips and
   they carry a disproportionate share of both the separable and the dead
   cells. Any attribute that separates DAVIS from the newer clips may be
   predicting provenance and coverage history rather than content. Reported
   per attribute as a rank correlation with the source split; if that exceeds
   the attribute's correlation with the outcome, the attribute is not a
   content predictor and will be said so.
2. **Unequal cell counts.** Videos contribute 1–12 cells. Rates from a single
   cell are 0 or 1 with no precision. Robustness check: repeat with videos
   contributing ≥2 cells, and repeat leaving out `bear` and `camel` (the two
   densest).
3. **Operating-point mix.** Videos differ in which codecs, QPs and resolutions
   they were run at, so a per-video rate mixes rate points. This is a known
   limitation of using existing data and is not correctable within Wave 2A.

---

## Part 2 — Method as run

*(filled in after the pre-registration commit)*

## Part 3 — Results

*(filled in after the pre-registration commit)*

## Part 4 — What the map becomes if this fails

*(filled in after the pre-registration commit)*
