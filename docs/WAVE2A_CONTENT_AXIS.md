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

Reproduce:

```bash
python tools/build_operating_map.py --db results/presley.db --json map.json
python tools/analyze_content_axis.py --map map.json \
    --cache cache --annotations dataset/annotations --json content.json
```

Tests: `tests/test_analyze_content_axis.py` (20 cases, fast tier; whole suite
387 passed). No GPU, no runs, read-only.

Everything in Part 1 was executed as written. Two things were **added** after
the fact, and both are labelled as such wherever they appear:

- **A cell-count confound check.** Added after A1 fired. A video's cell count
  is a record of what was run, not a property of its content.
- **An adjudication step** requiring a fired attribute to survive both its
  pre-registered robustness subset and that confound before it is reported.

One bug was found and fixed mid-analysis: the cache lookup flattened video
names to a basename, so the nine MOSEv2/YouTube-VOS clips (nested under
`cache/mosev2/`, `cache/youtube_vos/`) silently dropped out and attributes
covered only 14 of 23 videos. **The truncated run produced a false positive**
— A1 and A3 came back with byte-identical rho and p, which is what flagged it.
Every number below is from the corrected 23-of-23 run.

## Part 3 — Results

**The content axis is negative.** One attribute fired under the pre-registered
rule and was withdrawn on its own robustness check.

### T1 — which arm wins: no variance to predict

Confirmed as pre-registered, by computation rather than assumption:
`downsample+realesrgan` 18, `blackout+propainter` 1, modal share **0.947**.
No attribute was tested against it. **This is the single most important
negative in Wave 2A**: even a perfect content predictor could not have told a
deployer which transport to pick, because under the quality-first objective
the map does not offer a choice — it offers one arm and one exception.

### T2 — separable vs tie (46 contested cells, 19 videos)

| attribute | n videos | rho | p | Holm p | pre-registered verdict |
|---|---|---|---|---|---|
| A1 motion magnitude | 19 | **+0.642** | 0.0043 | **0.0172** | PREDICTIVE |
| A2 hole-region instability | 19 | +0.182 | 0.451 | 0.903 | negative |
| A3 background texture | 19 | +0.314 | 0.187 | 0.560 | negative |
| A4 residual-information proxy | 19 | −0.038 | 0.889 | 0.903 | negative |

A1 met the pre-registered bar. It does not survive:

- **Its own pre-registered robustness subset.** Restricted to the 9 videos
  with ≥2 contested cells, A1 collapses to **rho +0.297, Holm p = 1.00**.
  10 of the 19 videos contribute exactly one contested cell, where the "rate"
  is necessarily 0 or 1.
- **The cell-count confound.** A1 correlates with a video's cell count at
  **rho +0.634** — as strongly as with the outcome (+0.642) — and the outcome
  itself tracks cell count at **+0.604**. A video's cell count is run history.
- **A non-content variable that beats it outright.** Dataset provenance alone
  (DAVIS vs MOSEv2/YouTube-VOS) predicts T2 at **rho +0.907, p = 0.0002** —
  and that fires the pre-registered |rho| > 0.9 alarm, so it is reported as a
  confound to investigate, not as a discovery. The individual attributes are
  nearly orthogonal to provenance (|rho| ≤ 0.12), so provenance is not acting
  through them: it carries structure the content attributes do not.

The mechanism of the artifact is visible in the raw table: every video with a
T2 rate of 1.0 is a sparse MOSEv2/YouTube-VOS clip with 1–3 contested cells,
while the two densest clips (`bear` 10 cells, `camel` 6) sit at 0.30–0.33.
A1 is a weak correlate of that split, not of separability.

**A1 is withdrawn.** Reported as an association with corpus and coverage, not
with content.

### T3 — no deployable arm (84 cells, 23 videos)

All four negative, nothing near the bar:

| attribute | n videos | rho | p | Holm p |
|---|---|---|---|---|
| A1 motion magnitude | 23 | −0.249 | 0.253 | 0.650 |
| A2 hole-region instability | 23 | −0.473 | 0.025 | 0.100 |
| A3 background texture | 23 | −0.271 | 0.217 | 0.650 |
| A4 residual-information proxy | 23 | −0.086 | 0.700 | 0.700 |

A2 is the only raw p under 0.05 and it does not survive Holm over the four
pre-registered candidates. Per the decision rule it is **negative**, not a
near miss, and it is not quoted as one. Its sign (less stable hole region →
*fewer* dead cells) is also opposite to the intuition that would have been
offered had it survived, which is a reason for more caution rather than less.

In the ≥2-cells subset, A1, A2 and A3 all clear Holm at once (−0.731, −0.628,
−0.656). **This is the pre-registered ≥3-attributes alarm firing**, and it is
reported as an alarm: the three are collinear EVCA means, the effect is absent
in the full sample and absent again when `bear` and `camel` are dropped, so a
result present in exactly one of three subsets and only as three correlated
attributes together is a confound, not a discovery.

### Bounds, checked against what was read

| quantity | pre-registered plausible | observed |
|---|---|---|
| \|rho\| for a null attribute | 0.00–0.45 | 0.03–0.47 — in band |
| any single \|rho\| | ≤ 0.9 | **0.907 (outcome vs provenance) — ALARM fired** |
| attributes surviving Holm over both targets | 0–1 | **1** (A1/T2), then withdrawn — in band |

The |rho| > 0.9 alarm fired on a variable that is not a content attribute, and
was handled as the pre-registration requires: investigated, and reported as a
confound rather than a finding. No bound was revised.

### What was *not* tested, and why

FG area — refuted for win/loss by prior work, not re-run. Per-frame mask
dynamics for the DAVIS clips — deliberately excluded so that the foreground
definition means the same thing for all 23 videos. Cell-level correlations —
computable and roughly 4× more "significant", and exactly the inflation the
video-level unit exists to prevent.

## Part 4 — What the map becomes

**An empirical lookup table.** This is the third consecutive negative content
result, and the first two tested a different question, so it is not a
replication — it is an independent negative on the axis the plan called "the
open problem".

Concretely, for the article:

1. **Do not claim a content rule.** There is no defensible "on high-motion
   clips, use X". The honest statement is: *which arm to deploy is read off a
   table indexed by measured operating point, and the content attributes we
   can compute do not shortcut that lookup.*
2. **The lookup table is still the contribution**, and Wave 1 is what makes it
   one — the map exists, it has a separable winner in 41% of contested cells,
   and its recommendation depends on what the deployer optimises. None of that
   needed a content rule.
3. **State the ceiling honestly.** Under quality-first, the map's answer is
   `downsample+realesrgan` in 18 of 19 separable cells. The map's real content
   is therefore *where it declines to recommend* — the 27 ties and the 24 dead
   cells — more than *what* it recommends.
4. **The interesting negatives stay open.** Nothing predicts the ties and
   nothing predicts the dead cells. Both look like properties of the operating
   point rather than of the clip, which is a hypothesis Wave 2B's added
   coverage could test and Wave 2A's data cannot.
5. **The corpus is confounded with coverage.** The strongest correlate of
   anything here is which dataset a clip came from, entangled with how densely
   it was run. Wave 2B should be told this: adding operating points to the
   sparsely-covered clips is what would break the entanglement, and until it
   does, per-video rates over this corpus should not be correlated with
   anything.

**What would change the verdict.** Not more attributes on this corpus — the
limiting factor is that 10 of 19 videos contribute a single contested cell.
Balanced coverage (Wave 2B) at ≥3 contested cells per video across ≥6 videos
would let this test be re-run with rates that carry information. Re-running
`analyze_content_axis.py` after Wave 2B costs nothing and is the right trigger.
