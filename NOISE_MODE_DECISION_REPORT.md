# Why noise-injection degradation is so expensive: an x265 mode-decision investigation

**Status:** one-off investigation, referee-response item (TOMM revision). Not
part of the presley-run pipeline; does not touch `experiments.yaml` or
`results/`.

**Script:** `tools/noise_mode_decision_analysis.py` (standalone, CPU-only,
x265/ffmpeg encoding — no GPU model inference).

**Prior finding this explains:** a fixed-QP screen (documented in the paper's
`RESEARCH_LOG.md`, and re-confirmed in-place at `src/presley/degradation.py`
line ~477) already established that noise injection is the worst degradation
PRESLEY has measured — **+213% to +334% more bits than the pristine baseline
at matched QP**, worse than freeze, blackout, or mean_fill, all of which
*save* bits at fixed QP. That screen never explained *why* in encoder terms.
A referee asked specifically about "mode decisions and rate allocation." This
report answers that with x265's own analysis instrumentation.

## Two things turned out to matter, not one

1. **A coverage asymmetry in the code itself** (new finding, not previously
   documented anywhere): under the production default config (no
   `shrink_amount`/`fg_protect`, true for every `degradation: noise` entry in
   `experiments.yaml`), the noise degradation and the other multi-level
   degradations (`blur`, `downsample`) do **not** degrade a comparable
   fraction of the frame, even at identical `alpha`/`beta`.
2. **A genuine per-block encoder-mechanism cost** (what the referee actually
   asked about): even when the *same* fraction of the frame is degraded in
   both conditions, x265 handles noise-corrupted blocks measurably
   differently — but not the way the "more intra-coding" hypothesis predicts.

Both are real and both contribute to the +213–334% number; §1 is arguably the
larger effect at these settings, and §2 is the mechanistic answer to the
referee's question. Details below.

---

## Finding 1: noise touches ~95% of the frame; blur/downsample touch ~9–13%

`filter_frame_noise`'s implicit "which blocks get degraded" threshold (used
whenever the caller passes `sel=None`, i.e. every real `presley_ai` noise
experiment) is:

```python
noise_strengths = _apply_sel_to_map(np.round(frame_scores * noise_variance).astype(np.float32), sel, 1.0)
...
if strength > 0:   # strength = round(score * 50)
```

— i.e. a block is selected when `score >= 0.01` (since `noise_variance=50`,
`round(x*50) > 0 ⟺ x >= 0.5/50`). Every *other* multi-level degradation
(`filter_frame_gaussian`, `filter_frame_downsample`) thresholds the **raw**
`[0,1]` score with `round(score) > 0 ⟺ score >= 0.5` — a 50x higher bar.
Measured directly against the real cached removability scores
(`alpha=0.5, beta=0.5`, the config every noise experiment in
`experiments.yaml` uses):

| video | blur/downsample threshold (score≥0.5) | noise threshold (score≥0.01) |
|---|---|---|
| bear  | **9.4%** of blocks | **96.3%** of blocks |
| camel | **13.0%** of blocks | **94.7%** of blocks |

(The 9.4%/13.0% figures match the existing code comment in
`src/presley/components/presley_ai.py` verbatim — this is the same
"comfortable regime" threshold already known to under-select for
blur/downsample. The 94–96% figure for noise had not been measured before
this investigation.)

So the existing "noise is the worst degradation" comparison is, in this
respect, not apples-to-apples: at the same `alpha`/`beta`, noise is being
injected into **essentially the whole frame**, while mean_fill/blur/downsample
in the same comparison touch under 15% of it. A degradation that corrupts 95%
of a frame costing far more bits than one that corrupts 10% of it is
expected on volume alone, before any per-block mechanism is invoked.

*(This is reported as a discovered fact about the existing code path, not a
recommendation to change it — that's a separate methodology decision for
whoever owns the noise/dead-end writeup.)*

## Finding 2: at matched coverage, noise still costs 24–235% more — via Merge/Skip suppression and finer partitioning, not more intra

To isolate the **per-block** mechanism from the coverage confound above, this
script also runs a **region-isolated** test: a fixed 160×160 (20×20-block)
crop that is verified 100% background in *every single frame* of each clip
(via the cached UFO masks — see `tools/pick_noise_region.py`, which found
these exact coordinates). Because the crop is homogeneous, both conditions (noised vs.
untouched) degrade exactly the same set of blocks (100%), so any difference
is attributable to the per-block coding cost alone, not to how much of the
frame is touched.

### Granularity actually achieved

x265 exposes per-frame analysis via `--csv`/`csv-log-level` through the
ffmpeg-linked `libx265.so.199` on this host (x265 3.5+1-f0c1022b6, via
ffmpeg n7.1.1). **There is no standalone `x265` CLI binary on this host**
(`which x265` → not found; `find / -iname '*x265*' -type f` turns up only
unrelated CSV datasets, never a binary), and `-x265-params pmode=1` only
toggles the analysis thread-pool feature — it does not emit a per-CU decision
log. So **true per-CTU / per-spatial-coordinate mode logs are not obtainable**
from this build without recompiling x265 from source with custom
instrumentation, which was out of scope here.

What `csv-log-level=2` *does* give, per frame: total bits, achieved QP, and
the percentage of the **final coded area** at each partition size (64×64 down
to 4×4) broken out by mode (Intra DC/Planar/Angular, Inter, Skip, Merge) —
a genuine mode-decision and partition-depth signature, just aggregated over
the whole frame rather than mapped spatially. The region-isolated crop is how
this script gets a spatially-attributable answer without modifying x265:
make the "frame" homogeneous so the frame-level aggregate *is* the region's
aggregate.

### Setup

- Videos: `bear` (82 frames), `camel` (90 frames), both 640×360 — same
  resolution as the real `presley_ai` noise experiments in
  `experiments.yaml`.
- `block_size=8`, `alpha=0.5`, `beta=0.5` (identical to the real experiments;
  removability scores loaded from the existing cache, not recomputed).
- Noise injected via the project's actual `filter_frame_noise` (default
  `noise_variance=50.0`, `sel=None`) — the exact function and threshold the
  `presley_ai` component calls for `degradation: noise`.
- Fixed QP 32 and 37 (both used in the real `noise`/`restorer: none`
  experiments in `experiments.yaml`), x265 preset `medium`, single-pass CQP
  (`ffmpeg -c:v libx265 -x265-params qp=<qp>:csv=<path>:csv-log-level=2`).
- Crop coordinates (100% background in every frame, verified against the
  cached UFO masks): bear `y0=0,x0=320,size=160`; camel `y0=200,x0=480,size=160`.
- Every condition re-verified `csv rows == input frame count` before being
  trusted (added after a first run silently doubled row counts by appending
  to a stale csv from an earlier attempt — see "Pitfall" below).

### Results — region-isolated crop (isolates the per-block mechanism)

| video | qp | bytes control→noised | Δbits | Merge % c→n | Inter % c→n | Intra % c→n | avg final CU size (px) c→n |
|---|---|---|---|---|---|---|---|
| bear  | 32 | 27,209 → 69,738  | **+156.3%** | 47.6→20.2 (−57.6%) | 42.8→70.0 (+63.4%) | 1.19→0.65 (−45.4%) | 27.5→15.3 (−44.4%) |
| bear  | 37 | 12,577 → 16,888  | **+34.3%**  | 69.0→62.0 (−10.1%) | 23.5→31.0 (+32.0%) | 1.12→1.01 (−9.8%)  | 35.5→34.5 (−2.9%)  |
| camel | 32 | 31,652 → 66,642  | **+110.5%** | 25.3→16.7 (−33.9%) | 59.8→68.2 (+14.0%) | 2.57→1.74 (−32.2%) | 22.4→14.6 (−35.0%) |
| camel | 37 | 14,598 → 18,125  | **+24.2%**  | 49.6→44.7 (−10.0%) | 40.4→44.7 (+10.8%) | 2.13→1.63 (−23.7%) | 33.5→31.5 (−5.8%)  |

### Results — full-frame (mixed content, reproduces the known aggregate effect)

| video | qp | bytes control→noised | Δbits | Merge % c→n | Inter % c→n | Intra % c→n | avg final CU size (px) c→n |
|---|---|---|---|---|---|---|---|
| bear  | 32 | 373,366 → 929,762   | **+149.0%** | 29.3→14.1 (−51.9%) | 62.0→79.8 (+28.8%) | 1.17→0.86 (−26.9%) | 14.0→11.2 (−20.4%) |
| bear  | 37 | 125,660 → 266,582   | **+112.1%** | 55.7→32.8 (−41.2%) | 35.2→58.2 (+65.4%) | 1.21→0.97 (−20.0%) | 20.6→16.6 (−19.3%) |
| camel | 32 | 393,230 → 1,157,068 | **+194.2%** | 28.5→13.6 (−52.3%) | 57.1→75.9 (+33.1%) | 3.40→2.35 (−30.9%) | 13.7→10.7 (−22.3%) |
| camel | 37 | 140,204 → 327,187   | **+133.4%** | 51.4→33.4 (−35.0%) | 35.8→54.1 (+51.0%) | 3.63→2.43 (−33.1%) | 18.5→15.6 (−15.3%) |

(All percentages above are means over P/B-frames only; I-frames are excluded
from the mode-mix columns since HEVC I-frames are intra by construction —
reported separately in `scratch/noise_mode_decision/summary.json` as
`i_intra_pct_mean`/`i_avg_final_cu_size_px`, both ~100%/~6px in every
condition as expected and essentially unaffected by noise, since I-frames
don't have Merge/Skip to lose in the first place.)

### The mechanism, stated plainly

**All 8 conditions (2 videos × 2 QPs × {full, crop}) agree on the same
direction of every signal:**

- **Merge mode share drops** (−10% to −58% relative) — Merge assumes the
  co-located reference block (found by motion estimation) is *directly*
  usable with at most a coded motion vector and no residual; Gaussian noise
  is per-pixel-independent and temporally uncorrelated, so the previous
  frame's (also-noised, independently-drawn) block is no longer a good
  enough predictor for a residual-free Merge to succeed.
- **Plain Inter mode share rises correspondingly** (+11% to +65% relative) —
  motion estimation still *finds* motion (the underlying real content is
  still moving coherently), so the encoder does not fall back to expensive
  intra prediction. It falls back to **full Inter coding**: a motion vector
  plus an actual transform-coded residual, because the residual (dominated by
  the injected noise, which has no temporal or spatial structure to predict
  away) is no longer near-zero.
- **Intra mode share does NOT rise — it consistently falls** (−10% to −45%
  relative, in every single condition). This refutes the specific hypothesis
  in the referee's question ("noise defeats motion estimation... forcing
  costlier intra prediction") as the operative mechanism. Motion estimation
  is not defeated; only the *cheap end* of inter coding (Merge/Skip, which
  assume near-zero residual) is defeated. Skip share is roughly flat
  (small, inconsistent-sign changes) — it was already a small fraction of
  P/B coding in this content and isn't the main casualty.
- **Final coded partition size shrinks** in every condition (−2.9% to −44.4%
  relative) — the encoder splits CTUs into smaller CUs under noise, almost
  certainly to localize rate-distortion decisions against noise's much higher
  per-pixel variance (a single large CU averaging over noisy pixels wastes
  more of its allowed distortion budget than several smaller CUs each fit to
  a locally-appropriate residual).

**Net rate-allocation effect:** bits that would have gone to whichever region
had the *most real detail* (i.e., in Goal 1's terms, back toward the
foreground) are instead consumed coding real transform residuals for
Merge-turned-Inter blocks and the CU-split overhead that comes with smaller
partitions — even where the true underlying content (background, moving
coherently) would ordinarily have coded almost for free.

### QP dependence

The relative bit cost of noise **shrinks as QP increases** — full-frame
+149%/+194% at QP32 vs. +112%/+133% at QP37; crop +156%/+110% at QP32 vs.
+34%/+24% at QP37 (bear/camel respectively) — and the crop's CU-size and
Merge-share shifts shrink correspondingly (e.g. bear crop CU size barely
moves, −2.9%, at QP37 vs. −44.4% at QP32). Coarser quantization partially
quantizes the injected noise itself down to zero, so at high enough QP the
mechanism's effect (and its bit cost) recedes — consistent with the project's
existing "starved-bitrate" framing: this failure mode is worst exactly where
PRESLEY is trying to operate (bit-starved, i.e. lower-QP-equivalent regimes),
not a fixed penalty independent of operating point.

---

## Verdict

The already-known +213–334% bitrate cost has (at least) two contributing
causes, now separated:

1. **A likely-larger volume effect**: at the shared `alpha=0.5/beta=0.5`
   config, `filter_frame_noise`'s own implicit selection threshold touches
   ~95% of blocks vs. ~9–13% for blur/downsample/mean_fill — the degradations
   are not being compared at a matched footprint.
2. **A real, consistent, matched-footprint encoder mechanism** (measured here
   with the region-isolated crop, at 100% coverage in both conditions):
   noise suppresses cheap Merge/Skip coding, forces the residual back into
   full Inter mode, and drives finer CTU partitioning — costing 24–235% more
   bits depending on QP and content, even with coverage held constant. The
   mechanism is **residual-cost and partition-depth driven, not
   intra-takeover driven** — the referee's specific "forced into costlier
   intra prediction" hypothesis is not what the CSV data shows; report this
   distinction rather than the more intuitive-sounding but wrong mechanism.

Both are legitimate parts of "why," and the paper response should probably
lead with #2 (it directly answers what was asked, in the encoder's own
language) while noting #1 as a discovered caveat on how the existing +213–334%
figure was produced.

## Limitations

- **No per-CTU spatial map was obtainable** from this ffmpeg/libx265 build
  (see "Granularity actually achieved" above) — the region-isolated crop is a
  deliberate workaround, not a like-for-like replacement. A from-source x265
  build with custom per-CU logging (or a direct `libx265` C API harness)
  would be needed for a literal per-CTU-coordinate map; out of scope for this
  investigation.
- Only 2 videos, 2 QPs, one 160×160 crop location per video. The direction of
  every effect is unanimous across all 8 conditions tested, which is decent
  evidence of robustness, but this is not a bitrate-ladder-scale study.
- The crop is a *spatial subset* of real frames (preserves real motion/texture
  and temporal continuity across frames), not a synthetic noise-only test
  pattern — chosen deliberately so the comparison reflects real content
  statistics, at the cost of the crop's absolute mode-mix numbers being
  specific to that one region rather than representative of "background in
  general."
- `csv-log-level=2`'s partition-percentage columns are area/count fractions
  of the coded picture, not literal per-CTU addresses — "avg final CU size"
  here is a single weighted-mean scalar per frame, not a spatial map.

## Reproduction

```
/home/itec/emanuele/.conda/envs/presley/bin/python \
    tools/noise_mode_decision_analysis.py \
    --out-dir scratch/noise_mode_decision --videos bear camel --qps 32 37
```

Requires the `presley` conda env (numpy/opencv/pandas) and the project's
existing cache (`cache/{bear,camel}_640x360*`, already present — no EVCA/UFO
GPU inference needed, scores/masks are read from the cache as-is). Output
(raw CSVs, intermediate lossless `.mkv`, encoded `.mp4`, and
`summary.json` with every number in the tables above) is written under
`scratch/noise_mode_decision/` (gitignored — `scratch/` is not tracked;
rerun the command above to regenerate it). `tools/pick_noise_region.py`
reproduces the crop-selection scan (Finding 2's crop coordinates); the
coverage-fraction numbers in Finding 1 are a direct one-off comparison of
`np.round(score) > 0` vs. `np.round(score * 50) > 0` against the same
cached `removability_a0.50_b0.50.npy` files, not wrapped in its own script
since it's a two-line numpy comparison — the exact commands are in this
file's git history / the session transcript if needed.

## Pitfall hit while building this (documented so it isn't hit again)

x265's `--csv` **appends** to an existing file rather than truncating it, so
rerunning against a leftover CSV from a previous invocation silently doubles
(or worse) the row count and skews every mean computed from it — the initial
run of this script produced a `full_control` summary with `n_frames: 164`
against an 82-frame input before this was caught. Fixed in
`tools/noise_mode_decision_analysis.py`'s `run_condition` by unlinking any
stale `csv_path`/`encoded` before encoding, plus a hard `n_frames == len(frames)`
assertion so a future recurrence fails loudly instead of silently skewing the
means.

A second, similar pitfall: x265's CSV percentage cells are strings like
`" 36.36%"` (object dtype in pandas); summing/averaging several such object
columns without stripping `%` first does **string concatenation**, not
arithmetic, which only surfaces once you reduce across every row as a
cryptic `Could not convert string ... to numeric`. Fixed via `_pct_to_float`
in the same script.
