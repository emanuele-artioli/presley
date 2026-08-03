# Wave 2C — efficiency, and the real-time question

**Status:** complete 2026-08-03. No GPU time spent, read-only against
`results/`. Wave 2C of `docs/PLAN_OPERATING_MAP.md`, building on
`docs/WAVE1_OPERATING_MAP.md`.

Reproduce:

```bash
python tools/analyze_throughput.py --db results/presley.db --json throughput.json
```

Tests: `tests/test_analyze_throughput.py` (26 cases, fast tier).

**Headline: the alarm closed, and closing it overturned the premise it was
attached to.** The plan's throughput table said the baselines are "not
real-time at all above 360p". That table was built from medians pooled across
acquisition batches. Once the batches are separated, svtav1 preset 8 encodes
comfortably faster than real time at every resolution in the corpus, including
1080p. **Encoding was never the bottleneck; restoration is, by an order of
magnitude, and that was already Wave 1's finding.**

---

## 1. The alarm, and how it closed

**Fired:** median encode throughput said svtav1 baseline runs at **3.2 fps at
1280×720** but **28.9 fps at 1920×1080**. Faster at higher resolution is
impossible.

**Bounds stated before looking.** For svtav1 preset 8 (a fast preset) on an
~80-frame clip on server CPU: 720p should land in 30–300 fps, 1080p in 15–150
fps. Note what this implies — **3.2 fps at 720p is already outside the lower
bound, while 28.9 fps at 1080p is inside it.** The suspect measurement was the
720p one, not the 1080p one. The alarm had been pointed at the wrong number.

**Cause: the 720p cell is a mixture of two acquisition batches, and its
"median" falls in a gap that no run occupies.**

| batch | when | n | QP range | fps range | median |
|---|---|---|---|---|---|
| 0 | 2026-07-06 → 07-15 | 8 | 30–37 | 0.68 – 1.24 | 1.00 |
| 1 | 2026-07-22 15:18 | 8 | 43–62 | 23.78 – 37.37 | 34.51 |

The two ranges do not overlap. The pooled median of 3.17 fps sits between 1.24
and 23.78 — a value no svtav1 720p encode in this corpus ever achieved. It is
an artifact of taking the centre of a bimodal mixture.

**Two mechanisms, both confounded with resolution by accident of scheduling:**

- **QP.** The 720p cell's slow batch was encoded at QP 30–37; the 1080p cell
  contains only QP 43–62. QP moves encode time in this corpus by more than
  resolution does — at 640×360, svtav1's median is 36.3 fps at QP 43 and 81.9
  fps at QP 58. The cells were never comparable.
- **Contention.** Within-batch spread on this shared box reaches IQR/median
  1.71 (x265 640×360). Contention only ever slows a run, so it inflates spread
  downward and corrupts any pooled centre.

**The test that closes it — hold codec, video set and QP set fixed:**
svtav1, videos `bear`+`camel`, QP ∈ {43, 50, 58, 62}:

| resolution | n | median fps | range |
|---|---|---|---|
| 640×360 | 5 | 85.0 | 41.9 – 118.9 |
| 1280×720 | 8 | 34.5 | 23.8 – 37.4 |
| 1920×1080 | 8 | 31.4 | 16.6 – 45.4 |

**Monotone decreasing in resolution. The impossibility is gone.** The residual
720p/1080p closeness (34.5 vs 31.4, overlapping ranges) is within run-to-run
noise on a contended box and is not claimed as a difference.

**Verdict: alarm closed by diagnosis, not by discarding data.** The 720p *cell
median* is thrown out with a stated reason — it pools disjoint batches. The
per-batch rows are kept and are publishable. `judge_cell()` in the tool now
refuses any cell whose batches are disjoint in fps, and
`test_disjoint_batches_are_not_measurable` pins that refusal on the real
numbers.

**A second correction while here.** The plan's table quoted `kvazaar 640×360
12.6 fps` and (via x264) other rows. **Every kvazaar and x264 baseline in the
corpus is VBR** (`vbr_1pass` / `vbr_2pass`). Under this project's fixed-QP-only
rule those runs are not evidence, and they are dropped from the table below
rather than caveated. There is presently **no fixed-QP kvazaar or x264
throughput measurement at all.**

---

## 2. Throughput per configuration, with conditions

**Measurement conditions, applying to every row.** Shared multi-user GPU/CPU
server, no isolation, no CPU pinning, no repeat trials — each run is measured
once, as part of an experiment, not as a benchmark. Clips are 20–162 frames at
a 24 fps source rate. Times are the pipeline's own logged
`encoding_time_seconds` / `restoration_time_seconds`, which include per-run
process and I/O overhead; at these clip lengths that overhead is not
negligible, so all figures are **lower bounds on the software's capability**.
Fixed-QP only (`cqp`/`crf`), `invariant_failures` empty. Medians, never means.
"Batches" are runs within 6 h of each other; the corpus's batches are days
apart.

### Encoding

Rows are per acquisition batch, because a pooled cell median is a mixture.

| codec | resolution | preset | batch | n | QP range | median fps | range | vs 24 fps |
|---|---|---|---|---|---|---|---|---|
| svtav1 | 640×360 | 8 | 0 | 4 | 43–61 | 65.7 | 41.9–85.0 | 2.7× |
| svtav1 | 640×360 | 8 | 1 | 4 | 42–62 | 88.8 | 9.9–118.9 | 3.7× |
| svtav1 | 640×360 | 8 | 2 | 16 | 43–62 | 68.9 | 26.0–114.7 | 2.9× |
| svtav1 | 640×360 | 8 | 3 | 7 | 43 | 33.2 | 26.7–39.1 | 1.4× |
| svtav1 | 640×360 | 8 | 4 | 10 | 40–63 | 30.8 | 16.9–44.2 | 1.3× |
| svtav1 | 1280×720 | 8 | 0 | 8 | 30–37 | 1.00 | 0.68–1.24 | 0.04× |
| svtav1 | 1280×720 | 8 | 1 | 8 | 43–62 | 34.5 | 23.8–37.4 | 1.4× |
| svtav1 | 1920×1080 | 8 | 0 | 8 | 43–62 | 31.4 | 16.6–45.4 | 1.3× |
| svtav1 | 1920×1080 | 8 | 1 | 8 | 43–62 | 28.8 | 23.3–36.0 | 1.2× |
| x265 | 640×360 | medium | 6 | 14 | 42–47 | 54.0 | 38.2–92.0 | 2.3× |
| x265 | 640×360 | medium | 4 | 36 | 32–37 | 37.6 | 2.5–106.6 | 1.6× |
| x265 | 640×360 | medium | 5 | 14 | 32–37 | 12.6 | 1.2–38.1 | **0.5×** |

x265 640×360's remaining batches (0–3, n=4/3/12/9) span 18.3–28.7 fps at QP
30–38. **The whole x265 640×360 cell is flagged `wide`** — IQR/median 1.71
within a single batch — so its numbers describe the machine's load as much as
the codec. `x265 640×480` and the transposed `x265 360×640` cell each hold
n=2 per batch and are reported as **not measurable**.

**What is safe to say about encoding:** svtav1 preset 8 clears 24 fps in
*every* clean batch at every resolution measured, 360p through 1080p. x265
medium at 360p clears it in most batches but not all, and its spread is
contention-dominated. The plan's claim that the baselines are "not real-time
above 360p" **does not survive the batch split** and should not be repeated.

### Restoration, 640×360

| fill | n | videos | median fps | range | IQR/med | restoration share of wall clock |
|---|---|---|---|---|---|---|
| realesrgan | 108 | 20 | 4.85 | 1.54–6.59 | 0.35 | 73% |
| unsharp | 12 | 4 | 4.57 | 2.32–6.78 | 0.30 | 59% |
| telea | 26 | 5 | 3.75 | 0.43–12.51 | 0.99 | 81% |
| nafnet | 50 | 9 | 3.14 | 0.51–5.80 | 0.31 | 76% |
| bsrgan | 2 | 2 | 2.92 | 2.38–3.45 | 0.18 | 81% |
| stream_diffvsr | 2 | 2 | 1.67 | 1.61–1.72 | 0.03 | 91% |
| e2fgvi | 16 | 4 | 1.63 | 1.09–2.03 | 0.23 | 92% |
| real_hat_gan | 2 | 2 | 1.14 | 1.14–1.15 | 0.00 | 94% |
| propainter | 144 | 16 | 1.14 | 0.02–4.16 | 1.26 | 95% |
| instantir | 4 | 2 | 0.17 | 0.11–0.55 | 1.19 | 97% |

`bsrgan`, `stream_diffvsr`, `real_hat_gan` and `instantir` rest on n=2–4 and
are indicative only. `telea` and `propainter` carry IQR/median ≈ 1.0–1.3 and
are contention-affected.

These are computed over **all** fixed-QP 360p runs; Wave 1's cost section
computed them over the subset that entered the map. The orderings agree; the
levels differ slightly (propainter 1.14 here vs 0.78 there, realesrgan 4.85 vs
4.48). Neither supersedes the other — they are different populations, and the
scope should be stated wherever either is quoted.

**Restoration is 59–97% of wall clock**, confirming Wave 1's 66–98%. End-to-end
at 360p the median pipeline runs at **2.05 fps** (n=366) — 3.45 fps with
realesrgan, 1.02 fps with propainter.

### The bottom line on real time

**No configuration in this corpus runs the full pipeline in real time**, and
the gap is entirely on the restoration side: the fastest generative fill at
360p reaches 0.20× real time while the encoder in front of it reaches 2.7–3.7×.
Saying that plainly is the useful statement. What we may **not** say any more
is that the baselines fail to be real-time — corrected, they clear it.

---

## 3. Cost attached to each map recommendation

Cost is the third axis alongside BG-LPIPS and bits. **244 of 420 arms are
Pareto-dominated inside their own cell** — another arm in the same cell is no
worse on BG-LPIPS *and* no slower (reproduced exactly from Wave 1).

For each decided cell, the tool now reports the winner's throughput and the
fastest **quality-tied** alternative (within one BG-LPIPS JND of the winner —
i.e. a swap a viewer could not see).

| objective | decided cells | cells with a faster quality-tied arm | max speedup |
|---|---|---|---|
| quality-first | 33 | **1** | 1.1× |
| rate-first | 50 | **14** | **9.8×** |

**This is the answer to "which recommendations change once cost is visible".**

**Quality-first barely moves.** Its winner is `downsample+realesrgan` in 18 of
19 separable cells, and realesrgan is already the fastest generative fill in
the corpus (4.85 fps). Optimising for quality here happens to also optimise for
cost. Only one cell offers a faster tie, and at 1.1× it is noise.

**Rate-first moves a lot.** Its winner is `blackout+propainter` in 27 of 36
separable cells, and propainter is the *slowest* generative fill (1.14 fps,
4.3× slower than realesrgan; 5.7× on Wave 1's in-map subset). In 14 of 50
decided cells a perceptually indistinguishable arm runs faster:

| operating point | rate-first winner | fps | quality-tied alternative | fps | speedup |
|---|---|---|---|---|---|
| drift-turn / svtav1 / QP43 / 360p | blur+nafnet | 0.52 | downsample+realesrgan | 5.09 | **9.8×** |
| bear / svtav1 / QP58 / 360p | blackout+propainter | 1.46 | freeze+telea | 8.78 | **6.0×** |
| bear / svtav1 / QP43 / 360p | downsample+realesrgan | 1.77 | downsample+realesrgan | 5.36 | 3.0× |
| mosev2/8i1uo3x9 / x265 / QP42 / 360p | blackout+propainter | 2.21 | downsample+realesrgan | 4.90 | 2.2× |
| camel / x265 / QP34 / 360p | blackout+propainter | 1.36 | blackout+propainter | 2.26 | 1.7× |
| color-run / svtav1 / QP43 / 360p | blur+nafnet | 3.46 | downsample+realesrgan | 5.39 | 1.6× |
| drift-straight / svtav1 / QP43 / 360p | downsample+realesrgan | 4.34 | downsample+realesrgan | 6.10 | 1.4× |

(remaining 7 are 1.1–1.2×)

Two of these rows name the **same label** on both sides (`downsample+realesrgan`
→ `downsample+realesrgan`). That is not a swap — it is two runs of the same arm
at different block sizes or shrink amounts, and it measures the corpus's
run-to-run timing spread rather than a design choice. **The genuine
cost-motivated swaps are the ones that change the label**, of which the
`drift-turn` (9.8×) and `bear`/QP58 (6.0×) rows are the strongest.

**The recommendation that changes:** Wave 1 reported that the two objectives
almost never agree (1 of 19 doubly-separable cells). Cost breaks the tie in
favour of quality-first. Rate-first's preferred arm is the corpus's slowest
generative fill and is, in 28% of its decided cells, replaceable by an
invisible-to-the-viewer alternative that runs up to 9.8× faster. **A deployer
who states rate-first as the objective and then reads only the bits column is
paying an order of magnitude in throughput for a saving the quality metric
cannot see.** Rate-first recommendations should be quoted with their fps
attached, or not quoted alone.

---

## 4. What a real-time setup would require

A reader's hardware and settings may put them on the other side of this line,
so the requirement is stated as a budget rather than a verdict.

**The budget.** At 24 fps the whole pipeline has 41.7 ms per frame. Measured at
640×360 on this host: encode 11–15 ms/frame (svtav1 preset 8), restoration
206 ms/frame (realesrgan, the fastest generative fill). Restoration overruns the
entire frame budget roughly **5×** on its own.

**Encoder.** Not the constraint. svtav1 preset 8 already delivers 1.2–3.7×
real time at 360p–1080p here. A deployer needing more headroom moves to preset
10–12 (svtav1 presets trade compression for speed monotonically), which is
untested in this corpus and would need its own BD-rate check. x265 `medium`
is marginal at 360p and should be moved to `veryfast`/`ultrafast` or replaced.

**Resolution.** Restoration cost scales with pixel count and is applied to the
**whole frame** by design (`upscale_realesrgan_adaptive` — see Wave 3(a) in the
plan), so it does not shrink with the degraded fraction. 360p is already the
corpus's smallest resolution; there is no resolution headroom left below it
that keeps the content meaningful. **Resolution is not the lever.**

**Hardware.** The measured 4.85 fps for realesrgan is on a *shared, contended*
host with no isolation, and IQR/median of 0.35 says contention is material. A
dedicated GPU would plausibly recover a factor of ~2 — **not the ~5× needed**.

**Therefore: real time requires a cheaper restorer, not a better machine.**
Concretely, one of —

1. **Restrict restoration to the degraded region** (Wave 3(a)). Cost is
   presently independent of `shrink_amount`, so a block-local or dilated
   variant should cut cost roughly in proportion to the untouched area. This is
   the largest single lever and it is unmeasured. It trades away the neighbour
   context the current design exists to preserve, so it must be measured as a
   speed/quality tradeoff, not assumed free.
2. **Sweep the parameters that have never been varied** (Wave 3(b)) —
   `denoise_strength`, NAFNet `width`, lighter model variants, tiling.
3. **Accept the fastest fill and state it.** `unsharp` (4.57 fps, non-generative)
   and `telea` (3.75 fps) are already within ~5× of the budget and would benefit
   most from (1).

Only after (1) and (2) are exhausted does weight compression or a new model
become justified — which is exactly the ordering Wave 3 of the plan sets out.

---

## Caveats

- **No repeat trials.** Every timing is a single measurement taken during an
  experiment, not a benchmark. n counts distinct runs, not repetitions of one
  configuration.
- **Contention is unquantified.** It can only slow runs, so every median here
  is a lower bound; the *maxima* are closer to the machine's capability.
- **Per-run overhead is included** and is material at 20–162-frame clips.
- **No fixed-QP kvazaar or x264 timing exists.** Both codecs' baselines are
  VBR-only in this corpus.
- **Nothing here is a significance claim.** These are descriptive medians with
  their n and spread stated, as Wave 1's map is.
