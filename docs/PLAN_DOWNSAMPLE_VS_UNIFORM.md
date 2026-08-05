# Selective vs. uniform downsampling at matched rate

**Closes `HOLE(sec:downsample-vs-uniform)`** (`sections/background.tex`), which is
TOMM referee 2 item 6 sub-point 2: *why is importance-selected per-block
downsampling different from / better than the whole-frame downscale +
client-side super-resolution that large streaming services already deploy?*
Today `background.tex` §2.1 and the end of `sec:insight` answer that
**architecturally only** and explicitly decline a superiority claim.

**Bounds pre-registered 2026-08-05, before any uniform-arm run exists.**
Committed before launch, per the bound-before-believing rule and to the standard
`docs/RATEMATCHED_BREADTH.md` set. The two-sided requirement below is a direct
consequence of `NOTE(tab:conditioned-breadth)`: *"Do not re-pre-register a
one-sided band on a quantity a HOLE exists to generalize."*

---

## 1. The design: one variable, and it is the mask

Two arms, identical in **codec, rate control, QP, resolution, block size,
degradation operator, per-block degradation strength, restorer, compositing and
metrics**. The only thing that differs is *which blocks* the degradation lands on.

| | Selective arm | Uniform arm |
|---|---|---|
| `shrink_amount` | `0.25` | `80` |
| `fg_protect` | `true` | `false` |
| resulting footprint | top-25% removable blocks, UFO foreground hard-excluded | **100% of blocks** |
| per-block strength | level 1 (2x down, `INTER_AREA` -> `INTER_LINEAR`) | level 1, **identical** |

The uniform arm is therefore a whole-frame 2x downscale-then-upscale, run through
*our* encoder, *our* Real-ESRGAN, *our* masked FG/BG metrics. This is deliberately
**not** a reimplementation of a Netflix/YouTube pipeline; it is the controlled A/B
that isolates spatial selectivity, the same logic `tab:mask-morph` uses ("fair A/B
is morph vs UFO none, not gt").

### 1.1 Scope limit that must travel with the result

The uniform arm transmits **full-resolution but low-passed** frames, not a genuine
lower-rendition bitstream with fewer coded pixels. A real ABR rendition also saves
on pixel count and on decoder-side scaling. So this experiment answers *"does
spatial selectivity beat spatial uniformity, transport held identical"* and
**not** *"does PRESLEY beat Netflix's ladder"*. Any text that lands from this must
say so. Whatever the outcome, no claim about deployed commercial systems follows.

## 2. Code verification — no new code is needed (checked, not assumed)

Verified by reading `src/presley/degradation.py` and
`src/presley/components/presley_ai.py` and by executing the functions:

1. `select_removal_mask_global` (`degradation.py:297`) computes
   `per_row = int(amount * num_blocks_x) if amount < 1.0 else int(amount)`, then
   `k = min(per_row * num_blocks_y, num_blocks_y * num_blocks_x)`. At 640x360 with
   `block_size: 8` the grid is exactly 45 x 80 (no padding), so `shrink_amount: 80`
   gives `k = 3600 = 45*80`. **Executed: mask sum 3600 / 3600 blocks.**
2. `fg_protect: false` leaves `fg_block_masks = None` (`presley_ai.py:232`), hence
   `excl = None` at `:258`, hence no `exclude` argument and no `k` cap.
   **Executed: with a synthetic FG mask covering 25% of blocks and
   `fg_protect: true`, `k` is capped at 2700 / 3600 — i.e. `fg_protect: true`
   would silently make the "uniform" arm not uniform. It is required to be false.**
3. The mask that reaches the pixel operation is the same object: `sel` is passed to
   `filter_frame_downsample` at `:269`, and `_apply_sel_to_map` floors every
   selected block to level >= 1. **Executed: the returned strength map for the
   uniform config is `unique == [1]`, coverage 1.0, shape (45, 80)** — every block
   degraded, no foreground survivor.
4. **`downsample_uniform_level` is NOT used.** It raises unless `levels > 1`, and
   with `levels: 1` (the historical default) a full-coverage `sel` already yields a
   uniform level-1 map. Executed check: `filter_frame_downsample(..., sel=all_true)`
   and `filter_frame_downsample(..., sel=all_true, levels=2, uniform_level=1)`
   return **byte-identical pixels and byte-identical maps** (`block_size // 2**1 == 4
   == int(block_size * 0.5)`). Using `levels`/`uniform_level` would therefore add
   two knobs, change the hash, and buy nothing. The uniform arm adds **no new key**
   to the selective recipe — it changes two existing values.
5. Consequence of (4): the selective arm's degradation strength is *pixel-identical*
   to the uniform arm's on every block both degrade. Executed: strength values on
   the overlap are `{1}` in both. Selectivity really is the only variable.

Because `mask_source` only enters through the removability score, and a full-coverage
`sel` makes `round(score)` irrelevant (every block is floored to level 1), the
uniform arm is **independent of the importance model entirely** — which is exactly
what "uniform" must mean.

## 3. Recipe

Copied from the configs stored in `results/<hash>/result.json` for the cited src
hashes of `CLAIM(tab:conditioned)`, `CLAIM(tab:priced-trade)` and
`CLAIM(tab:conditioned-breadth)` — read from the result files, **not** inferred
from the tables' prose. All 24 selective hashes were re-hashed with
`runner.compute_experiment_hash` and reproduce their own directory names, so the
stored config is the yaml entry.

```
component: presley_ai      degradation: downsample     restorer: realesrgan
width: 640  height: 360    block_size: 8               alpha/beta: 0.5
codec: svtav1              codec_params: {preset: "8", qp: <rung>}
target_bitrate: 0          composite_output: true
restorer_params: {denoise_strength: 1.0, tile: 400}
```

`rate_control` on the existing runs reads `crf` (SVT-AV1 fixed QP). **Fixed
QP/CRF throughout, never VBR** (hard rule 1).

### 3.1 Videos (n = 6) and rungs

Chosen as *the videos that already have a complete 4-rung selective
downsample+Real-ESRGAN curve at this exact recipe with empty
`invariant_failures`*, plus matching `svtav1` 640x360 baselines at the same QPs.
That set is exactly six, which is also the `n >= 6` floor hard rule 2b requires
before any significance wording. Rungs are recalibrated per video, the established
practice of `CLAIM(tab:av1)` / `tab:av1-breadth`.

| video | rungs | selective hashes (exist, reused) |
|---|---|---|
| bear   | 43, 51, 58, 61 | `ceac3559f8af0c3f`, `e2cb6bed165d69b1`, `660bfa8e58ad4dc0`, `aa00beae3eca0b00` |
| camel  | 42, 50, 58, 62 | `a1dc7f1557c09867`, `c29c94b5c290f208`, `fde83c22fc01c85c`, `b06092cca2df6726` |
| dog    | 43, 50, 58, 62 | `dc189c206da6cd4d`, `36d973b3d6662b9b`, `c85b79fa26560bc3`, `5d046a46384dd4de` |
| india  | 43, 50, 58, 62 | `474fbb90d358caf0`, `a1fab6d2e641f27a`, `93657ec7fe2751d9`, `33c8f6d4dfd857e7` |
| pigs   | 43, 50, 58, 62 | `340195a69b34b413`, `9cdf895adbad37bd`, `b23228908f7d34a7`, `065853196970874b` |
| tennis | 43, 50, 58, 62 | `6d169af7b8a90f24`, `69f428bd0178726a`, `8b2b11e91badd1e8`, `1ec9cbffcdbf0988` |

**So only the uniform arm is new: 24 runs, not 48.** The selective arm is reused
verbatim, which is both cheaper and stronger — it is literally the arm the paper
already cites, not a re-run of it.

### 3.2 Analysis

- Rate axis: `actual_bitrate_bps` (never `target_bitrate`). Cross-check against
  `transmitted_size_bytes`; `tab:conditioned-breadth` found the two give identical
  BD-rates, and a divergence here is itself an alarm.
- BD-rate/BD-quality via `scripts/bd_rate.py` (4 points/curve, report
  `overlap_fraction`; a curve pair with low overlap is not interpretable).
- FG claims **only** from `foreground.lpips_mean`, `foreground.dists_fg`, and
  `foreground.psnr_mean`. FG-VMAF and FG-FVMD are banned union-bbox artefacts.
- JND gating via `presley-compare`; suite verdict via
  `presley-compare --suite` (two-tailed exact sign test, Holm-corrected). n=6
  floors at p=0.031 two-tailed, so 6/6 unanimity is the only outcome that can
  clear alpha uncorrected.
- Any run with non-empty `invariant_failures` is dropped and the whole comparison
  is uncitable until it is explained.

## 4. Pre-registered bounds

Sign convention: **BD-rate of the SELECTIVE arm with the UNIFORM arm as anchor.
Negative = selective needs fewer bits for the same quality = selective wins.**
Bands are per-video. Basis for the corpus-scale calibration: BD-rates of this
family observed to date span roughly -60% to +80%
(`tab:breadth-ratematched` -51.4%, `tab:conditioned-breadth` +25.3/+34.3%,
`tab:priced-trade` -13.8/-16.1% FG and +27.5/+80.9% BG-LPIPS).

### 4.1 BD-rate (FG), from `foreground.lpips_mean` and `foreground.psnr_mean`

- **Plausible best case for selective: -60%.** Basis: the selective arm never
  touches a foreground block, so its FG is degraded by the codec alone; the uniform
  arm puts a 2x downscale + Real-ESRGAN pass on top of that. -51.4%
  (`tab:breadth-ratematched`) is the largest gap a purely spatial change has
  produced in this corpus, so a little beyond it is the optimistic edge.
- **Plausible worst case for selective: +25%.** Basis: at *matched rate* the
  uniform arm can afford a materially lower QP, because a fully low-passed frame
  codes far cheaper at the same QP — and a lower QP improves the foreground too.
  `tab:conditioned-breadth` measured exactly this class of inversion at +25.3/+34.3%
  FG BD-rate when the assumption under test was generalized to new content.
- **Alarm outside [-80%, +50%].**

### 4.2 BD-rate (BG), from `background.lpips_mean`

- **Plausible best case for selective: -20%.** Basis: selective spends its
  degradation budget on the *most-removable* 25% of blocks, so per bit freed it
  should cost less BG quality than degrading everything indiscriminately.
- **Plausible worst case for selective: +70%.** Basis: the uniform arm is strictly
  more aggressive on background — it degrades 100% of BG blocks versus selective's
  <=25% — so it frees far more bits, and Real-ESRGAN restores low-passed content
  well. `tab:priced-trade` already measured +80.9% BG-LPIPS BD-rate against a
  less aggressive comparator, so a large positive value here is ordinary, not
  anomalous.
- **Alarm outside [-50%, +120%].**

### 4.3 FG-LPIPS, absolute delta at matched QP (uniform minus selective)

Positive = uniform's foreground is worse. Selective-arm FG-LPIPS on these six
videos currently spans 0.085 (tennis qp43) to 0.403 (tennis qp62), so the metric
has real headroom in both directions at every rung.

- **Plausible best case for selective: +0.15.** Basis: LPIPS JND is 0.05; a 2x
  downscale+SR of the whole foreground is a visible operation, so ~3x JND is the
  optimistic edge.
- **Plausible worst case for selective: -0.02.** Basis: at *matched QP* the uniform
  arm has no lower-QP advantage, so it should not beat selective on FG; a small
  negative is still reachable through Real-ESRGAN sharpening a low-passed
  foreground more favourably than the codec's own ringing.
- **Alarm if |delta| > 0.25** — that magnitude would mean the uniform arm annihilated
  the foreground, and the first thing to check is whether `composite_output` or the
  strength map behaved as section 2 predicts, not the science.
- **Alarm if delta < -0.05** — uniform materially *better* on FG at matched QP
  contradicts the mechanism; check `fg_protect` wiring on the selective arm before
  believing it.

### 4.4 Mechanism sanity gate (checked before any metric is read)

The uniform arm's degraded footprint must be **100% of blocks**. If a spot check of
the emitted strength map on any uniform run shows coverage < 1.0, the arm is not
uniform and the whole comparison is void regardless of how the numbers look. This
gate exists because a config that *looks* uniform but silently still protects
foreground would produce a quietly invalid result that no metric would flag.

### 4.5 Bits at fixed QP

The uniform arm must encode to **fewer** bits than the selective arm at the same
QP (it low-passes strictly more area). Expected range **-15% to -60%**; a uniform
arm that costs *more* bits than selective at fixed QP is an alarm on the encoder
path, not a finding.

## 5. Both interpretations, pre-registered

Written now so neither outcome can be rationalized afterwards.

**A. Selective wins** (FG BD-rate clearly negative, BG BD-rate not catastrophically
positive, direction unanimous across 6/6): lands as a **positive result answering
referee 6.2 directly**. `background.tex` §2.1 and the close of `sec:insight` gain
the matched-rate comparison, `HOLE(sec:downsample-vs-uniform)` is retired to a
`CLAIM`, and the architectural-only hedge is replaced by a measured statement —
carrying the §1.1 scope limit verbatim.

**B. Uniform wins, or the arms tie** (BG and/or FG BD-rate positive, or within
JND with no unanimous direction): lands as a **scoped limitation in
`sec:insight`**. The wording is that spatial selectivity buys foreground fidelity
at a rate cost, and that at matched rate uniform downscaling plus super-resolution
is competitive on this corpus — with the architectural distinction (per-block
strength travelling in a side channel, conditioned restoration) retained as the
contribution it already is. The `HOLE` still closes; it closes negatively.

**Uniform winning is a live possibility, not a formality.** Three reasons:

1. At matched rate the uniform arm can spend a much lower QP for the same bits,
   and QP is the strongest lever in the whole pipeline.
2. This corpus has just produced **two independent non-replications on exactly
   this class of comparison** — `NOTE(tab:av1-breadth)` (the starved win inverts on
   dog/pigs) and `NOTE(tab:conditioned-breadth)` (same reversal, different
   component). Both were pre-registered one-sided and both bounds fired.
3. `tab:av1` already shows this class of approach losing at comfortable bitrates,
   which is uniform downscaling's home regime — and two of our six rungs (qp 42/43)
   are the least starved.

No result in either direction is citable while `invariant_failures` is non-empty,
and neither band above may be re-fitted after the numbers are seen. If a bound
fires, it is recorded as fired and investigated — implementation, then evaluation,
then data — before anything is written into the paper.

## 6. Cost

24 new runs. Existing selective runs at this recipe take 23-37 s wall
(`total_time_seconds`: encode 7-13 s, restore 16-24 s). Uniform runs restore a
larger footprint, so budget ~1.5x. Estimate: **~20-30 min of run time, ~1.5-2.5 h
including evaluation and the `--backfill-lpips` pass.**

Operational notes for whoever launches this:

- `presley-run` exits 0 even when every experiment fails. Verify the result count
  against the entry count with `grep -c 'Error running experiment' <log>`.
- Region LPIPS is missing from fresh runs; run
  `presley-evaluate results/ --backfill-lpips` afterwards. **Never two backfills
  concurrently.**
- Launch detached (background job or the `experiment-runner` subagent), never
  attached to an SSH session.
