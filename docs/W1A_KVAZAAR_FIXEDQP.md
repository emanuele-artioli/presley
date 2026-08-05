# W1a — a fixed-QP kvazaar baseline for `tab:roi`

**Status:** bounds pre-registered 2026-08-03, before any code was written or any
run launched. Plan: `~/.claude/plans/after-wave-2b-reported-cosmic-sunbeam.md`.

## Why

`tab:roi` compares kvazaar **ROI arms encoded at fixed QP** against kvazaar
**baselines encoded VBR**. Kvazaar VBR overshoots its target by 30–45%
(`research-log/hard-rules.md`), and the table's headline saving is 24.9–40.2% —
the same band. The reported saving is therefore consistent with being nothing
but the baseline's overshoot. The claim cannot stand until the baseline is
encoded the same way the ROI arm is.

## Why no such baseline exists (it was not an oversight in the config)

`baselines.py` has a fixed-QP branch for x265 (`encode_video_x265_qp`) and for
SVT-AV1 (`encode_video_svtav1_qp`), **but none for kvazaar** — the kvazaar
branch only ever calls `encode_video_kvazaar(..., target_bitrate, ...)`, which
is `--bitrate`, i.e. VBR. `derive_rate_control` hard-codes `'vbr_1pass'` for
kvazaar for the same reason. So every kvazaar baseline in the corpus is VBR
because **the code could not produce anything else**. This is a missing code
path, not a mis-specified experiment.

## What "matched" has to mean here

`encode_with_roi_kvazaar` does not take a QP. It **binary-searches the base QP
whose actual bitrate is closest to `target_bitrate`**, then encodes with
`--qp <that> --roi <map>`. The matched baseline is therefore the identical
search and the identical encoder invocation **minus `--roi`**. Anything else
(e.g. picking a QP by hand) reintroduces a confound of exactly the kind this
work exists to remove.

## Pre-registered bounds — written before the runs

**The central prediction is that the published bit saving largely disappears,
and that this is the correct outcome rather than a failure.** If both arms
binary-search to the same target bitrate, both land near it by construction, so
a large rate delta is no longer *available* to be measured. The comparison
converts from "ROI saves bits" into "at matched rate, ROI buys foreground
quality" — which is the claim the table should have been making.

| quantity | plausible | alarm |
|---|---|---|
| \|Δbits\| ROI vs new fixed-QP baseline | **0–10%** | >25% — the search is not converging, or the arms differ in something beyond `--roi` |
| FG-PSNR delta (ROI − baseline) at matched rate | **+0.2 … +1.5 dB** | >2.0 dB, or < −0.5 dB |
| BG-PSNR delta (ROI − baseline) | **−0.2 … −1.5 dB** (ROI spends BG bits on FG; this is the mechanism, and its absence would be the alarm) | positive, or < −3 dB |
| Published "24.9–40.2% fewer bits" | **collapses toward 0** | survives above 20% — would mean the VBR overshoot explanation is wrong and something else is going on |

Three outcomes, all publishable:

1. **Savings vanish, FG gain survives** (expected). `tab:roi` becomes a
   quality-at-matched-rate claim. Stronger than today's, because it is finally
   calibrated against a like-for-like baseline.
2. **Savings shrink but stay non-zero.** Report the corrected figure.
3. **Savings survive.** The overshoot explanation is wrong; investigate before
   claiming anything, because it would contradict a measured hard rule.

**Bounds are not to be revised after reading the numbers**, only recorded as
fired with a stated reason.

## Scope

16 baseline points at 640×360 matching every existing kvazaar ROI arm:
`bear` ×4 (150k / 250k / 460.8k / 555k / 800k → the distinct targets),
`bmx-trees`, `drift-chicane` @460.8k, and `camel` / `dog` / `india` / `pigs` /
`tennis` @300k and @600k. CPU-only; no GPU restoration is involved. The
960×540 and 1280×720 `bear` points are deferred until the 640×360 result is in.

**Also in scope, same edit:** `evaluation.tex:~1168` discloses only
`india`/`tennis` as lacking kvazaar baselines. `camel`, `dog` and `pigs` ROI
runs are equally orphaned — 10 ROI runs across 5 videos, not 2.

---

# Result — 2026-08-03. Outcome 1: the saving was the baseline's overshoot.

**17/17 entries produced results; 0 `Error running experiment` lines; 17/17 have
metrics and empty `invariant_failures`** (verified against the DB, not against
the runner's exit code, which is 0 even when every entry fails).

| video | target | Δbits vs new fixed-QP base | ΔFG dB | ΔBG dB | published (vs VBR) |
|---|---|---|---|---|---|
| bear | 150k | **+6.3%** | +0.53 | −0.40 | −27.7% |
| bear | 250k | **+4.1%** | +0.57 | −0.53 | −40.2% |
| bear | 460.8k | −6.5% | +0.71 | −0.73 | −24.8% |
| bear | 555k | **+17.3%** | +1.34 | −0.22 | — |
| bear | 800k | −11.7% | +0.73 | −0.89 | −32.7% |
| bmx-trees | 460.8k | +7.3% | +0.24 | −0.04 | −11.7% |
| camel | 300k | +8.7% | +0.51 | −1.12 | — |
| camel | 600k | −2.7% | +0.58 | −1.28 | — |
| dog | 300k | −14.1% | +0.57 | −0.88 | — |
| dog | 600k | −3.8% | +1.00 | −0.62 | — |
| drift-chicane | 460.8k | −0.1% | +0.21 | −0.13 | +3.2% |
| india | 300k | +2.9% | +0.23 | −0.20 | — |
| india | 600k | +2.0% | +0.20 | −0.25 | — |
| pigs | 300k | −3.5% | +0.13 | −0.95 | — |
| pigs | 600k | −15.4% | +0.09 | −1.33 | — |
| tennis | 300k | −4.9% | **−0.45** | −0.25 | — |
| tennis | 600k | +6.9% | +0.14 | **+0.13** | — |

**Δbits median −0.1%** (range −15.4…+17.3). **ΔFG median +0.51 dB**, positive on
**16/17**. **ΔBG median −0.53 dB**, negative on **16/17**.

## What this does to `tab:roi`

**The rate claim is refuted.** Against a like-for-like fixed-QP baseline the
bitrate difference is centred on zero. The published "24.9–40.2% fewer bits" was
measuring the VBR baseline's 30–45% overshoot, exactly as the hard rule warned.
It must be removed, not softened.

**The quality claim survives and is now calibrated.** ROI buys **+0.51 dB median
foreground PSNR at essentially equal rate**, and pays for it with **−0.53 dB
background** — bits moving BG→FG, which is the mechanism the method claims. That
is a cleaner result than the one it replaces: it is a like-for-like comparison,
and it isolates the ROI map as the only difference between the arms.

The revised claim: *at matched bitrate, kvazaar ROI moves ~0.5 dB from the
background to the foreground.* No bitrate saving is claimed.

## Bounds — three fired, none silently

| bound | outcome |
|---|---|
| \|Δbits\| 0–10% plausible | **outside on 4/17** (+17.3, −11.7, −14.1, −15.4); none reached the >25% alarm |
| ΔFG +0.2…+1.5 dB | **outside on 1/17** — `tennis`@300k at −0.45 dB, inside the −0.5 alarm but the wrong side of plausible |
| ΔBG negative | **ALARM fired on 1/17** — `tennis`@600k at **+0.13 dB** |

**Δbits spread — revised with a reason, not dropped.** The band assumed both
arms land on the target. Both binary-search an **integer** base QP, and one QP
step is worth roughly 10–15% of bitrate at these rates, so ±15% is the
*granularity floor* of the search rather than a defect. The band should have
been ±1 QP step expressed in bitrate. Recorded as a mis-set bound.

**`tennis` fired both other bounds and is the one thing to investigate before
this is cited.** It is the only video where ROI loses foreground (−0.45 dB) and
the only one where background improves (+0.13 dB) — i.e. the mechanism runs
*backwards* there. Both deltas are below the 0.5 dB PSNR JND, so the honest
reading is that `tennis` shows **no ROI effect** rather than a reversed one; but
`tennis` is also the clip whose union-bbox foreground is 4% of frame while the
box covers 59%, so it is exactly the content where mask-driven methods have
misfired before. **Do not quote tennis as a counterexample without a masked
re-check.**

## Still outstanding

`evaluation.tex:~1168` discloses only `india`/`tennis` as lacking kvazaar
baselines. That sentence is now obsolete in a better way — **every** ROI arm at
640×360 has a fixed-QP baseline as of this run, including the previously
orphaned `camel`, `dog` and `pigs`. The disclosure should be replaced, not
merely corrected.
