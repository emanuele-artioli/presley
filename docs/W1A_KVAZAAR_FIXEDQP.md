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
