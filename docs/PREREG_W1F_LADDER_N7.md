# Pre-registration — extending the resolution ladder to break the n=5 floor

Written **before the runs exist**; the git log order is the check. Committed
2026-08-09.

## Why

The resolution ladder reports 5/5 clips saving bits at 720p and 1080p, which
cannot reach significance: at n=5 the smallest attainable exact two-tailed
p is 0.0625. The article therefore reports it as "consistent and underpowered,
not a win". It is n=5 rather than n=6 because `camel`'s runs above 360p trip
the saturation invariant and are not citable.

Adding clips is the only route to a claim that does not involve relaxing an
invariant to unblock our own campaign, which the project forbids.

## What is added, and why both

Every DAVIS sequence that is (a) natively 1920x1080, (b) has >= 82 frames with
annotations, and (c) is not already in the ladder:

| video | frames | native |
|---|---|---|
| `breakdance` | 84 | 1920x1080 |
| `bmx-bumps` | 90 | 1920x1080 |

**Both are added, and this is the point of writing it down first.** Two clips
qualify under a rule fixed before looking at any result, so both run. Picking
one and reporting it would be candidate shopping on an outcome we already know
the shape of. `scooter-board` and `dogs-scale` are excluded by the rule --
both are 3840x2160, so their 1080p rung would be a downscale of a different
factor than every other clip's, changing the reference each rung is scored
against.

Note what the corpus already says about these two, so it cannot be presented as
a surprise afterwards: `bmx-bumps` is one of the hardest sequences here (fast
camera motion; its sibling `bmx-trees` is the worst detector-mask failure at
-2.90 dB), and `breakdance` is one of five sequences with a large true
foreground. A reversal on either is a plausible outcome, not an anomaly.

## Recipe

Identical to the existing ladder, so the new cells pool with the old ones:
`presley_ai`, downsample + Real-ESRGAN, `restorer_params: {}`, alpha=beta=0.5,
shrink 0.25, `fg_protect`, `composite_output`, SVT-AV1 preset 8, QP
43/50/55/60, block size 8/16/24 at 360p/720p/1080p so every rung keeps an
80x45 grid. Baseline arm is pristine SVT-AV1 at the same QP.

`restorer_params: {}` is deliberate and matches the existing ladder rather than
the published 640x360 corpus, which passes `{denoise_strength: 1.0, tile: 400}`.
`analyze_ratematched_n13.py` whitelists both values for exactly this reason.

## Bounds, stated before the numbers

Primary quantity: FG-LPIPS BD-rate per clip per resolution, selective arm
against its own matched pristine baseline. Negative = fewer bits at equal
foreground quality.

| | plausible best | plausible worst | basis |
|---|---|---|---|
| new-clip BD-rate, 720p/1080p | -15% | +30% | published rungs run -4.6% to -5.8% median; `dog`/`pigs` reverse by +27%/+18% elsewhere in the article |
| new-clip BD-rate, 360p | -20% | +30% | 360p already splits 3/3 |
| sign count at 720p | 7/7 | 5/7 | 5/5 published, two new clips either way |
| exact two-tailed p at n=7 | 0.0156 | 0.45 | 7/7 vs 5/7 |

**Alarms — investigate before reporting, do not fold into a median:**

1. Any |BD-rate| > 50% on a new clip. Nothing in this corpus moves that far on
   this axis; that magnitude has previously meant a rate-control or pairing
   fault, not an effect.
2. A new clip whose 360p and 1080p results have the same sign *and* a magnitude
   ratio above 5x. The rungs are confounded with bits-per-pixel already; a
   ratio that large suggests the confound, not resolution.
3. Any run with a non-empty `invariant_failures`, in particular the saturation
   invariant that made `camel` uncitable. If a new clip trips it above 360p,
   that is the second instance of a pattern and is worth more than the ladder.
4. Baseline bits-per-pixel at the new clips falling outside 0.03--0.15 at
   QP 43, the range the existing six occupy. Outside it the clip is not in the
   same operating regime and its rungs are not poolable.

**What will be reported either way.** The sign count and exact two-tailed p at
whatever n the citable runs support, with the bits-per-pixel confound restated.
If the result stays underpowered, it stays described as underpowered. Adding
clips until a p-value crosses a threshold is not the design here: these two are
every clip that qualifies, and the analysis runs once.

## After the run, before believing anything

The ladder shares rungs with published analyses, and new runs have broken one
before (`analyze_ratematched_n13.py`'s arm selector went ambiguous when the
first ladder introduced a second `restorer_params` value). Re-run both and
check they still reproduce their published numbers:

```bash
python tools/analyze_resolution_ladder.py --data-root /home/itec/emanuele/presley
python tools/analyze_ratematched_n13.py  --data-root /home/itec/emanuele/presley   # must still give -51.4%
```
