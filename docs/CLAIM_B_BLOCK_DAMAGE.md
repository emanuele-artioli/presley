# Claim (b): is damage-after-restoration where the variance is?

Companion to `docs/F1_ORACLE_BITS.md`, which did the same job for claim (a).
Design and bounds written **before** any run of this component existed.

## What is being closed

`NOTE(sec:implementation)` states the selection objective as a ratio,
`dBits / dDamage-after-restoration`, and observes that only the numerator was
ever modelled — which is the mechanism behind the alpha/beta null. The
supporting measurement for the denominator is a **p90–p10 spread of 6.2 dB
(realesrgan) and 8.2 dB (propainter) in per-64×64-superblock damage, within a
single run**, against the 0.03–0.05 dB that alpha/beta move FG-PSNR.

The number is real. It is not citable: `tools/mine_block_damage.py` computed it
**outside the runner** by joining 111 run/baseline pairs on disk, so it has no
`results/<hash>` and `NEXT(sec:implementation)` forbids it from any
reviewer-visible sentence. This re-measures the same quantity through
`presley-run`.

## Design

- **Component `probe_block_damage`.** It does not reimplement the pipeline — it
  dispatches to `run_presley_ai` and then measures. A probe that measured a
  reimplementation would be evidence about the reimplementation.
- **Baseline encoded inside the run**, pristine frames at the same fixed QP and
  codec. Looking one up in `results/` is exactly what made the original number
  unciteable; the run must be self-contained.
- **Grid: 64×64 superblocks**, shared with the mining tool via
  `presley.blockdamage` so the two cannot drift apart on geometry. Partial edge
  superblocks are kept with their true area (360 rows = 5 full SBs + a 40-row
  strip, ~11% of the frame, which must not be silently dropped).
- **MSE pooled, never PSNR.** PSNR is logarithmic; a PSNR-space average across
  blocks is not a quantity.
- **`delta_psnr(s) = psnr(baseline, s) − psnr(restored, s)`**, so what is
  measured is damage attributable to degrade→restore *over and above* what the
  codec already did at that QP.
- **Headline: the p90–p10 spread of `delta_psnr` over the degraded superblocks
  of one run.** Within-run dispersion, which is what the claim asserts. Spread
  across runs is a different (and weaker) statement and is not the headline.
- **An SB is "degraded" at >50% area, "untouched" at exactly 0%.** The gap is
  deliberate: a straddling SB is genuinely part-degraded and belongs in neither
  group.
- **Fixed QP throughout** (hard rule 1). The probe raises rather than falling
  back to a bitrate target for a codec it has no fixed-QP encode for.

### Runs

8: 4 videos (`bear`, `camel`, `bike-packing`, `dogs-jump`) × 2 restorers —
`realesrgan` on `downsample`, `propainter` on `freeze`. Every other key is
copied from the canonical existing `presley_ai` entry for that pairing (svtav1
QP 43 preset 8, 640×360, bs16, alpha/beta 0.5, shrink 0.25, fg_protect,
composite_output). Cost is dominated by ProPainter at ~49 min/run: ~3.5 h total.

Both restorers are needed because the claim quotes both, and they differ by 2 dB
in the mined number — reproducing only the smaller one would leave the wider
figure uncitable.

## Bounds, written before any number exists

Per run, on the degraded superblocks:

| quantity | plausible | reportable null | **alarm** |
|---|---|---|---|
| `spread_p90_p10` | 2–12 dB | < 1 dB — claim (b) is not reproduced | > 15 dB |
| median `delta_psnr` | 0–6 dB | ≤ 0 dB (restoration beat the pristine encode) | > 10 dB |
| `degraded_sb_fraction` | 0.05–0.95 | — | outside that: selection or mask is wrong |
| median `delta_psnr`, untouched SBs | 0–1 dB | — | > 1 dB, or untouched spread ≥ degraded spread |

Basis: the mined 6.2/8.2 dB sits mid-range in the first row; the codec-only
baseline bounds the second (damage over-and-above the codec cannot plausibly
exceed the codec's own contribution by 10 dB without something being broken);
the fourth is the contamination check — untouched superblocks should move only
through neighbour bleed in inter prediction, and if they move as much as the
degraded ones then the within-run spread is not attributable to selection and
the claim does not follow from it.

**A spread below 1 dB is a real outcome and gets reported as one.** It would say
the denominator's variance is a property of the pooled 111-pair mining rather
than of a single run, and claim (b) would have to be reworded or dropped rather
than quoted from a different source.

Out-of-range values are **alarms, not findings**: investigate strength-grid /
SB-grid misalignment (the single most likely bug — the probe raises on a shape
mismatch, but an aligned-yet-wrong grid would not raise), a diverging restorer
(now caught independently by the saturation invariant), and frame-count
mismatch between the restored output and the baseline encode, before any number
is written down.
