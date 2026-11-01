---
paths:
  - "src/**"
  - "scripts/**"
  - "config/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Evaluation methodology — every experiment has a comparison target' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

## Evaluation methodology — every experiment has a comparison target

PRESLEY has **two co-equal goals**. Every experiment tests one of them, and a
result is only complete when it says something about both:

- **Goal 1 — bit relocation.** Degradation moves encoding bits **BG→FG**, so at
  the same bitrate FG is *better*, respecting the chain
  `baseline < roi < elvis < presley_ai` (elvis and presley_ai may legitimately
  **tie** — see the FG-flatness finding). Lower BG quality is an *accepted cost*.
  **Metric:** FG-PSNR/FG-LPIPS at matched *actual* bitrate; BD-rate for
  paper-grade claims. Expected signature: FG ↑, BG ↓.
- **Goal 2 — generative restoration.** The client-side model restores the BG as
  close as possible to the **original**, without hurting FG; ideally exceeding
  original BG and/or FG. **Metric (perceptual primary): BG-LPIPS / BG-DISTS of
  the restored output vs the ORIGINAL**, compared against the pristine
  baseline's BG at matched bitrate. **BG-PSNR is reported alongside but is never
  the verdict** — `mean_fill` scores the *highest* BG-PSNR while being
  perceptually the *worst* (flat DC blocks are mathematically "closer" than
  hallucinated detail), so a PSNR-primary Goal 2 rewards a fill for **not**
  hallucinating, i.e. punishes the generative model for doing its job. The
  restoration *gain* (`metrics.background` − `metrics.transmitted.background`)
  is the mechanism; the **headline is restored-vs-original**, not
  restored-vs-degraded.

Goal 1 is not evidence for Goal 2 or vice versa. A method can free bits and
still fail to restore (that is the current standing — see the reports).

### ⛔ Hard rule: degradation experiments MUST use fixed-QP/CRF

Under **VBR the encoder spends the bitrate target regardless of source
complexity**, so degradation *cannot* free bits — it only makes the content
harder to code at that target, and the holes steal bits *from* FG, inverting
Goal 1. This is not a hypothesis: **25/25 matched VBR pairs, across every
degradation method ever run (freeze, downsample, blur, shrink), encode to MORE
bits than the pristine baseline. Zero counterexamples.** Under fixed QP the same
methods free bits (elvis_blackout −8.6% avg, elvis_freeze −9.7%, mean_fill
−6.8%).

**A VBR degradation curve is not evidence about the method — do not commission
one, and do not accept a spec that asks for one** (a 2026-07-16 TOP-PRIORITY
spec did exactly this and burned hours of GPU time re-measuring VBR laundering).
This is the same mechanism that already bit the codec-ROI work; see
the fixed-QP hard rule (rule 1 in
`68e8b6bb11d0dd9e62a67aef/research-log/hard-rules.md`).

### Reporting rule: distinguish metric effects from human perception

Appendix B of the manuscript is authoritative: the declared margins in
`src/presley/compare.py` are reporting margins, not calibrated human
just-noticeable differences. `presley-compare` summarizes region metric deltas
and the bitrate comparison at those margins; being within a margin does not
establish perceptual equality, and crossing it does not establish visibility.
Report effect sizes directly. Keep matched-QP quality comparisons separate
from bitrate savings at equal quality and from comparisons against ELVIS.

For N>1 paired runs, use `presley-compare results/ --suite --arm-key restorer
--arm-a <baseline> --arm-b <candidate> --pair-by video --candidates-tried <k>`.
Report the exact two-tailed paired test, interval and correction family,
including candidates that lost. Five unanimous pairs give p=0.0625, so small
suites are underpowered rather than evidence of equality. A nonsignificant
difference is not an equivalence test.

Numerical calculations and historical JSON codes/fields (`JND`, `jnd`,
`within_jnd`, `indistinguishable`, `perceptual_win`, `sub_jnd_significant`)
remain for compatibility with saved analyses. Use `reporting_verdict` and the
`margin_kind`/`margin_notice` fields for interpretation; legacy names do not
constitute evidence about human perception. Do not rewrite historical records.

Never report only overall metrics — the `metrics.foreground`/`metrics.background`
split is the point (and for bridge runs `overall` is actively misleading, since
the collapsed BG dominates it). Analyze each component against its designated
target:

- **Codec ROI methods** (`kvazaar`, `x265_aq`, `svtav1`) vs the **same codec's
  baseline** at comparable bitrate. Expected signature: FG quality ↑, BG
  quality ↓. If it's absent, assume our usage is wrong before blaming the
  codec — "codec X doesn't implement ROI correctly" is a strong claim needing
  evidence beyond reasonable doubt (see
  `68e8b6bb11d0dd9e62a67aef/research-log/bugs.md` for past false alarms).
- **presley_* ROI methods** (mask-driven degradation before encoding) vs the
  codec ROI methods: does direct block-level control buy more FG quality, and
  at what BG cost?
- **elvis** vs baselines, same analysis as ROI: did dropping removable blocks
  leave more bits for FG blocks at the same bitrate?
- **presley_ai** vs all of the above: FG quality must be best-in-class at
  matched bitrate, and the bitrate accounting must use
  `transmitted_size_bytes` (video + side-channel strength maps), not just the
  video file.

Exact bitrate matches are rare: compare at *similar actual* bitrates
(`actual_bitrate_bps`, not `target_bitrate`) for preliminary conclusions, and
use BD-rate curves (multiple target bitrates per method) for paper-grade
claims. If a link in the chain breaks, first search for regimes where it holds
(video subsets, bitrate ranges, codecs, parameters) before concluding the
method is worse — and only after that, re-examine the implementation.

**Fast iteration:** `presley-run … --fast-metrics` / `presley-evaluate
results/ --fast-metrics` compute only FG/BG/overall **PSNR+MSE** (SSIM,
LPIPS/DISTS/VMAF/FVMD and block-level maps are deferred to the full pass).
Fast-only results are tagged `metrics.fast_only` and get upgraded in place by a
later full `presley-evaluate results/`. The eval bottleneck is *not* the
metrics (~7% of time) — it's loading reference frames/masks from NFS, so
`evaluate_all` memoizes them across experiments in one pass (load once, not
per-experiment).

**FG-perceptual backfill:** the paper argues *foreground* perceptual quality,
but the base metrics are PSNR/SSIM. `presley-evaluate results/ --backfill-lpips`
appends region-restricted **LPIPS** (`foreground`/`background`/`overall`
`lpips_mean`) to every existing `result.json` *in place* — a metric-only pass
that re-reads the on-disk output videos and needs **no re-encoding** and no
rerun of experiments. It works on `fast_only` results too and is re-entrant
(skips ones that already have FG-LPIPS; use `--force` to recompute). LPIPS is
computed in spatial mode (per-pixel map averaged over the UFO mask), so FG/BG
are true region metrics, not bbox crops. LPIPS-alex is the fastest perceptual
metric (~0.76 s / 82 frames); DISTS/VMAF stay in the full pass.

**Starved-bitrate rule:** generative methods (elvis, presley_ai) only pay off
where the codec is bit-starved — hallucinating detail is only cheaper than
coding it when the codec can't afford the detail. Run their experiments at
bitrates low enough that the *baseline* is visibly quality-limited; a
comfortable-bitrate result understates them. The claim to pursue is "presley
wins in the starved regime," not "at every bitrate."
