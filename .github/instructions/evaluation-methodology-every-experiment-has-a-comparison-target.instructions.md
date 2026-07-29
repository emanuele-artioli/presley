---
applyTo: "src/**,scripts/**,config/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Evaluation methodology — every experiment has a comparison target' section. Copilot's cloud agent and code review
     read the whole of AGENTS.md; this copy is for Copilot Chat, which
     reads only .github/. -->

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
the RESEARCH_LOG's fixed-QP hard rule.

### Reporting rule: never dress up imperceptible deltas

Imperceptible deltas are not a result or a trend. **Run `presley-compare` to
decide whether a quality difference is real** — don't eyeball deltas. Its JND
table (`src/presley/compare.py`) is the single source of truth and is
deliberately not restated here. `presley-compare results/ --hash-a <h1>
--hash-b <h2>` for a pair; `presley-compare results/ --group-by
component,video,codec_params.qp --baseline-component baselines` for a
matched-QP sweep, which reports each group's quality verdict and its bitrate
winner. At matched QP this is the *whole* analysis: FG differences are small
by construction, so the question is never "who wins FG" but "who encodes
fewest bits at indistinguishable FG quality." State it the way it lands: *"at
FG quality that is indistinguishable, method X costs N% fewer bits than the
baseline, and BG-LPIPS is Y vs the baseline's Z."*

For **N>1 paired runs**, JND alone is too blunt: a sub-JND effect that
reproduces on every video is real even though it stays imperceptible. Add
`presley-compare results/ --suite --arm-key restorer --arm-a <baseline> --arm-b
<candidate> --pair-by video --candidates-tried <k>` (`src/presley/suite.py`),
which layers an exact sign/Wilcoxon test, a bootstrap CI and Holm correction on
top of the JND verdict without ever overriding it. Three things it exists to
enforce, all of which have already been got wrong: p-values are **two-tailed**
(5/5 is p=0.0625, not 0.031); n≤5 cannot reach α=0.05 at all, so a consistent
small suite is `underpowered`, not "no effect"; and `--candidates-tried` must
count every candidate ever tried against that baseline, **including the losers**.
The strongest verdict a sub-JND effect can earn is `sub_jnd_significant`, which
is never worded as a win. Details and the audit of existing claims:
`docs/SIGNIFICANCE_AUDIT.md`; the rule itself is RESEARCH_LOG hard rule 2b.

Never report only overall metrics — the `metrics.foreground`/`metrics.background`
split is the point (and for bridge runs `overall` is actively misleading, since
the collapsed BG dominates it). Analyze each component against its designated
target:

- **Codec ROI methods** (`kvazaar`, `x265_aq`, `svtav1`) vs the **same codec's
  baseline** at comparable bitrate. Expected signature: FG quality ↑, BG
  quality ↓. If it's absent, assume our usage is wrong before blaming the
  codec — "codec X doesn't implement ROI correctly" is a strong claim needing
  evidence beyond reasonable doubt (see RESEARCH_LOG.md for past false alarms).
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
