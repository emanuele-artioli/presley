# Rate-matched PRESLEY vs ELVIS on the 9 non-DAVIS clips

**Bounds pre-registered 2026-07-31, before any of these runs exist.** Committed
before launch, per the bound-before-believing rule. The previous session breached
two of its pre-registered bounds and recorded them as breached rather than
re-fitting them; that is the standard this document is held to.

## Why this experiment exists

`tab:breadth-ext-presley` landed the non-DAVIS PRESLEY arm on 2026-07-31. Against
ELVIS at the **same QP**, PRESLEY won the background above JND on 9/9 clips at
both rate points (BG-LPIPS −0.127 / −0.120, BG-PSNR +6.88 / +6.37 dB, sign
p=0.004), with foreground protection intact.

**But it spent +23.1% (QP 32) / +14.9% (QP 37) more bits to do it.** The paper's
expected chain

```
  baseline < roi < elvis < presley_ai
```

is specified **at matched bitrate** (`research-log/hard-rules.md`, Goal 1). A win
bought with a 15–23% bit premium does not test that ordering, so the chain is
currently **unclaimed outside DAVIS** and `open-questions.md` item 0 holds it
open. This experiment closes it, in whichever direction it closes.

## Why BD-rate, and not a QP search

There is **no QP-search or rate-targeting machinery in this repo** — confirmed
across `src/presley/`, `tools/` and `scripts/`. Building one is not an option
either: hard rule 1 forbids VBR / bitrate-targeting rate control for degradation
comparisons, because under VBR the encoder spends its target regardless of source
complexity and degradation cannot free bits at all (25/25 matched VBR pairs
encode to *more* bits than pristine; zero counterexamples).

So the only rule-compliant route to a matched-rate statement is **more fixed-QP
rungs, then Bjøntegaard interpolation** across the resulting rate/quality curves,
using the existing `scripts/bd_rate.py` (`bd_rate`, `bd_quality`,
`overlap_fraction`). It needs **>= 4 points per curve**; there are currently 2.

## Design

- **Rungs: QP {32, 37, 42, 47}.** 32 and 37 already exist for all three
  components. 42 and 47 are new, and are chosen deliberately to extend into the
  **bit-starved regime**: the starved-bitrate rule says generative methods only
  pay off where the codec cannot afford the detail, so a comfortable extra rung
  would spend runs exactly where the claim is weakest.
- **9 clips**, unchanged: `mosev2/{8i1uo3x9,fii86rku,jxmcdk8k,ptq7rtia,zofozj6l}`,
  `youtube_vos/{0e4068b53f,282651c6f7,30fe0ed0ce,b1a8a404ad}`.
- **36 new runs**: `elvis` and `presley_ai`, 9 clips x 2 new QPs each. Every key
  is copied verbatim from that clip's existing QP-32 entry; **only
  `codec_params.qp` differs**, so the new rungs sit on the same curve as the old
  ones rather than forming a second, subtly different experiment.
- `baselines` at QP 42/47 is a **second wave, launched only if the first 36 land
  clean.** It is not needed for the ELVIS-vs-PRESLEY comparison; it exists to
  keep the baseline/ELVIS/PRESLEY triple intact at the new rungs.
- Fixed QP throughout. Never VBR.

### The rate axis

`actual_bitrate_bps` for **both** arms. This satisfies hard rule 4 without any
special-casing: `presley_ai` already derives `actual_bitrate_bps` from
`total_transmitted_bytes = transmitted video + strength maps`
(`src/presley/components/presley_ai.py:492-496`), which is exactly the quantity
rule 4 demands. `file_size_bytes` (the restored output) is never the rate.

### The quality axis

- **Primary: BG-LPIPS** (`background.lpips_mean`). Goal 2's headline metric, and
  the axis on which the same-QP win was measured.
- **Protection check: FG-LPIPS** (`foreground.lpips_mean`).
- **Corroborating: BG-DISTS.** BG-PSNR is reported but is never the verdict —
  hard rule 3 (a flat fill scores the best BG-PSNR while being perceptually
  worst).

## Bounds, written before any number exists

The honest prior: **at matched rate the background win should shrink, and it may
vanish.** Some of the same-QP advantage was bought with the 15–23% premium, and
removing that is the whole point of the experiment. A null is a real outcome here
and narrows what the paper may claim; it does not get tuned away.

1. **BD-rate on BG-LPIPS, PRESLEY vs ELVIS (primary).** Negative means PRESLEY
   reaches the same background quality for fewer bits. The same-QP gap
   (−0.127 LPIPS, well over 2x the 0.05 JND) is large relative to a ~20% bit
   premium, so the expectation is still clearly negative:
   **−70% to −20%**.
   **ALARM if positive** — PRESLEY worse at matched rate would reverse the
   same-QP direction entirely, and must be investigated as an eval/curve bug
   before it is reported as a finding.
   **ALARM below −85%** — implausibly large, and the likely cause is
   extrapolation across barely-overlapping curves rather than a real effect.

2. **Overlap, and the one way this analysis can silently lie.** ELVIS blacks out
   the background and in-paints it, so its BG-LPIPS may be nearly **flat in
   rate** — the in-painted content does not depend much on QP. If ELVIS's
   BG-LPIPS range never reaches PRESLEY's, a BD number is pure extrapolation and
   is meaningless however authoritative it looks.
   Expect rate `overlap_fraction` **>= 0.5** and the two BG-LPIPS ranges to
   intersect.
   **ALARM if the BG-LPIPS ranges do not overlap at all.** In that case **no BD
   number may be quoted.** The honest statement is instead: *PRESLEY's background
   quality is outside ELVIS's reach at any rate in this range* — which is a
   stronger claim than a BD-rate, and must be worded as non-overlap, not as a
   number.

3. **BD-rate on FG-LPIPS.** Both arms run `fg_protect` + `composite_output`, so
   foreground pixels are passthrough in both and the curves should nearly
   coincide. Expect a wash: **−15% to +15%**.
   **ALARM outside ±35%** — that would mean foreground protection behaves
   differently between the two arms, which is a wiring question, not a result.

4. **Monotonicity, per arm per clip.** Bitrate must fall monotonically as QP
   rises, and BG-LPIPS must worsen (rise) monotonically.
   **ALARM on any non-monotone rung.** That points at an encode, a cache
   collision or an evaluation bug, and is investigated before anything is read.

5. **Is QP 47 actually starved?** Expect QP 47's bitrate to be **<= 50%** of
   QP 32's on the same clip and arm.
   **ALARM if QP 47 is not at least 30% cheaper than QP 32** — the new rungs
   would then not be sampling the regime they were added for, and the curve is
   too short to interpolate across honestly.

## Decision rule, fixed in advance

Stated now so it cannot be reinterpreted after the numbers land — the failure
mode the previous session hit was an *underspecified* rule, not a broken one.

- **BD-rate(BG-LPIPS) <= −20%, with overlapping quality ranges:** the chain is
  confirmed at matched rate on non-DAVIS data. The paper may say `presley_ai >
  elvis` holds outside DAVIS, quoting the BD-rate and its `overlap_fraction`.
- **−20% < BD-rate < 0%:** directionally consistent but modest. Report the
  number; do **not** word the chain as confirmed.
- **BD-rate >= 0%:** the chain is **not** confirmed at matched rate — the same-QP
  win was bought with bits. This gets reported as such, and
  `tab:breadth-ext-presley`'s surrounding text narrows accordingly.
- **Non-overlapping BG-LPIPS ranges:** report as non-overlap per bound 2. No BD
  number.
- **Any bound breached:** recorded as breached, investigated before reporting,
  and the band revised only with an explicit stated reason.

In every branch the result lands with its `results/<hash>` set, and
`NEXT(tab:breadth-ext-presley)` plus `open-questions.md` item 0 are cleared only
by the edit that lands the data.
