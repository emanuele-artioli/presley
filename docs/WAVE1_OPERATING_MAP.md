# Wave 1 — the operating map from existing data

**Status:** complete 2026-08-03. No GPU time spent. Wave 1 of
`docs/PLAN_OPERATING_MAP.md`, which is its falsification gate.

**Verdict: PASSES.** None of the three hard stops triggered, the soft
threshold stop did not trigger, and all three pre-registered bounds came in
band. Waves 2A / 2B / 2C are unblocked and may launch together.

Reproduce:

```bash
python tools/build_operating_map.py --db results/presley.db --verbose --json map.json
```

Tests: `tests/test_build_operating_map.py` (23 cases, fast tier).

---

## Coverage

926 fixed-QP citable runs → **420 scored arms across 130 operating points on 23
videos**. An operating point is `(video, codec, QP, width, height)`; an arm is a
`transport+fill` pair. **84** operating points carry ≥2 arms and enter the map;
**46** of those have ≥2 *eligible* arms, where eligible means the arm actually
saves bits against the pristine baseline at the same QP **and** costs no more
than the 0.5 dB FG-PSNR JND.

That 46-of-84 is itself a result: **24 cells have no deployable arm at all** and
14 have exactly one. Roughly half the corpus does not pose the map's question.

Everything below is an **effect size measured within an operating point** — no
pass rates, no medians pooled across cells, so nothing is confounded by which
videos a transport happened to be run on.

## The map, under two objectives

| objective | separable winner | ties | distinct winners |
|---|---|---|---|
| quality-first (lowest BG-LPIPS) | **19/46 (41.3%)** | 27 | 2 |
| rate-first (fewest bits) | 36/46 (78.3%) | 10 | 3 |

Winners, quality-first: `downsample+realesrgan` 18 cells, `blackout+propainter`
1. Rate-first: `blackout+propainter` 27, `downsample+realesrgan` 6,
`blur+nafnet` 3.

**The two objectives almost never agree: 1 of the 19 cells separable under both
names the same arm** (`bear`/x265/QP30, `blackout+propainter`). This is the
single most consequential number in Wave 1. The recommendation is not a property
of the content and rate alone — it is a property of *what the deployer is
optimising*. An article that reports one scalarization silently reports a choice
it never made explicit.

The quality-first winners span **8 videos** (bear, camel, 4 MOSEv2 clips, 2
YouTube-VOS clips) and QPs 30–62. Eight videos clears hard rule 2b's n≥6, so a
significance test on "downsample+realesrgan wins quality-first" is now *in
scope* — it has not been run, and until it is, this stays descriptive.

## Ladder residuals

The rate–damage ladder is fitted per operating point over the arms' unrestored
(`none`-control) BG-LPIPS, and each arm is scored by its signed residual — the
BD-rate analogue the article was missing. **106 arms across 12 operating points**
carry a fit; the rest lack the `none` controls (far sparser than baselines,
which is why the map itself never requires one).

**10 deployable arms sit ≥1 JND off the ladder in the good direction** — less
background damage than their bit saving should have cost. Best from a
well-fitted ladder: **+1.32× JND**, `camel`/svtav1/QP50, `downsample` (every
fill), ending at BG-LPIPS 0.152–0.213.

The residual is never reported alone. `blackout` is the standing warning — the
corpus's largest restoration *gain* and its worst absolute result — so the tool
prints residual, absolute restored BG-LPIPS and bit delta on the same row, and
tests pin that behaviour.

Two guards were added after inspecting the first output, both because the first
ranking was misleading:

- **Fits with a positive slope are refused, not scored.** A positive slope means
  that cell did not exhibit the rate–damage tradeoff at all; residuals from it
  are distances from an arbitrary line.
- **Fits with R² < 0.5 are flagged `!` and excluded from the gate.** `pigs`/QP50
  fits at R²=0.016 — a cloud. Its residual (+1.26×) may not be quoted, and the
  gate uses +1.32× from a well-fitted cell instead.

The first ranking, before these guards, was topped by `freeze` arms that *spend*
bits (+3.9%, +13.3% vs baseline). Those can never be recommendations — they fail
the map's own feasibility test — so the headline ranking is now over deployable
arms only, with the rest visible under `--verbose`.

## Cost

Restoration throughput at 640×360, median fps by fill: unsharp 4.57,
realesrgan 4.48, telea 3.75, nafnet 3.14, bsrgan 2.92, stream_diffvsr 1.67,
e2fgvi 1.63, real_hat_gan 1.14, **propainter 0.78**, instantir 0.17. Source is
24 fps.

**244 of 420 arms are Pareto-dominated inside their own cell** — another arm is
no worse on BG-LPIPS *and* no slower. The map's live frontier is far smaller
than the arm count suggests. Note the objectives' disagreement has a cost
dimension too: rate-first's favourite (`propainter`, 0.78 fps) is the slowest
generative fill in the corpus, and quality-first's (`realesrgan`, 4.48 fps) is
5.7× faster.

## JND threshold sensitivity

The LPIPS threshold in `src/presley/compare.py` is adopted convention — it cites
nothing and gives LPIPS and DISTS the same 0.05. Recomputing the map at 0.03 /
0.05 / 0.08:

| LPIPS JND | separable | of contested | distinct winners |
|---|---|---|---|
| 0.030 | 25 | 54.3% | 2 |
| 0.050 | 19 | 41.3% | 2 |
| 0.080 | 17 | 37.0% | 2 |

**17 cells name a winner at every threshold and none of them names a different
winner** — the ranking cannot move with the constant, only the licence to call
it separable. **8 cells name a winner only at 0.03** and must be reported as
threshold-dependent. The soft stop does not fire.

Still outstanding from the plan's item 3: **cite or downgrade**. Nothing here
finds provenance for the 0.05; the article should state it as an operating
convention and lean on this table.

## Gate — reported against all four conditions

| condition | outcome |
|---|---|
| 1. No cell has a separable winner | **no** — 19 quality-first, 36 rate-first |
| 2. One arm wins everywhere | **no** — 2 quality-first winners, 3 rate-first |
| 3. Every off-ladder residual within JND | **no** — 10 deployable arms ≥1 JND off |
| 4. (soft) Recommendations flip under sensitivity | **no** — 0 flips |

Pre-registered bounds, all in band:

| quantity | pre-registered plausible | observed |
|---|---|---|
| separable cells | 25–60% | **41.3%** |
| distinct winners | 2–4 | **2** |
| best off-ladder residual | 1–3× JND | **1.32×** |

Distinct winners landed on the low edge of its band, and the =1 alarm was one
cell away: 18 of 19 quality-first winners are the same arm. If Wave 2B's added
coverage pushes that to 19/19, gate condition 2 fires retroactively and the map
collapses to a single global recommendation under that objective. **Re-run this
tool after every batch of new runs**, not only at the end.

## What Wave 1 hands to Wave 2

- **2A (content axis).** The map is a lookup table until "this kind of video" is
  defined. The 27 quality-first ties and the 24 no-eligible-arm cells are the
  interesting negatives: whatever predicts transport choice must also predict
  where there is nothing to choose.
- **2B (fill the holes).** Two specific asks, in order: `none` controls at the
  operating points that lack them (they gate the ladder, which currently covers
  12 of 130 cells), then the coverage that lets the n=8-video quality-first
  result be tested rather than described.
- **2C (efficiency).** The fps table above is per-fill and unconditioned; 2C
  still owns the encode-throughput alarm (svtav1 measuring 28.9 fps at 1080p vs
  3.2 at 720p) before any throughput number is published.
