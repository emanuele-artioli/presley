# W4 — `freeze+propainter` to n=10: bounds written before the runs

Written 2026-08-05, **before launching**, per the bound-before-believing rule.
Plan: `~/.claude/plans/after-wave-2b-reported-cosmic-sunbeam.md`, W4.

## What is being run

8 runs: `freeze+propainter` and its matched `none` control on **color-run,
dancing, drift-turn, motorbike** at svtav1 QP43, 640×360, **bs16**.
Scoped run-file `config/w4_freeze_propainter.yaml`.

Those four were chosen over the plan's six candidates because each already
carries an **eligible** bs16 `downsample+realesrgan` arm and a pristine
baseline at that cell, so no new baseline is needed and the block size matches
on both sides. `india` is excluded: its downsample arm at this cell is bs8, so
pairing bs16 freeze against it would make the sign test partly a measurement of
block size — the trap the `dog`/`pigs` wave had to spend four extra runs
avoiding. `drift-straight` is excluded as second reserve: four of its five
downsample arms are ineligible (they *add* bits), leaving the cell resting on a
single −3.07% arm.

## Bounds

Baseline arm `downsample+realesrgan`; unit = video; two-tailed exact sign test;
Holm within a family of 14 per objective. Analysis:
`tools/analyze_w4_significance.py`.

| quantity | plausible | alarm | basis |
|---|---|---|---|
| new videos where the baseline wins **quality** | 3–4 of 4 | ≤1 of 4 | 6/6 so far, and the same ordering holds 11/11 for `blur+nafnet` and 10/10 for `blackout+propainter`. A freeze win on 3+ of 4 fresh videos would contradict three independent series |
| resulting **quality** row | n=10, 9–10/10, p_Holm 0.025–0.11 | — | at 10/10 raw p = 0.002 → ×13 ≈ 0.025; at 9/10 raw p = 0.021 → n.s. This run **can fail to reach significance without anything being wrong** |
| new videos where the baseline wins **bitrate** | 3–4 of 4 | ≤1 of 4 | 6/6 so far. freeze transmits a frozen block, which is cheap, so it should keep losing the rate axis to downsample |
| `freeze+propainter` **ΔFG** vs baseline | −0.5…+0.3 dB | \|ΔFG\| > 1.0 dB | `fg_protect: true`; the two existing bs16 cells read +0.08 and +0.18 dB |
| **Δbits** vs pristine baseline | −20…0 % | > +5 % | existing bs16 freeze cells: −4.0 % and −12.4 %. A positive Δbits makes the arm ineligible and the video does not count |
| wall clock, all 8 runs incl. evaluation | 20–90 min | > 4 h | ProPainter measured at 39–77 s per run at 640×360; wall clock is dominated by evaluation and backfill, not the restorer |
| runs landing with metrics and empty `invariant_failures` | 8 of 8 | < 8 | `presley-run` exits 0 when every entry fails — count results, never trust the exit code |

## The two failure modes this wave must not repeat

1. **`n` not moving after clean runs.** A fresh run carries only `overall`
   LPIPS; `presley-evaluate results/ --backfill-lpips` is required before the
   arms exist for the analysis at all. `analyze_w4_significance.py` inherits the
   drop-counting from `build_operating_map`, so an arm missing for this reason
   is now printed rather than skipped in silence.
2. **Exit 0 on total failure.** Verify 8 result directories with metrics
   against 8 entries; `grep -c 'Error running experiment' <log>`.

---

# Result — `freeze+propainter` crosses. Three significance-backed comparisons.

Run 2026-08-05. 8/8 runs landed with metrics and **empty `invariant_failures`**,
verified against the DB rather than the runner's exit code; `grep -c 'Error
running experiment'` = 0. Region LPIPS was absent on all 8 as expected and was
backfilled with `presley-evaluate results/ --backfill-lpips --only <hash>`, one
job at a time, 8/8 OK.

| candidate | objective | n | base wins | p raw | p_Holm | verdict |
|---|---|---|---|---|---|---|
| `blur+nafnet` | quality | 11 | 11/11 | 0.0010 | **0.0137** | SIGNIFICANT |
| `blackout+propainter` | quality | 10 | 10/10 | 0.0020 | **0.0254** | SIGNIFICANT |
| **`freeze+propainter`** | quality | **10** | **10/10** | 0.0020 | **0.0254** | **SIGNIFICANT** |
| `ac_truncate+nafnet` | bitrate | 7 | 7/7 | 0.0156 | 0.2188 | underpowered |
| `blackout+propainter` | bitrate | 10 | 1/10 | 0.0215 | 0.2793 | n.s. |
| `freeze+propainter` | bitrate | 10 | 9/10 | 0.0215 | 0.2793 | n.s. |
| `ac_truncate+nafnet` | quality | 7 | 6/7 | 0.1250 | 1.0000 | underpowered |
| `blur+nafnet` | bitrate | 11 | 5/11 | 1.0000 | 1.0000 | n.s. |

**`downsample+realesrgan` now beats three different arms on background quality,
each on every video tested: 11/11, 10/10, 10/10.** That is no longer a result
about one rival.

**On bitrate nothing is significant and the direction still is not uniform** —
`blackout` wins the rate axis (baseline wins 1 of 10), `freeze` loses it
(baseline wins 9 of 10), `blur` is a coin flip. The quality ordering is
established three times over; the rate ordering is established zero times, and
the two disagree about which arm to deploy. That is the tradeoff, stated
precisely.

## Bounds: all held, no alarms

| bound | predicted | observed | status |
|---|---|---|---|
| new videos, baseline wins quality | 3–4 of 4 | **4 of 4** | in band |
| resulting quality row | n=10, 9–10/10, p_Holm 0.025–0.11 | n=10, 10/10, p_Holm 0.0254 | in band, at the good end |
| new videos, baseline wins bitrate | 3–4 of 4 | 3 of 4 (`color-run` flips) | in band |
| `freeze` ΔFG vs baseline | −0.5…+0.3 dB | −0.23…+0.15 dB | in band |
| `freeze` Δbits vs pristine | −20…0 % | −18.75…−4.02 % | in band |
| runs citable | 8 of 8 | 8 of 8 | in band |

One observation not covered by a bound, recorded rather than dressed up:
`dancing`'s restoration gain is **−0.12 JND** — ProPainter left the background
marginally worse than the unrestored control there. Sub-JND, so the defensible
reading is "no restoration effect on that cell", not "restoration hurt".

## Remaining gap

`ac_truncate+nafnet` is the last open row at n=7, losing quality 6/7 and winning
bitrate 7/7. **One video** takes it to the n≥8 floor.
