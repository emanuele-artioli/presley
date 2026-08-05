# W4 — the significance picture, after two added videos

**Status:** SUPERSEDED 2026-08-05 by `docs/W4_FREEZE_BOUNDS.md`, which added four
videos and took `freeze+propainter` to n=10, 10/10, **p_Holm 0.0254 —
SIGNIFICANT**. The table below is the 2026-08-03 state and every `freeze` row in
it is stale; the rest still reproduces exactly. The analysis behind both is now
committed as `tools/analyze_w4_significance.py` (it was not, when this was
written, which is why it had to be reconstructed).

Plan: `docs/PLAN_TRADEOFF_SURFACE.md`.

## What was run

8 runs: `blur+nafnet`, `downsample+realesrgan` and their matched `none`
controls at `dog` and `pigs`, svtav1 QP43 640×360, **bs16 on both sides**. Those
cells previously carried only bs8 downsample arms, so pairing bs16 blur against
bs8 downsample would have made the sign test partly a measurement of block size.
Adding the pair keeps the arm definition constant across the suite.

8/8 landed with metrics and empty `invariant_failures`, verified against the DB.

## Result — `blur+nafnet` crosses, and there are now two significant comparisons

Baseline arm `downsample+realesrgan`; two-tailed exact sign test on per-video
means; Holm over **m=14** candidates (every arm ever compared against this
baseline, losers included); hard rule 2b's n≥8 floor for restorer comparisons.

| candidate | objective | n | base wins | p raw | p_Holm | verdict |
|---|---|---|---|---|---|---|
| `blur+nafnet` | quality | **11** | 11/11 | 0.0010 | **0.0137** | **SIGNIFICANT** |
| `blackout+propainter` | quality | 10 | 10/10 | 0.0020 | **0.0254** | **SIGNIFICANT** |
| `freeze+propainter` | quality | 6 | 6/6 | 0.0312 | 0.3750 | underpowered (n<8) |
| `ac_truncate+nafnet` | quality | 7 | 6/7 | 0.1250 | 1.0000 | underpowered (n<8) |
| `ac_truncate+nafnet` | bitrate | 7 | 7/7 | 0.0156 | 0.2188 | underpowered (n<8) |
| `blackout+propainter` | bitrate | 10 | 1/10 | 0.0215 | 0.2793 | n.s. |
| `freeze+propainter` | bitrate | 6 | 6/6 | 0.0312 | 0.3750 | underpowered (n<8) |
| `blur+nafnet` | bitrate | 11 | 5/11 | 1.0000 | 1.0000 | n.s. |

**Before this wave there was exactly one significance-backed comparison in the
corpus. There are now two**, and they point the same way: on background quality
`downsample+realesrgan` beats `blackout+propainter` on 10/10 videos and
`blur+nafnet` on 11/11. Two runs bought the second one — `blur+nafnet` was at
p_Holm 0.0508, missing by a hair.

**On bitrate, nothing is significant and the direction is not uniform.**
`blackout+propainter` wins the rate axis (baseline wins only 1 of 10) but at
p_Holm 0.279; `blur+nafnet` is a coin flip (5/11). This is the tradeoff stated
precisely: the quality ordering is established, the rate ordering is not, and
they disagree about which arm is best.

## What would close the remaining gaps

- `freeze+propainter` (n=6): **n=8 is not enough.** At 8/8 raw p is 0.0078 and
  p_Holm ≈ 0.10. It needs ~10/10 → four more videos.
- `ac_truncate+nafnet` (n=7): loses on quality (6/7 to the baseline) and wins on
  bitrate (7/7), both underpowered. One more video takes it to the n≥8 floor.

## The trap this wave walked into, worth more than the result

All 8 runs landed clean and **`blur+nafnet`'s n did not move.** The new arms
were not in the analysis at all: a fresh run carries only `overall` LPIPS until
`presley-evaluate --backfill-lpips` touches it, and `score_arms` skipped them
silently. **A newly added arm does not error — it simply never appears, and that
is indistinguishable from the runs having failed.** The tool now counts every
exclusion by reason and prints it (which also surfaced 34 runs with no usable
baseline, previously invisible).

Second trap, same shape: `python -m presley.evaluation` is a package and cannot
be executed. The first backfill loop failed on all 8 hashes, wrote `FAILED` per
hash into its log, and the shell still **exited 0**. The entry point is the
`presley-evaluate` console script. Count failures explicitly; never read exit 0
as success.
