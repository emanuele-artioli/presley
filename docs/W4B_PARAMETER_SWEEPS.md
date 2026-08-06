# W4b — the sweeps that were already on disk

No GPU time was spent on this. The runs existed and were correct; the *analysis*
keyed arms on a tuple that omitted every degradation-strength parameter, so
three-point sweeps read as duplicate runs and were collapsed. `tools/
analyze_parameter_sweeps.py` finds every config key that varies inside one
(operating point, arm) group — same video, codec, QP, resolution, arm and every
other config value — and reports what varying it did.

Run 2026-08-05, `results/presley.db`:

```
python tools/analyze_parameter_sweeps.py --db results/presley.db
python tools/analyze_parameter_sweeps.py --db results/presley.db --key blur_kernel
```

| key | groups | videos | median quality spread (BG-LPIPS) | median bits spread (pp) | best value, by group |
|---|---|---|---|---|---|
| `block_size` | 24 | 4 | 0.0151 | 5.02 | quality 16×13, 24×8, 8×3 |
| `ac_keep` | 8 | **8** | 0.1026 | 5.69 | quality **4×7**, 1×1 |
| `blur_kernel` | 8 | **8** | 0.0641 | 2.26 | quality **7×7**, 31×1 |
| `downsample_uniform_level` | 8 | **8** | 0.0896 | 3.92 | quality **graded×7**, 2×1 |
| `downsample_levels` | 8 | **8** | 0.0149 | 1.07 | quality **absent×8** |
| `downsample_level_map` | 8 | 8 | 0.0050 | 0.56 | — |
| `mask_source` | 7 | 7 | **0.0038** | 4.20 | quality yolo×4, gt×2, default×1 |
| `fg_protect` | 4 | 2 | 0.0024 | 5.47 | — |
| `mask_morphology` | 2 | 2 | 0.0083 | 4.91 | — |
| `shrink_amount` | 2 | 2 | 0.1508 | 26.38 | — |

Three sweeps reach **8 videos**, which is hard rule 2b's restorer floor — so
unlike the rest of this table they can carry a sign test rather than only a
direction.

## The result worth having: graded downsampling beats uniform on quality and
## loses on bitrate

`downsample_uniform_level` absent means the **graded** mode (per-block level
chosen by the selection objective); present means every selected block is
downsampled to that one level.

> ⚠ **CORRECTED 2026-08-05, before this reached the paper.** An earlier version
> of this section said this is the comparison `HOLE(sec:downsample-vs-uniform)`
> is waiting for. **It is not.** That hole asks for importance-selected
> per-block downsampling against **whole-frame uniform** downsampling at matched
> rate. Both arms here are importance-selected; they differ only in whether the
> *strength* is graded or uniform *within the selected footprint*. The hole
> stays open, and no run on disk closes it.
>
> Worse, this is not new data: it is the same comparison the manuscript already
> carries as `CLAIM(tab:graded)` plus the S1b uniform-level probes, down to the
> same hashes (`dogs-jump` graded `a65c763b51ecdee1`, uniform k=2
> `dbe01c1d0fe8363b`, k=3 `ab2c9c567229be90`). **The paper's reading is stricter
> than the one below and governs:** the quality difference is **sub-JND**
> (max |Δ| 0.0385 against the 0.05 threshold) and `NOTE(tab:graded)` states it
> "must never be worded as a quality result". The sign test below is a
> significance test on an imperceptible difference, which is precisely what
> hard rule 3 forbids reporting as a quality finding.
>
> What survives is the *bitrate* half — uniform-3 transmits fewer bits than
> graded on 8 of 8 videos — and even that is already in the paper as the mixing
> cost identified in `NOTE(tab:graded)`. Nothing here lands.

Unit = video, n=8 (`bear`, `bike-packing`, `color-run`, `dancing`, `dogs-jump`,
`drift-straight`, `drift-turn`, `motorbike`), two-tailed exact sign test, Holm
over the family of 4 tests below:

| comparison | objective | graded wins | p raw | p_Holm |
|---|---|---|---|---|
| graded vs **uniform-3** | quality | **8/8** | 0.0078 | **0.0312** |
| graded vs **uniform-3** | bitrate | **0/8** | 0.0078 | **0.0312** |
| graded vs uniform-2 | quality | 7/8 | 0.0703 | 0.1406 |
| graded vs uniform-2 | bitrate | 2/8 | 0.2891 | 0.2891 |

**Graded is not free.** Against the aggressive uniform level it wins background
quality on every video and loses bitrate on every video — the same conflict the
tradeoff surface reports between arms, appearing here *inside* one arm's
parameter. Against the milder uniform-2 neither axis separates. The honest
statement for the paper is that graded buys quality with bits at the strong end
of the ladder and is indistinguishable from uniform at the mild end.

**Disclosure:** the direction was visible in the descriptive table above before
the test was run, so this is post-hoc. The sign test adds a p-value to a
direction that was already read off the data; it is not a pre-registered
confirmation, and n=8 sits exactly on the floor. It should be replicated on
videos outside these eight before it carries weight in the text.

## What the other sweeps say

- **`blur_kernel` is a bad lever.** Kernel 7 gives the best background quality
  in 7 of 8 videos, and the whole 7→31 range moves bitrate by a median of only
  2.26 pp. Blurring harder costs quality and buys almost nothing. If a single
  setting is quoted, it is 7.
- **`ac_keep` is a real ladder**, and it runs the other way: `ac_keep` 4 wins
  quality in 7 of 8 videos, while `ac_keep` 1 wins bitrate in 6 of 8. Median
  quality spread 0.103 BG-LPIPS is **two JND**, the largest of any sweep here —
  this parameter genuinely moves the operating point.
- **`mask_source` barely touches quality**: median spread 0.0038 BG-LPIPS,
  roughly a *thirteenth* of a JND, across ground-truth, YOLO and the default.
  Mask provenance is not what determines restored quality on these clips.
  It does move bitrate (4.20 pp).
- **`block_size` is 24 groups but only 4 videos** — it looks like the best-
  covered sweep in the table and is the worst-powered. 16 wins quality in 13 of
  24 groups; the modal share is 0.54, so it is a knob, not a default.
- **`shrink_amount` shows the largest spread in the table** (0.151 BG-LPIPS,
  26.4 pp bits) on 2 groups. Consistent with the budget-knee result; not
  evidence by itself.

⚠ `inpainter_params` remains negative on the ProPainter timing split: 158 of 161
ProPainter runs at 640×360 carry an empty params dict and still split 77 slow /
81 fast. **W3's timing campaign is still required.**

## A bug in this tool, found by reading its own output

The first version reported `downsample_uniform_level`'s best quality value as
`{'2': 1}` — one group out of eight. `None` is a legitimate parameter *value*
here (it means the key is absent, i.e. the graded mode) and the code also used
`None` to mean "no winner could be determined", so every group the graded mode
won was silently dropped. **8 of 8 was displayed as 1 of 8.** Fixed with an
explicit `NO_BEST` sentinel and pinned by two tests.

This is the fifth instance of one pattern in this plan's sessions: a correct
computation whose *absence* of output was mistaken for a null. It is worth
stating as a rule — **whenever a value's absence is meaningful data, absence can
no longer double as the error signal.**

## What this changes about the plan

The old Wave 3(b) asserted these parameters "have never been swept". Several
have been. `alpha`/`beta` (5 values each) do not appear in the table above
because no two runs sharing an operating point and arm differ in them alone —
so the selection-objective ablation is genuinely still missing, but the
degradation-strength sweeps are not.
