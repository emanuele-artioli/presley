# `ac_truncate+nafnet` — the last open row, and why one video is not enough

Written 2026-08-05 **before launching**, per the bound-before-believing rule.

## The plan's estimate was wrong, and this is the correction

`docs/PLAN_TRADEOFF_SURFACE.md` W4 says of `ac_truncate+nafnet`: *"One video
reaches the n≥8 floor."* True, and misleading — **clearing the floor is not
reaching significance.** At n=8 and 8/8 on bitrate, raw p is 0.0078, and against
the m=14 family that is p_Holm ≈ 0.109. Nothing is citable. The arithmetic:

| n | wins | p raw | p_Holm (first in family, ×14) |
|---|---|---|---|
| 8 | 8/8 | 0.0078 | 0.109 — n.s. |
| 9 | 9/9 | 0.0039 | 0.055 — n.s., misses by a hair |
| **10** | **10/10** | **0.0020** | **0.027 — SIGNIFICANT** |

So this needs **three more videos**, exactly as `freeze+propainter` did. Note
the row that is worth buying is **bitrate**, not quality: `ac_truncate` *loses*
quality 6/7 and *wins* bitrate 7/7. **The corpus currently has three
significance-backed quality comparisons and zero on bitrate.** This is the one
candidate that could produce the first.

`drift-straight` cannot be recovered: all three of its `ac_truncate` runs *add*
bits, so they are ineligible and no re-run of that video fixes it. That is why
the arm sits at n=7 with 8 videos on disk.

## What is being run

At svtav1 QP43, 640×360, **bs16** throughout:

- **`dog`, `pigs`** — `ac_truncate+nafnet` at `ac_keep` 1 / 2 / 4 plus a matched
  `none` control each. Both already carry an eligible bs16
  `downsample+realesrgan` arm and a pristine baseline, so nothing else is needed.
- **`tennis`** — the same six, **plus** a bs16 `downsample+realesrgan` arm and
  its control, because tennis's only downsample arm at this cell is bs8. Adding
  the pair on both sides is the precedent set by the `dog`/`pigs` blur wave: it
  keeps the arm definition constant across the suite instead of letting the sign
  test partly measure block size.

All three `ac_keep` values are added per video, not one, because the aggregation
rule is best-of under the objective being reported — a video offered one setting
would be competing against videos offered three.

20 runs. Scoped run-file `config/w4c_ac_truncate.yaml`.

## Bounds

| quantity | plausible | alarm | basis |
|---|---|---|---|
| new videos where **`ac_truncate` wins bitrate** | 3 of 3 | ≤1 of 3 | 7/7 so far. `ac_truncate` transmits fewer AC coefficients, so it should keep winning the rate axis; 1 of 3 would contradict the existing series |
| resulting **bitrate** row | n=10, 9–10/10, p_Holm 0.027–0.11 | — | at 10/10 it crosses; at 9/10 it does not. **Failing to reach significance is a possible outcome with nothing wrong** |
| new videos where the **baseline wins quality** | 2–3 of 3 | 0 of 3 | baseline wins 6/7 so far |
| resulting **quality** row | n=10, 8–9/10, n.s. | p_Holm < 0.05 | if quality *also* crossed, `ac_truncate` would be dominated on both axes, which contradicts 6/7-vs-7/7 |
| `ac_truncate` Δbits vs pristine | −35…0 % | > +5 % | observed range on eligible cells: −2.1…−33.5 %. A positive Δbits makes the arm ineligible and the video does not count, exactly as on `drift-straight` |
| `ac_truncate` ΔFG vs baseline | −0.5…+0.3 dB | \|ΔFG\| > 1.0 dB | `fg_protect: true`; observed −0.22…+0.19 dB |
| tennis's new bs16 downsample arm, Δbits | −25…0 % | > 0 % | its bs8 arm reads −6.6 %; bs16 has been more negative than bs8 on dog (−24.6 vs −11.1) and pigs (−23.0 vs −12.1) |
| runs citable | 20 of 20 | < 20 | count results against entries; `presley-run` exits 0 when every entry fails |

## Reminders this wave must not re-learn

- Region LPIPS is absent on fresh runs; `presley-evaluate results/
  --backfill-lpips --only <hash>` is required or the arms never enter the
  analysis at all. **One backfill job at a time** — two writing one
  `result.json` is a corruption risk on a gitignored, hours-expensive artifact.
- `tennis` is the one video where the ROI mechanism does not appear (W4c in the
  plan: sub-JND in both directions, read as "no ROI effect"). That is about
  kvazaar ROI, not about this arm, and it is **not** a reason to expect anything
  unusual here — but if tennis is the video that breaks the pattern, this note
  is where to start rather than treating it as a fresh surprise.
