# Wave 2B — filling the operating map's holes

**Status:** in progress 2026-08-03. Wave 2B of `docs/PLAN_OPERATING_MAP.md`,
run against the Wave 1 tool and report
(`tools/build_operating_map.py`, `docs/WAVE1_OPERATING_MAP.md`, branch
`claude/operating-map-implementation-34a243`).

Baseline before any Wave 2B run, reproduced exactly from Wave 1:
**926 fixed-QP citable runs → 420 arms, 130 operating points, 23 videos;
84 cells with ≥2 arms, 46 contested; ladder fitted on 12 of 130.**

---

## Headline: two of the plan's three GPU priorities rested on wrong premises

Wave 2B's priority list was written from Wave 1's summary. Checking it against
the database before spending GPU time found that priorities 1 and 2 were
mis-scoped, and priority 3 needed no GPU time at all.

| plan priority | as written | what the data says |
|---|---|---|
| 1. run the 16 pending `none` controls | "legitimate, already specified, never run" | **all 16 are unusable as written and were retired** (below) |
| 2. transport coverage at the 24 cells with no deployable arm | implies a coverage gap | **21 of 24 already have a `downsample` arm**; the cells are empty for a *rate* reason, not a transport reason. Addressable surface: **3 cells** |
| 3. paired coverage for a significance test | "target `downsample+realesrgan` vs its runner-up" | the pairing **already exists at n=10 videos**; the test had simply never been run |

---

## Pre-registered bounds

Written before launching each batch and before reading any headline metric.

**Batch 1** — 5 `none` controls, `presley_ai` / `downsample` / svtav1 QP43 /
640×360 / `block_size` 16 / `shrink_amount` 0.25 / `fg_protect` on, on
`motorbike`, `drift-straight`, `dancing`, `drift-turn`, `color-run`:

| quantity | plausible range | basis |
|---|---|---|
| results vs entries | 5 of 5 | `none` is the cheapest run type; no restorer to fail |
| ladder operating points | 12 → 14–17 | best case all 5 fit; worst case 12 if slopes come out positive |
| quality-first winners | unchanged at 18 + 1, distinct = 2 | controls add no *scored* arms; movement here is an alarm |
| best deployable residual | 1.0–3.0× JND | 1.32× observed in Wave 1 |
| unrestored control BG-LPIPS | 0.15–0.45 | restored `bear` QP43 was 0.141; unrestored must be worse |

**Batch 2** — `downsample+realesrgan` plus its matched `none` control at
`bmx-trees`, `india`, `pigs`, x265 QP34 / 640×360 / `block_size` 8 /
`shrink_amount` 0.25 / `fg_protect` on (6 entries):

| quantity | plausible range | basis |
|---|---|---|
| results vs entries | 6 of 6 | |
| `downsample` bit delta | −25% to +5% | these cells have no eligible arm today; this tests whether that is coverage or structure |
| new eligible cells | 0–3 | **0 is itself the finding**, not a failure |
| restored BG-LPIPS | 0.10–0.35 | corpus range at comparable QP |
| gate condition 2 | must **not** fire | batch can only add `downsample+realesrgan` wins (18→21); the lone `blackout+propainter` cell (`bear`/x265/QP30) is untouched, so distinct winners stay 2 |

---

## Priority 1 — the 16 pending `none` controls are unusable; all 16 retired

The plan carried these as ready-to-run. They are not, for two independent
reasons, and the count reconciles cleanly: the YAML holds **20** unrun
fixed-QP `none` entries, of which **4 are `noise`** and already carry
`_retired` (noise injection retired for spending 76.5–83.0% more bits at
indistinguishable FG-PSNR). That leaves exactly the 16 the plan meant.

**Reason 1 — they can never match the arms they were meant to control.**
The map tool keys a control by
`(operating point, component, transport, block_size, shrink_amount)`
(`build_operating_map.py`, `Run.config`). All 16 entries omit `shrink_amount`
and `fg_protect`, which default to `None` and `False`
(`components/presley_ai.py`: `experiment.get('shrink_amount')`,
`experiment.get('fg_protect', False)`). Every restored arm at those operating
points uses `shrink_amount: 0.25, fg_protect: true`. The control key therefore
never matches, and the run would have produced a result that the ladder
silently ignores.

**Reason 2 — even corrected, none of them controls anything.** Constructing the
corrected form of each of the 16 and querying the database:

| video / QP | corrected control already exists | restored `downsample`/`blur` arm exists |
|---|---|---|
| bear, camel @ QP32, QP37 (8 entries) | **yes** — redundant | **no** |
| bear, camel @ QP30, QP34 (8 entries) | no | **no** |

**All 16 have no restored `downsample` or `blur` arm at their operating point.**
At those x265 cells the only presley-side arms are `elvis` `blackout`/`freeze`
and `presley_ai` `freeze`/`mean_fill` — different transports, which a
`downsample` control cannot anchor.

**Disposition: retired in place**, 8 as `superseded` and 8 as `orphan`, each
with the reason recorded in `experiments.yaml` so the next session does not
re-plan around them. `presley-run` now reports `Skipping 8 retired
experiment(s)` for the blur filter and excludes them. Reason strings were
written through `yaml.safe_dump`; the file re-parses at 901 entries and the
diff is exactly 16 added lines.

> **Trap worth recording:** `experiments.yaml` uses YAML anchors/aliases
> (`&id001` / `*id001`). Entry blocks therefore **cannot be parsed or
> re-dumped independently** — a first attempt hit
> `ComposerError: found undefined alias 'id001'`, and re-dumping a block that
> *defines* an anchor would silently break later blocks that reference it. The
> safe edit is to parse the whole file once, map entries to text blocks by
> order, and **append** the new key as text rather than re-serialising.

### What actually unlocks the ladder

The binding constraint is not the 16. Across the corpus, **24 operating points
would reach a ≥3-config ladder if their missing controls were run**, needing 69
controls in total. Five of them need exactly **one** control each — all the same
config, `presley_ai` / `downsample` / bs16 / shrink 0.25 at svtav1 QP43 640×360,
on `motorbike`, `drift-straight`, `dancing`, `drift-turn` and `color-run`. That
is batch 1: the cheapest possible purchase of new ladder cells, and it targets
the exact arm family the quality-first map is built on.

---

## Priority 2 — the 24 empty cells are a rate phenomenon, not a coverage gap

Of the 67 arms living in the 24 cells with no deployable arm, the reason they
fail eligibility is:

| reason | arms |
|---|---|
| **spends bits** vs the pristine baseline at the same QP | **59** |
| FG damage beyond the 0.5 dB JND | 8 |

So these cells cannot pose the map's question because at those rates no
transport *saves* bits — not because a transport is missing. The cells
concentrate at high QP (x265 QP47 ×8, QP42 ×4), which is the expected
direction: the baseline is already cheap enough that block signalling overhead
outweighs the saving.

**Only 3 of the 24 lack a `downsample` arm** — `bmx-trees`, `india` and `pigs`,
all x265 QP34, all currently carrying only `blackout`/`freeze`. Those 3 are the
entire addressable surface of priority 2, and they are batch 2. Adding a 3rd or
4th transport to the other 21 would not create a map cell.

---

## Priority 3 — no GPU time needed; the pairing already existed

Hard rule 2b wants the same arm pair compared across enough videos at
comparable operating points. For `downsample+realesrgan` that was already
satisfied — the test had never been run.

Method: restrict to cells where **both** arms are eligible, average the
per-cell difference **within each video first** (videos are the unit, so
repeated cells do not become pseudo-replication), then a two-sided exact sign
test over videos. Multiplicity is controlled with **Holm over a family of
m = 14 — every candidate ever compared against the baseline, losers included**,
not just the ones that looked good. Hard rule 2b's **n ≥ 8 for restorer
comparisons** is applied as a hard floor: anything below it is reported
`underpowered`, never as a win.

### Quality-first (BG-LPIPS), baseline `downsample+realesrgan`

| candidate | n videos | baseline wins | p (2-sided) | p Holm | verdict |
|---|---|---|---|---|---|
| `blackout+propainter` | 10 | 10 | 0.0020 | **0.0273** | **significant** |
| `blur+nafnet` | 9 | 9 | 0.0039 | 0.0508 | n.s. |
| `freeze+propainter` | 6 | 6 | 0.0312 | 0.3750 | underpowered (n<8) |
| `ac_truncate+nafnet` | 7 | 6 | 0.1250 | 1.0000 | underpowered (n<8) |

### Rate-first (bit delta), same baseline, same family

| candidate | n videos | baseline wins | p (2-sided) | p Holm | verdict |
|---|---|---|---|---|---|
| `freeze+propainter` | 6 | 6 | 0.0312 | 0.4375 | underpowered (n<8) |
| `blackout+propainter` | 10 | **2** | 0.1094 | 1.0000 | n.s. |
| `blur+nafnet` | 9 | 3 | 0.5078 | 1.0000 | n.s. |

**Exactly one comparison in the corpus survives multiplicity correction:**
`downsample+realesrgan` beats `blackout+propainter` on background LPIPS,
10 videos out of 10, p_Holm = 0.027. Under rate-first **nothing is
significant**, and the direction reverses — `blackout+propainter` wins on bits
on 8 of those same 10 videos.

An earlier uncorrected reading of these numbers (10/10, 9/9 and 6/6 all "wins")
overstated the result on three counts: it quoted one objective as though it
were both, it applied no multiplicity correction, and it counted an n=6
comparison that hard rule 2b does not permit. The corrected table above
supersedes it.

### What this does and does not threaten

It does **not** fire gate condition 2. Condition 2 is "one arm wins
everywhere", and the two objectives have *different* everywhere-winners: Wave 1
measured `blackout+propainter` winning 27 of 36 separable rate-first cells
against `downsample+realesrgan`'s 6. That is a sharper version of Wave 1's
headline — the objectives agreed on 1 of 19 cells — not a collapse to a single
global recommendation.

What it does threaten is different, and should be stated in these terms:
**the map may be one-dimensional in the objective rather than two-dimensional
in (content, rate).** If the deployer's objective selects the arm and content
never does, the plan's "content class" axis is empty and the claim has to be
reworded. **That is Wave 2A's question, running in parallel — this section is
evidence for 2A to be read against, not a finding that pre-empts it.**

---

## Batches

### Batch 1 — 5 `none` controls at svtav1 QP43

Filter: `--filter restorer=none --filter degradation=downsample --filter codec=svtav1`,
which dry-ran to exactly 5 entries.

| | |
|---|---|
| entries | 5 |
| results written | 5 |
| `Error running experiment` lines | 0 |

*(map re-run and bound check pending evaluation completion)*

## Batch 1 post-check — completed 2026-08-03

Run count 926 → 931. The map is **unchanged** (420 arms, 84 cells, 19/46
separable, 18 of 19 naming `downsample+realesrgan`), which is expected: a `none`
control is not an arm, so it feeds the ladder rather than the map.

**The check earned its keep.** Ladder coverage moved the wrong way — 12 → 11
operating points — and adding controls cannot legitimately reduce coverage.
Diagnosis: a control-matching bug in `tools/build_operating_map.py`, not
anything about batch 1. Controls were keyed on `(component, transport,
block_size, shrink_amount)`, which does not separate degradation *strength*;
`dancing`@QP43 has three `blur` runs at `blur_kernel` 7/15/31 collapsing to one
key, so arms were scored against another run's damage while keeping their own
bitrate. Fixed on the map branch (commit `4d5fc29`) by matching on the whole
config minus the fill, with regression tests.

Corrected ladder figures: **81 arms across 11 operating points**, and **3**
deployable arms ≥1 JND off the ladder, not 10. Gate verdict and all three
pre-registered bounds unchanged.

**Bounds check for batch 1 itself:** the 5 new controls were pre-registered to
change no map cell (they add no arm) and to raise ladder coverage. The first
held; the second did not, for the reason above. Once the fix landed, the five
controls do what was intended. No bound is recorded as fired, because the
prediction failed through a tool defect rather than through the data.
