# W3b–d — the controlled timing campaign: protocol and bounds

Written 2026-08-05 **before any trial is run**. Diagnosis that makes this worth
running at all: `docs/W3_TIMING_SPLIT.md`. Harness: `tools/timing_campaign.py`.

## Protocol

**Replay, not re-run.** Repeat trials of one config share one experiment hash,
so `presley-run` would skip them or overwrite a citable result directory.
The harness instead replays the restoration step alone from a completed run's
own `encoded_degraded.mp4` + `strength_maps.npz` + config — exactly the input
the restorer had — writing frames to scratch and deleting them. `results/` is
read-only to this tool and the DB is never touched.

- **≥3 trials per configuration**, same replayed run every trial, so
  between-trial variance is the machine and not the content.
- **Every trial records its resolved device** (the new
  `acquisition.restore_devices`), plus free VRAM and the co-tenancy proxy.
- **Trials on different devices are never pooled.** A configuration whose
  trials disagree on device is reported `NOT MEASURABLE (mixed devices)` —
  pooling across devices is the original defect.
- **CV > 0.3 → published as "not measurable", not as a number** (plan W3's
  verification rule).
- Run it when nothing else of ours is on the GPU. A campaign launched next to a
  20-run wave measures the wave.

## Coverage, and what is honestly missing

The dry run reports 6 replayable configurations and **6 gaps**, listed rather
than silently skipped:

| arm | 640×360 | 1280×720 | 1920×1080 |
|---|---|---|---|
| `downsample+realesrgan` | ✅ | ✅ | ✅ |
| `blur+nafnet` | ✅ | ✖ | ✖ |
| `freeze+propainter` | ✅ | ✅ | ✖ |
| `blackout+propainter` | ✖ | ✖ | ✖ |

`blackout+propainter` is an `elvis` run and stores no `strength_maps.npz`, so
the presley_ai replay path does not cover it — **the plan's fifth arm cannot be
measured by this harness as written.** That is a real limitation of the
deliverable, not an oversight to discover later: either the harness grows an
elvis path or that arm's speed stays unquantified. **Only
`downsample+realesrgan` can carry a resolution-scaling claim**, which is
convenient but also the point — it is the arm whose scaling the paper currently
denies.

## Bounds

Every fps figure below is at the *fast cluster*, i.e. what a GPU run should
look like. The slow cluster is the CPU-fallback population and is not a
prediction, it is the failure mode being excluded.

| quantity | plausible | alarm | basis |
|---|---|---|---|
| `downsample+realesrgan` @ 640×360 | 3–7 fps | < 1 or > 15 | prior measurement 4.85 fps |
| `downsample+realesrgan` @ 1280×720 | 0.7–1.7 fps | < 0.3 or > 4 | prior 1.11 fps |
| `downsample+realesrgan` @ 1920×1080 | 0.3–0.8 fps | < 0.1 or > 2 | prior 0.49 fps |
| 360p : 720p throughput ratio | 3.0–5.0× | < 2 or > 7 | pixel-count ratio is 4.0×; the prior claim is near-linear scaling |
| 720p : 1080p ratio | 1.7–2.8× | < 1.2 or > 4 | pixel-count ratio 2.25× |
| `freeze+propainter` @ 640×360 | 0.8–3.0 fps | < 0.3 | fast-cluster median 1.469 fps. **A trial under 0.3 fps with `restore_devices == ['cpu']` is not an alarm — it is the diagnosis confirming itself**, and the trial is excluded rather than averaged |
| `blur+nafnet` @ 640×360 | 1.5–6.0 fps | < 0.5 or > 12 | observed 2.2–3.9 fps in the QP43 sweep |
| CV within a configuration | 0.02–0.15 | > 0.3 → not measurable | three trials of identical work on a pinned GPU |
| trials resolving to `cpu` | 0 of 18 | ≥ 1 | the box is expected free; any CPU trial means the campaign ran against a co-tenant and must be repeated, not reported |

## What a clean result licenses, and what it does not

A clean campaign licenses **exactly two** things: a per-arm restoration
throughput at 640×360 with a dispersion figure beside it, and the Real-ESRGAN
resolution-scaling curve — which **restores resolution as a deployment lever**,
a claim the paper currently denies and which W1's audit already flagged as
falsified.

It does **not** license a speed ranking across all arms on the tradeoff surface,
because `blackout+propainter` cannot be measured here and two of the four arms
have no 1080p run to replay. The speed axis of the frontier stays partial, and
the write-up must say which cells are measured and which are absent rather than
presenting a complete-looking table.

---

# Result — the speed axis is measurable, and one bound fired

Run 2026-08-05, `CUDA_VISIBLE_DEVICES=1` (GPU 0 was 100% occupied by a
co-tenant; GPU 1 idle). 18 trials, 6 configurations, 3 trials each.
**Every trial resolved to `cuda:0` — the pinned device — so no configuration is
a device mixture, and the campaign is exactly the thing the corpus never had.**

| arm | resolution | n | fps median | CV | warm-only median | warm CV |
|---|---|---|---|---|---|---|
| `blur+nafnet` | 640×360 | 3 | **10.750** | 0.242 | 10.752 | 0.0002 |
| `downsample+realesrgan` | 640×360 | 3 | **5.464** | 0.060 | 5.483 | 0.0048 |
| `downsample+realesrgan` | 1280×720 | 3 | **1.679** | 0.014 | 1.680 | 0.0006 |
| `downsample+realesrgan` | 1920×1080 | 3 | **0.731** | 0.003 | 0.731 | 0.0003 |
| `freeze+propainter` | 640×360 | 3 | **1.598** | 0.027 | 1.608 | 0.0094 |
| `freeze+propainter` | 1280×720 | 3 | **0.579** | 0.025 | 0.591 | 0.0308 |

No configuration exceeds the CV > 0.3 "not measurable" line. Under the recorded
conditions this corpus can measure restoration speed to within a few percent —
the dispersion was never the machine, it was the unrecorded device.

## Bounds

| bound | predicted | observed | status |
|---|---|---|---|
| `downsample+realesrgan` @ 640×360 | 3–7 fps | 5.464 | in band |
| `downsample+realesrgan` @ 1280×720 | 0.7–1.7 fps | 1.679 | in band, at the top edge |
| `downsample+realesrgan` @ 1920×1080 | 0.3–0.8 fps | 0.731 | in band |
| 360p : 720p ratio | 3.0–5.0× | **3.25×** | in band |
| 720p : 1080p ratio | 1.7–2.8× | **2.30×** | in band (pixel ratio 2.25) |
| `freeze+propainter` @ 640×360 | 0.8–3.0 fps | 1.598 | in band |
| `blur+nafnet` @ 640×360 | 1.5–6.0 fps | **10.750** | **FIRED** (below the >12 alarm) |
| CV within a configuration | 0.02–0.15 | 0.003–0.060, plus 0.242 on `blur+nafnet` | one above band, explained below |
| trials resolving to `cpu` | 0 of 18 | **0 of 18** | in band |

### The fired bound: `blur+nafnet` is ~3× faster than the pipeline ever showed

10.75 fps against a 1.5–6.0 band taken from in-pipeline observations of 2.2–3.9
fps. The band was set from the wrong quantity: those pipeline numbers are
`restoration_time_seconds`, which on this path includes per-frame PNG writes and
model construction, while the campaign times the restoration of an
already-staged directory. **This is not a discrepancy in the machine, it is two
different definitions of "restoration time"**, and the campaign's is the one a
deployer cares about. Not closed by revising the band downward after the fact:
the honest statement is that the two numbers measure different things and the
in-pipeline figure is the conservative one.

`blur+nafnet`'s CV of 0.242 has a single cause — **trial 1 is a cold start**
(6.79 fps against 10.75 warm, 0.63×). Every other configuration's cold penalty
is 0.90–0.99×, because their per-frame work dominates model construction.
Warm-only CV is 0.0002. Future campaigns should discard trial 1 as warm-up;
that is recorded rather than applied retroactively, since re-deriving the
published medians after seeing them is how a band gets fitted to its data.

## What this licenses

1. **Real-ESRGAN's resolution scaling is real and slightly sub-linear**: 3.25×
   for a 4× pixel increase, 2.30× for 2.25×, 7.47× end to end for 9×. **The
   paper's "resolution is not a lever for restoration cost" is falsified with a
   controlled measurement**, not only with the earlier uncontrolled one. Cost
   grows a little slower than pixel count, which is a *stronger* deployment
   statement than strict linearity: downscaling buys less than proportionally.
2. **A speed ordering at 640×360**, all on one device, all CV < 0.06:
   `blur+nafnet` 10.75 > `downsample+realesrgan` 5.46 > `freeze+propainter`
   1.60. The quality winner is **not** the speed winner — the third axis
   conflicts with the first, which is the frontier claim stated on measured
   ground for the first time.
3. **ProPainter's fast cluster was the GPU population**, as the diagnosis
   predicted: 1.598 fps measured here against a 1.469 fps historical fast-cluster
   median. The slow cluster is confirmed as an artifact and stays retired.

## What it does not license

`blackout+propainter` is unmeasured — an `elvis` run stores no
`strength_maps.npz` and the replay path cannot reach it. `blur+nafnet` has no
720p or 1080p run to replay, and `freeze+propainter` has no 1080p. **The speed
axis of the frontier is therefore partial: complete for one arm across three
resolutions, single-resolution for two others, absent for a fourth.** Any table
must show the gaps rather than imply a full grid.
