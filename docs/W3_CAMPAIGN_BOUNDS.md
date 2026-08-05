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
