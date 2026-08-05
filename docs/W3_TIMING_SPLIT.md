# W3 — the ProPainter timing split, diagnosed from data already on disk

The plan's W3 step 3 says: *"Diagnose the ProPainter split. If it is co-tenancy,
the fast cluster is the true figure and the slow one is contention. If it is a
code change, find the commit. Do not publish a ProPainter speed number until
this resolves."* Both offered explanations are **wrong**, and the data that
settles it cost no GPU time.

## The DB has no timestamps, but the filesystem does

`result.json` is rewritten by `--backfill-lpips`, so its mtime is not an
acquisition time. The **oldest** file in each `results/<hash>/` directory is,
because the video artefacts are written once and never touched again. Ordering
169 ProPainter runs at 640×360 by that proxy:

| cluster | n | median fps | mtime range |
|---|---|---|---|
| slow | 82 | **0.040** | 2026-07-03 → 2026-07-25 |
| fast | 87 | **1.469** | 2026-07-16 → 2026-08-05 |

**Along the time order there are only 3 cluster changes in 169 runs.** Random
contention would interleave and produce dozens. The split is batch-structured at
the scale of days.

But it is **not one-way**:

```
… 07-12  0.024  slow
  07-16  0.406  FAST     <- change 1
… 07-21  1.237  FAST
  07-24  0.091  slow     <- change 2
… 07-25  0.103  slow
  07-25  1.185  FAST     <- change 3
```

**A code change cannot un-happen and then re-happen.** That rules out the
"find the commit" branch, and the diff confirms it: no ProPainter-relevant
commit lands on 07-24 to be reverted on 07-25. Equally, ordinary co-tenancy
jitter does not hold one machine 30× slow for a solid day and then release it.

## The mechanism is in our own code, and it is silent

`_resolve_device_list` is called with `allow_cpu_fallback=True` at **every**
restorer call site (`src/presley/restoration.py`, 7 call sites), and
`preflight_gpu` does not stop a run when no GPU has enough free VRAM — it
prints `WARNING: no GPU with >= 2000MB free … launching anyway` and continues.
So on a busy box a run does not fail and does not queue. It **falls back to CPU
and completes**, roughly 30× slower, and nothing in `result.json` records which
device it used.

That fits every feature of the split: the 30× ratio, the day-scale blocks, the
reversibility, and the total absence of any config that explains it.

**This is not yet proven** — the device was not recorded, so the 169 historical
runs remain unattributable and no retrofit can change that. It is the mechanism
that predicts all four observations, where neither candidate in the plan
predicts more than one.

## What changed as a result

`result['acquisition']['restore_devices']` now records what the restoration
actually resolved to (`['cuda:0']`, `['cpu']`, …). An empty list means no
restoration resolved a device at all — a `none` control — and is deliberately
distinguishable from CPU, because reporting "unknown" as "cpu" would invent the
fact this record exists to establish. The record is cleared before each dispatch
so one run cannot inherit the previous run's device.

Together with W3a's timestamp and VRAM occupancy, a timing is now attributable
at the moment it is taken. Three tests pin it, including the two ways it could
lie (inheriting a stale device, reporting absence as CPU).

## Consequences for the speed axis

1. **No ProPainter speed number from before 2026-08-05 is publishable.** The
   corpus's ProPainter timings are a mixture of two device populations, and
   which population a given run belongs to is unrecoverable. This retires the
   pooled medians rather than repairing them — the same disposal Wave 2C's
   throughput table received, for the same reason.
2. **The fast cluster is the plausible GPU figure** (median 1.469 fps at
   640×360), but it must be re-measured, not adopted. It is quoted here as an
   expectation for the campaign's bounds, not as a result.
3. **The controlled campaign is still required**, and it is now worth running,
   which it was not before: repeat trials with an unrecorded device would have
   produced the same uninterpretable mixture.
4. **Every other restorer inherits this risk.** Real-ESRGAN's clean resolution
   scaling (4.85 / 1.11 / 0.49 fps at 360p/720p/1080p) was measured under the
   same silent-fallback regime. It is internally consistent and monotone in
   pixel count, which is weak evidence it was all one device, but the campaign
   should re-take it rather than assume so.

## The design question this raises, and does not settle

Silent CPU fallback is the right default for a demo and the wrong one for a
measurement. The cheap fix is not to remove it but to make it **loud** —
`preflight_gpu` already knows when no GPU is free, and a run that is going to
take 30× longer should say so at launch rather than at analysis. Not changed
here: it alters run behaviour on a shared box mid-campaign, and that is a
decision to take deliberately rather than as a side effect of a diagnosis.
