# addroi confirmation (referee response, TOMM revision)

**Verdict: CONFIRMED.** ffmpeg's `addroi` ROI hint has zero effect on libx265's
fixed-QP output for this video — the hinted and control encodes are **byte-for-byte
identical** (same MD5, `cmp` reports no difference). PRESLEY's own `presley_qp`
QP-mapping, measured on the same video at the same nominal QP, does produce a
real FG/BG quality differential (+2.56 dB FG over BG, vs addroi's -0.43 dB in
both its "hinted" and control encodes — i.e., no differentiation at all).

This replaces the RESEARCH_LOG dead-end entry's documentation-based claim
("addroi side data is never read by libx265... AQ is variance-based and
mask-agnostic (control condition ≈ baseline)", `68e8b6bb11d0dd9e62a67aef/RESEARCH_LOG.md`
lines ~628-631) with an actual measurement. `grep -rn addroi` across the repo
previously found only that prose claim plus the `NotImplementedError` stub in
`src/presley/components/roi.py:95-96` — addroi had never been invoked.

## Setup

- Video: `bear`, 640x360, 82 frames, 24 fps (already cached at
  `cache/bear_640x360/` in the main checkout — this worktree carries no
  `dataset/`/`cache/`, so the script reads those read-only from
  `/home/itec/emanuele/presley/cache/`).
- FG region: real per-frame UFO masks (`cache/bear_640x360/ufo_masks/`),
  already computed by the existing pipeline.
- Fixed QP throughout (never VBR, per the project's hard rule that VBR rate
  control absorbs ROI/degradation deltas): `qp=25` for every encode below.
- Script: `scripts/addroi_confirmation.py` (this worktree). Not wired into
  `experiments.yaml`/the runner — a standalone, one-off confirmation, as scoped.

### (A) addroi-hinted encode

FG union bounding box across all 82 frames (the smallest box containing every
frame's real UFO foreground): `x=[66,360) y=[116,355)`, 294x239 px = 30.5% of
the 640x360 frame — deliberately generous to addroi, since a per-frame mask is
always tighter than its own union.

Region hints: the FG box gets `qoffset=-0.5` (strong "spend more bits here").
Its complement is tiled as four non-overlapping rectangles (top/bottom
full-width strips, left/right strips spanning the box's row range — an exact
partition of the frame, so there is no dependence on addroi's overlap-priority
semantics) at `qoffset=+0.4` ("spend fewer bits here"). `clear=1` on the first
region resets any pre-existing side data.

```
ffmpeg -hide_banner -loglevel error -y -framerate 24.0 \
  -i cache/bear_640x360/reference_frames/%05d.png \
  -vf addroi=x=66:y=116:w=294:h=239:qoffset=-0.5:clear=1,addroi=x=0:y=0:w=640:h=116:qoffset=0.4,addroi=x=0:y=355:w=640:h=5:qoffset=0.4,addroi=x=0:y=116:w=66:h=239:qoffset=0.4,addroi=x=360:y=116:w=280:h=239:qoffset=0.4 \
  -c:v libx265 -preset medium -x265-params qp=25 -pix_fmt yuv420p addroi_hinted.mp4
```

### (B) control encode

Identical command, minus `-vf`:

```
ffmpeg -hide_banner -loglevel error -y -framerate 24.0 \
  -i cache/bear_640x360/reference_frames/%05d.png \
  -c:v libx265 -preset medium -x265-params qp=25 -pix_fmt yuv420p addroi_control.mp4
```

### (C) presley_qp comparison point

This project's own QP-mapping degradation (`filter_frame_qp`,
`src/presley/degradation.py:514-554`), driven by this video's real
removability scores (`alpha=0.5, beta=0.5, block_size=16`, already cached),
applied per-frame, then encoded at the *same* fixed QP=25 as (A)/(B):

```
# per frame: filter_frame_qp(frame, removability_score, block_size=16, qp_range=15, base_qp=25)
# -> lossless ffv1 intermediate, then:
ffmpeg -hide_banner -loglevel error -y -i presley_qp_degraded.mkv \
  -c:v libx265 -preset medium -x265-params qp=25 -pix_fmt yuv420p presley_qp.mp4
```

## Measurement

FG/BG quality = region-restricted PSNR of the decoded output against the
pristine reference frames, masked by the **real per-frame UFO mask** (not the
static bbox used to shape the addroi hint — same convention as
`src/presley/evaluation/masked.py:_masked_psnr`, and deliberately the harder,
fairer bar for addroi to clear since the bbox is a superset of the true mask).

| Encode | File size | Bitrate | FG-PSNR | BG-PSNR | FG − BG |
|---|---|---|---|---|---|
| addroi_hinted | 1,287,013 B | 3.01 Mbps | 32.849 dB | 33.278 dB | **-0.429 dB** |
| addroi_control | 1,287,013 B | 3.01 Mbps | 32.849 dB | 33.278 dB | **-0.429 dB** |
| presley_qp | 2,658,851 B | 6.23 Mbps | 31.489 dB | 28.928 dB | **+2.560 dB** |

`addroi_hinted.mp4` and `addroi_control.mp4`: identical MD5
(`f3f834d05e4a9ee872d7e88e490b8444`), `cmp` reports no byte difference. The
`-vf addroi=...` filter chain changed nothing about the encoded bitstream —
not a small effect below JND (0.5 dB PSNR per this project's own `compare.py`
JND table), but exactly zero. libx265 CQP mode ignores the ROI side data
entirely, as the dead-end entry asserted, now with a direct measurement behind
it.

`presley_qp` at the same nominal encoder QP: FG beats BG by 2.56 dB, comfortably
above the 0.5 dB PSNR JND — a real, measured FG/BG differential, confirming
this project's own QP-mapping mechanism (as opposed to addroi) actually moves
quality between regions. Note the bitrate side-effect is *not* a bit-savings
result here: this standalone test applies the pixel-domain DCT-quantization
degradation and then re-encodes at a single fixed QP directly (block-boundary
discontinuities from spatially-varying quantization add high-frequency
content x265 has to spend more bits coding), which is a different pipeline
shape from production `presley_qp` runs (`src/presley/components/roi.py`),
which target a bitrate after degrading. The FG/BG quality differentiation is
the claim this experiment was scoped to check, and it holds; the bitrate
number here is reported for completeness, not as a Goal-1 bit-relocation claim.

## Files

- `scripts/addroi_confirmation.py` — the standalone script (committed).
- `scratch/addroi_confirm/` — encoded outputs, `results.json`, `commands.log`
  (gitignored via `scratch/`, not committed — regenerate with
  `python scripts/addroi_confirmation.py`).
