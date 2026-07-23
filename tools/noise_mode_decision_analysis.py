#!/usr/bin/env python
"""Standalone investigation: how does x265 handle noise-corrupted blocks
differently from untouched blocks, at the CTU/mode-decision level?

Context (referee-response item, TOMM revision): PRESLEY's Gaussian-noise
degradation (paired with the ``instantir`` restorer, see
``RESTORER_DEGRADATIONS`` in ``src/presley/components/presley_ai.py``) is
already a confirmed dead end on rate grounds -- a fixed-QP screen documented
in the paper's RESEARCH_LOG showed noise costs +213% to +334% more bits than
the pristine baseline at matched QP, the worst of every degradation measured.
A referee asked *why*, specifically in terms of encoder mode decisions and
rate allocation, not just the bitrate outcome. This script measures that
mechanism directly with x265's own per-frame analysis log (``--csv`` /
``csv-log-level``), which is the finest granularity exposed by the
ffmpeg-linked libx265 build on this host (see "Granularity achieved" below).

This is a ONE-OFF INVESTIGATION SCRIPT, not part of the presley-run pipeline.
It does not touch experiments.yaml or results/. It reuses the project's real
noise-injection code path (``presley.degradation.filter_frame_noise``,
``sel=None`` i.e. the plain round(score*noise_variance)>0 threshold that the
real presley_ai component uses whenever an experiment has no
``shrink_amount``/``fg_protect`` key -- true for every noise entry in
experiments.yaml) and the project's real cached removability scores
(alpha=0.5, beta=0.5) so blocks are selected exactly as the production
pipeline selects them, at the same width/height/block_size/QP already used
in experiments.yaml's noise runs (640x360, block_size 8, qp 32 and 37,
preset medium).

Two comparisons, both at fixed QP (matching the project's hard fixed-QP
rule -- see CLAUDE.md):

1. FULL-FRAME (mixed content, matches the real pipeline exactly): the whole
   reference video vs. the whole video with its real background blocks
   noise-degraded. This reproduces the already-known aggregate bitrate cost
   as a sanity check, but because x265's --csv reports PER-FRAME aggregates
   only (no per-CTU or per-region breakdown), a mixed frame cannot attribute
   mode-decision shifts to the noised blocks specifically -- foreground
   blocks in the same frame dilute the signal.

2. REGION-ISOLATED (attributes the mechanism): a fixed 160x160 crop
   (20x20 blocks) that is 100% background in every single frame of the clip
   per the cached UFO mask (verified by tools/pick_noise_region.py), encoded
   twice at the same QP -- once with noise injected into every block in the
   crop (matching the real per-block noise strength from the removability
   score map) and once left untouched. Because the crop is homogeneous
   (100% of the encoded content is background), the whole-frame CSV
   aggregate IS the region's aggregate -- no dilution, so this is the
   comparison that actually isolates the mechanism.

Granularity achieved
---------------------
x265 exposes per-frame stats via ``--csv``/``csv-log-level`` through the
ffmpeg-linked libx265.so.199 on this host; there is no standalone `x265` CLI
binary installed (checked: `which x265` and `find / -iname '*x265*' -type f`
turn up only unrelated data CSVs, never a binary), and `--pmode` only toggles
the analysis thread-pool feature, it does not emit a per-CU decision log.
So true per-CTU / per-block-coordinate mode logs are NOT available from this
build -- only per-frame aggregates. `csv-log-level=2` gives, per frame: total
Bits, achieved QP, and the percentage of the FINAL CODED AREA at each
partition size (64x64 down to 4x4) broken out by mode (Intra DC/Planar/Ang,
Inter, Skip, Merge), which is exactly what's needed for a frame-level
mode-decision / partition-depth signature -- just not a spatial map within
the frame. The region-isolated crop above is how this script gets around
that limitation without modifying x265: make the "frame" homogeneous so the
frame-level aggregate has nothing else to average against.

Usage
-----
    /home/itec/emanuele/.conda/envs/presley/bin/python \
        tools/noise_mode_decision_analysis.py --out-dir scratch/noise_mode_decision

No GPU needed (x265 encoding only, CPU). Reuses cached reference frames /
removability scores under cache/ (already present for bear and camel at
640x360, block_size 8) -- does not invoke EVCA or the UFO segmentation
model.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from presley.degradation import filter_frame_noise  # noqa: E402

DATASET_CACHE = "/home/itec/emanuele/presley/cache"

# (video, qp) pairs matching the real presley_ai noise experiments in
# experiments.yaml (component: presley_ai, degradation: noise, width 640,
# height 360, block_size 8, alpha 0.5, beta 0.5).
VIDEOS = ["bear", "camel"]
QPS = [32, 37]
WIDTH, HEIGHT, BLOCK_SIZE = 640, 360, 8
ALPHA, BETA = 0.5, 0.5
NOISE_VARIANCE = 50.0  # filter_frame_noise's own default, same as production
PRESET = "medium"

# 160x160 (20x20 blocks) crops verified 100% background in every frame of
# the clip by tools/pick_noise_region.py against the cached UFO masks.
CROPS = {
    "bear": dict(y0=0, x0=320, size=160),
    "camel": dict(y0=200, x0=480, size=160),
}

# Columns in x265's --csv (csv-log-level=2) that report the percentage of
# the FINAL CODED AREA at each partition size, grouped by mode. These sum to
# ~100% within a frame regardless of slice type.
INTRA_COLS = [
    "Intra 64x64 DC", "Intra 64x64 Planar", "Intra 64x64 Ang",
    "Intra 32x32 DC", "Intra 32x32 Planar", "Intra 32x32 Ang",
    "Intra 16x16 DC", "Intra 16x16 Planar", "Intra 16x16 Ang",
    "Intra 8x8 DC", "Intra 8x8 Planar", "Intra 8x8 Ang",
    "4x4",
]
SKIP_COLS = ["Skip 64x64", "Skip 32x32", "Skip 16x16", "Skip 8x8"]
MERGE_COLS = ["Merge 64x64", "Merge 32x32", "Merge 16x16", "Merge 8x8"]
INTER_COLS = ["Inter 64x64", "Inter 32x32", "Inter 16x16", "Inter 8x8"]

# size, in pixels, that each column's mode was finally coded at -- used to
# compute a single "average final CU size" scalar per frame (smaller =>
# more fragmented partitioning => more per-CU overhead).
COL_SIZE = {}
for c in INTRA_COLS:
    if c == "4x4":
        COL_SIZE[c] = 4
    else:
        COL_SIZE[c] = int(c.split()[1].split("x")[0])
for group in (SKIP_COLS, MERGE_COLS, INTER_COLS):
    for c in group:
        COL_SIZE[c] = int(c.split()[1].split("x")[0])


def load_reference_frames(video: str):
    frames_dir = Path(DATASET_CACHE) / f"{video}_{WIDTH}x{HEIGHT}" / "reference_frames"
    paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No cached reference frames at {frames_dir}")
    return [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]


def load_removability_scores(video: str):
    p = (Path(DATASET_CACHE) / f"{video}_{WIDTH}x{HEIGHT}_bs{BLOCK_SIZE}"
         / f"removability_a{ALPHA:.2f}_b{BETA:.2f}.npy")
    if not p.exists():
        raise FileNotFoundError(f"No cached removability scores at {p}")
    return np.load(p)


def save_lossless(frames, out_path: str, framerate: float = 24.0):
    height, width = frames[0].shape[:2]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", str(framerate), "-i", "-",
           "-c:v", "libx265", "-preset", PRESET, "-x265-params", "lossless=1",
           "-pix_fmt", "yuv420p", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"lossless intermediate encode failed for {out_path}")


def encode_qp_with_csv(lossless_path: str, out_video: str, csv_path: str, qp: int):
    """Same encode_video_x265_qp as production (src/presley/encode_utils.py),
    plus x265's own analysis CSV at the max level exposed by this ffmpeg
    build (see module docstring, "Granularity achieved")."""
    x265_params = f"qp={qp}:csv={csv_path}:csv-log-level=2"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", lossless_path,
         "-c:v", "libx265", "-preset", PRESET, "-x265-params", x265_params,
         "-pix_fmt", "yuv420p", out_video],
        check=True)


def _pct_to_float(series: pd.Series) -> pd.Series:
    """x265's --csv writes percentage cells as e.g. ' 36.36%' (object dtype);
    pandas silently accepts that as a column but `.sum()`/`.mean()` over
    multiple such object columns does STRING CONCATENATION, not arithmetic,
    which only surfaces as a cryptic `Could not convert string ... to numeric`
    once you reduce across every row. Strip '%' and whitespace and cast to
    float before any arithmetic."""
    if series.dtype == object:
        return series.astype(str).str.strip().str.rstrip("%").astype(float)
    return series.astype(float)


def summarize_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Type"] = df["Type"].str.strip()
    for c in set(INTRA_COLS + SKIP_COLS + MERGE_COLS + INTER_COLS):
        df[c] = _pct_to_float(df[c])

    total_bits = int(df["Bits"].sum())
    n_frames = len(df)
    mean_qp = float(df["QP"].mean())

    is_i = df["Type"].str.upper().eq("I-SLICE")
    i_bits = int(df.loc[is_i, "Bits"].sum())
    pb_bits = total_bits - i_bits

    pb = df.loc[~is_i]
    out = {
        "n_frames": n_frames,
        "total_bits": total_bits,
        "mean_qp": round(mean_qp, 3),
        "i_frame_count": int(is_i.sum()),
        "i_frame_bits": i_bits,
        "pb_frame_count": int((~is_i).sum()),
        "pb_frame_bits": pb_bits,
        "pb_mean_bits_per_frame": round(pb["Bits"].mean(), 1) if len(pb) else None,
    }

    if len(pb):
        intra_pct = pb[INTRA_COLS].sum(axis=1)
        skip_pct = pb[SKIP_COLS].sum(axis=1)
        merge_pct = pb[MERGE_COLS].sum(axis=1)
        inter_pct = pb[INTER_COLS].sum(axis=1)
        out["pb_intra_pct_mean"] = round(intra_pct.mean(), 3)
        out["pb_skip_pct_mean"] = round(skip_pct.mean(), 3)
        out["pb_merge_pct_mean"] = round(merge_pct.mean(), 3)
        out["pb_inter_pct_mean"] = round(inter_pct.mean(), 3)

        all_cols = INTRA_COLS + SKIP_COLS + MERGE_COLS + INTER_COLS
        weighted_size = sum(pb[c] * COL_SIZE[c] for c in all_cols) / 100.0
        out["pb_avg_final_cu_size_px"] = round(weighted_size.mean(), 3)

    # Same computation restricted to I-frames, for completeness.
    ii = df.loc[is_i]
    if len(ii):
        intra_pct_i = ii[INTRA_COLS].sum(axis=1)
        out["i_intra_pct_mean"] = round(intra_pct_i.mean(), 3)
        weighted_size_i = sum(ii[c] * COL_SIZE[c] for c in INTRA_COLS) / 100.0
        out["i_avg_final_cu_size_px"] = round(weighted_size_i.mean(), 3)

    return out


def run_condition(frames, out_dir: Path, tag: str, qp: int, framerate: float = 24.0):
    lossless = out_dir / f"{tag}_lossless.mkv"
    encoded = out_dir / f"{tag}_qp{qp}.mp4"
    csv_path = out_dir / f"{tag}_qp{qp}.csv"
    # x265's --csv APPENDS to an existing file rather than truncating it, so a
    # rerun against a leftover csv from a previous invocation silently doubles
    # (or worse) the row count -- always start from a clean csv_path/encoded.
    if csv_path.exists():
        csv_path.unlink()
    if encoded.exists():
        encoded.unlink()
    if not lossless.exists():
        save_lossless(frames, str(lossless), framerate)
    encode_qp_with_csv(str(lossless), str(encoded), str(csv_path), qp)
    stats = summarize_csv(str(csv_path))
    if stats["n_frames"] != len(frames):
        raise RuntimeError(
            f"{csv_path}: csv has {stats['n_frames']} rows but input had "
            f"{len(frames)} frames -- stale/appended csv or a dropped frame, "
            f"do not trust these stats")
    stats["file_bytes"] = os.path.getsize(encoded)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="scratch/noise_mode_decision")
    ap.add_argument("--videos", nargs="+", default=VIDEOS)
    ap.add_argument("--qps", nargs="+", type=int, default=QPS)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for video in args.videos:
        frames = load_reference_frames(video)
        scores = load_removability_scores(video)
        crop_cfg = CROPS[video]
        y0, x0, size = crop_cfg["y0"], crop_cfg["x0"], crop_cfg["size"]
        assert y0 % BLOCK_SIZE == 0 and x0 % BLOCK_SIZE == 0 and size % BLOCK_SIZE == 0

        # Full-frame degraded (real production path: sel=None).
        noised_full = []
        for i, frame in enumerate(frames):
            degraded, _ = filter_frame_noise(frame, scores[i], BLOCK_SIZE,
                                              noise_variance=NOISE_VARIANCE, sel=None)
            noised_full.append(degraded)

        # Region-isolated crop: control (pristine) and noised, cropped
        # AFTER degrading the full frame so block alignment / neighbouring
        # context is identical to the full-frame case, just windowed.
        control_crop = [f[y0:y0 + size, x0:x0 + size] for f in frames]
        noised_crop = [f[y0:y0 + size, x0:x0 + size] for f in noised_full]
        # Sanity check: crop must be the block-aligned size we assumed.
        assert control_crop[0].shape[:2] == (size, size)

        video_dir = out_root / video
        video_dir.mkdir(parents=True, exist_ok=True)

        video_results = {}
        for qp in args.qps:
            print(f"=== {video} qp={qp} ===", flush=True)
            full_control = run_condition(frames, video_dir, "full_control", qp)
            full_noised = run_condition(noised_full, video_dir, "full_noised", qp)
            crop_control = run_condition(control_crop, video_dir, "crop_control", qp)
            crop_noised = run_condition(noised_crop, video_dir, "crop_noised", qp)
            video_results[str(qp)] = {
                "full_control": full_control,
                "full_noised": full_noised,
                "crop_control": crop_control,
                "crop_noised": crop_noised,
            }
            print(json.dumps(video_results[str(qp)], indent=2), flush=True)
        results[video] = video_results

    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
