"""Standalone, cheap confirmation experiment for a referee comment (TOMM revision).

The RESEARCH_LOG dead-end registry asserts "addroi side data is never read by
libx265" -- reached by reading documentation/code, never by an actual measured
encode (grep confirms addroi has never been invoked anywhere in this repo;
src/presley/components/roi.py's x264_addroi/x265_addroi branch just raises
NotImplementedError). This script measures it directly instead:

  (A) addroi-hinted encode: FG union-bbox region gets a strong negative
      qoffset (more bits/better quality); the complement of that box, tiled as
      four non-overlapping rectangles (so there is no ambiguity about
      overlap-priority semantics), gets a positive qoffset (fewer bits/worse
      quality). Fixed QP=25 (libx265 -x265-params qp=25 -- rate control off,
      so any FG/BG differentiation can only come from the ROI side data, not
      from bitrate-target search).
  (B) control encode: identical, minus the -vf addroi chain. Same fixed QP=25.
  (C) presley_qp comparison point: apply this project's own QP-mapping
      degradation (filter_frame_qp, src/presley/degradation.py) using the same
      video's real removability scores, then encode the degraded pixels at the
      same fixed QP=25 (encode_video_x265_qp). This is the mechanism the paper
      already uses and already knows works ("elvis"/"presley_ai" family) --
      the comparison point that addroi is being measured against.

Quality is measured as region-restricted PSNR (foreground = real per-frame UFO
mask, not the encode-time bbox) against the pristine reference frames, same
convention as src/presley/evaluation/masked.py's _masked_psnr. Using the true
per-frame mask (not the static bbox used to shape the addroi hint) is
deliberately generous to addroi: the union bbox is a superset of every frame's
real FG, so if the hint does anything at all, it should show here.

Not wired into experiments.yaml/runner.py -- deliberately a one-off. This
worktree carries no dataset/ or cache/ (per CLAUDE.md), so this script reads
those read-only from the main checkout and writes only under --out-dir
(default: scratch/addroi_confirm, gitignored).

Usage:
    python scripts/addroi_confirmation.py
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

WORKTREE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(WORKTREE_SRC))

from presley.degradation import filter_frame_qp  # noqa: E402
from presley.encode_utils import encode_video_x265_qp, save_frames_as_video  # noqa: E402

MAIN_REPO = Path("/home/itec/emanuele/presley")
VIDEO = "bear"
WIDTH, HEIGHT = 640, 360
BLOCK_SIZE = 16
ALPHA, BETA = 0.5, 0.5
BASE_QP = 25
QP_RANGE = 15
FRAMERATE = 24.0

COMMAND_LOG = []


def run(cmd, **kwargs):
    COMMAND_LOG.append(" ".join(shlex.quote(c) for c in cmd))
    print("+", COMMAND_LOG[-1])
    return subprocess.run(cmd, check=True, **kwargs)


def masked_psnr(ref, dec, mask):
    ref_f, dec_f = ref.astype(np.float32), dec.astype(np.float32)
    diff = ref_f[mask] - dec_f[mask] if mask is not None and np.any(mask) else ref_f - dec_f
    mse = float(np.mean(diff ** 2)) if diff.size else 0.0
    if mse < 1e-10:
        return 100.0
    return float(min(20 * np.log10(255.0 / np.sqrt(mse)), 100.0))


def load_frames(frames_dir):
    paths = sorted(Path(frames_dir).glob("*.png"))
    return [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]


def load_masks(masks_dir):
    paths = sorted(Path(masks_dir).glob("*.png"))
    return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]


def union_bbox(masks, thresh=127):
    pooled = np.max(np.stack(masks), axis=0)
    ys, xs = np.where(pooled > thresh)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return x1, y1, x2, y2


def build_addroi_filter(x1, y1, x2, y2, w, h, fg_qoffset=-0.5, bg_qoffset=0.4):
    """FG bbox gets fg_qoffset (negative = better quality / more bits). The
    complement is tiled as up to four non-overlapping rectangles (top/bottom
    strips span the full width; left/right strips span only the bbox's row
    range) so the whole frame is partitioned exactly once -- no region overlaps
    another, so there is no dependence on addroi's overlap-priority order.
    clear=1 on the first region resets any pre-existing ROI side data."""
    regions = [f"addroi=x={x1}:y={y1}:w={x2 - x1}:h={y2 - y1}:qoffset={fg_qoffset}:clear=1"]
    if y1 > 0:
        regions.append(f"addroi=x=0:y=0:w={w}:h={y1}:qoffset={bg_qoffset}")
    if y2 < h:
        regions.append(f"addroi=x=0:y={y2}:w={w}:h={h - y2}:qoffset={bg_qoffset}")
    if x1 > 0:
        regions.append(f"addroi=x=0:y={y1}:w={x1}:h={y2 - y1}:qoffset={bg_qoffset}")
    if x2 < w:
        regions.append(f"addroi=x={x2}:y={y1}:w={w - x2}:h={y2 - y1}:qoffset={bg_qoffset}")
    return ",".join(regions)


def ffmpeg_encode(frames_pattern, out_path, vf, qp, framerate=FRAMERATE):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", str(framerate), "-i", frames_pattern]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx265", "-preset", "medium", "-x265-params", f"qp={qp}",
            "-pix_fmt", "yuv420p", out_path]
    run(cmd)


def decode_to_frames(video_path):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    COMMAND_LOG.append(" ".join(shlex.quote(c) for c in cmd) + "  # (decode for measurement)")
    proc = subprocess.run(cmd, capture_output=True, check=True)
    frame_bytes = HEIGHT * WIDTH * 3
    arr = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH, 3)
    return [f.copy() for f in arr]


def region_psnr(frames_ref, frames_dec, masks):
    fg_vals, bg_vals = [], []
    for ref, dec, m in zip(frames_ref, frames_dec, masks):
        fg_mask = m > 127
        bg_mask = ~fg_mask
        fg_vals.append(masked_psnr(ref, dec, fg_mask))
        bg_vals.append(masked_psnr(ref, dec, bg_mask))
    return float(np.mean(fg_vals)), float(np.mean(bg_vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "scratch" / "addroi_confirm"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = f"{VIDEO}_{WIDTH}x{HEIGHT}"
    frames_dir = MAIN_REPO / "cache" / key / "reference_frames"
    masks_dir = MAIN_REPO / "cache" / key / "ufo_masks"
    removability_path = MAIN_REPO / "cache" / f"{key}_bs{BLOCK_SIZE}" / f"removability_a{ALPHA:.2f}_b{BETA:.2f}.npy"

    frames = load_frames(frames_dir)
    masks = load_masks(masks_dir)
    removability = np.load(removability_path)
    n = len(frames)
    frames_pattern = str(frames_dir / "%05d.png")
    print(f"Loaded {n} frames from {frames_dir}")

    x1, y1, x2, y2 = union_bbox(masks)
    bbox_frac = (x2 - x1) * (y2 - y1) / (WIDTH * HEIGHT)
    print(f"FG union bbox: x[{x1},{x2}) y[{y1},{y2})  ({(x2 - x1) * (y2 - y1)} / {WIDTH * HEIGHT} px = {bbox_frac:.1%})")

    vf = build_addroi_filter(x1, y1, x2, y2, WIDTH, HEIGHT)
    print("addroi filter chain:", vf)

    hinted_path = out_dir / "addroi_hinted.mp4"
    control_path = out_dir / "addroi_control.mp4"
    presley_qp_path = out_dir / "presley_qp.mp4"
    degraded_intermediate = out_dir / "presley_qp_degraded.mkv"

    print("\n== Encoding A: addroi-hinted ==")
    ffmpeg_encode(frames_pattern, str(hinted_path), vf, BASE_QP)
    print("\n== Encoding B: control (no addroi) ==")
    ffmpeg_encode(frames_pattern, str(control_path), None, BASE_QP)

    print("\n== Degrading frames with filter_frame_qp (presley_qp mechanism) ==")
    degraded_frames = []
    for i in range(n):
        degraded, _ = filter_frame_qp(frames[i], removability[i], BLOCK_SIZE, qp_range=QP_RANGE, base_qp=BASE_QP)
        degraded_frames.append(degraded)
    save_frames_as_video(degraded_frames, str(degraded_intermediate), FRAMERATE, lossless=True, codec="ffv1")
    print("\n== Encoding C: presley_qp (fixed QP, same base QP) ==")
    encode_video_x265_qp(str(degraded_intermediate), str(presley_qp_path), FRAMERATE, BASE_QP)
    COMMAND_LOG.append(f"# presley_qp degradation: filter_frame_qp(frame, removability_score, "
                        f"block_size={BLOCK_SIZE}, qp_range={QP_RANGE}, base_qp={BASE_QP}) per frame, "
                        f"then encode_video_x265_qp(..., qp={BASE_QP})")

    results = {
        "video": VIDEO, "width": WIDTH, "height": HEIGHT, "num_frames": n,
        "framerate": FRAMERATE, "base_qp": BASE_QP,
        "fg_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "frac_of_frame": bbox_frac},
        "addroi_filter_chain": vf,
        "encodes": {},
    }
    for name, path in [("addroi_hinted", hinted_path), ("addroi_control", control_path), ("presley_qp", presley_qp_path)]:
        size_bytes = os.path.getsize(path)
        duration = n / FRAMERATE
        bitrate = size_bytes * 8 / duration
        dec_frames = decode_to_frames(str(path))
        fg_psnr, bg_psnr = region_psnr(frames, dec_frames, masks)
        results["encodes"][name] = {
            "file_size_bytes": size_bytes,
            "actual_bitrate_bps": bitrate,
            "fg_psnr_db": fg_psnr,
            "bg_psnr_db": bg_psnr,
            "fg_minus_bg_psnr_db": fg_psnr - bg_psnr,
        }
        print(f"{name}: size={size_bytes}B bitrate={bitrate:.0f}bps "
              f"FG-PSNR={fg_psnr:.3f}dB BG-PSNR={bg_psnr:.3f}dB (FG-BG={fg_psnr - bg_psnr:+.3f}dB)")

    hinted = results["encodes"]["addroi_hinted"]
    control = results["encodes"]["addroi_control"]
    results["addroi_vs_control_delta"] = {
        "fg_psnr_delta_db": hinted["fg_psnr_db"] - control["fg_psnr_db"],
        "bg_psnr_delta_db": hinted["bg_psnr_db"] - control["bg_psnr_db"],
        "bitrate_delta_bps": hinted["actual_bitrate_bps"] - control["actual_bitrate_bps"],
        "bitrate_delta_pct": (hinted["actual_bitrate_bps"] - control["actual_bitrate_bps"]) / control["actual_bitrate_bps"] * 100.0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "commands.log", "w") as f:
        f.write("\n".join(COMMAND_LOG) + "\n")
    print("\nWrote", out_dir / "results.json", "and", out_dir / "commands.log")


if __name__ == "__main__":
    main()
