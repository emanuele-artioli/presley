"""W3: measure restoration speed under conditions that are recorded, repeatedly.

The corpus cannot support a speed claim. `docs/W3_TIMING_SPLIT.md` shows why:
ProPainter's 10-40x split at 640x360 is silent CPU fallback, the device was
never recorded, and which population a historical run belongs to is
unrecoverable. Every pre-2026-08-05 restoration timing is therefore a mixture,
and mixtures are not measurements.

This harness re-measures. It does **not** re-run experiments -- it replays the
restoration step alone, from a completed run's own transmitted artefacts:

    results/<hash>/encoded_degraded.mp4   the degraded video the client receives
    results/<hash>/strength_maps.npz      the side-channel that conditions the restorer
    results/<hash>/result.json            the config, so the call matches exactly

which is precisely the input the restorer had in the original run. Nothing is
written under `results/`; output frames go to a scratch directory and are
deleted. The DB is never touched, so a timing run cannot alter a citable record.

Why replay rather than re-run: repeat trials of one config produce one
experiment hash, so `presley-run` would either skip them or overwrite a
citable result directory. Replaying isolates the axis under study -- the
restoration cost -- from encoding and evaluation, which is what the speed claim
is about anyway.

**Every trial records its device.** A trial that resolves to CPU is reported as
such and never averaged with GPU trials; the mixture is the original defect.

Usage:
    python tools/timing_campaign.py --db results/presley.db --trials 3 --dry-run
    python tools/timing_campaign.py --db results/presley.db --trials 3 \\
        --json docs/w3_timings.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# A configuration whose repeat trials scatter this much is reported as "not
# measurable" rather than as a number (plan W3's verification rule).
CV_UNMEASURABLE = 0.3
# Restorers this harness knows how to replay. Others are skipped by name rather
# than half-replayed with default parameters, which would time a different
# computation than the run performed.
REPLAYABLE = ("realesrgan", "nafnet", "propainter")


@dataclass
class Trial:
    hash: str
    label: str
    width: int
    height: int
    trial: int
    seconds: float
    frames: int
    devices: List[str]
    gpu_free_mb: Optional[List[int]] = None
    max_used_frac: Optional[float] = None

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else float("nan")


@dataclass
class Configuration:
    """One (arm, resolution) cell and its repeat trials."""
    label: str
    width: int
    height: int
    hash: str
    trials: List[Trial] = field(default_factory=list)

    @property
    def devices(self) -> List[str]:
        return sorted({d for t in self.trials for d in t.devices})

    @property
    def mixed_devices(self) -> bool:
        """Trials that did not all run on the same device cannot be pooled --
        that mixture is the defect this whole workstream exists to undo."""
        return len({tuple(sorted(t.devices)) for t in self.trials}) > 1

    def summary(self) -> Dict[str, Any]:
        fps = [t.fps for t in self.trials]
        median = statistics.median(fps) if fps else float("nan")
        cv = (statistics.stdev(fps) / statistics.mean(fps)
              if len(fps) >= 2 and statistics.mean(fps) else 0.0)
        measurable = bool(fps) and cv <= CV_UNMEASURABLE and not self.mixed_devices
        return {
            "label": self.label, "resolution": f"{self.width}x{self.height}",
            "hash": self.hash, "n_trials": len(fps),
            "fps_median": median, "fps_min": min(fps) if fps else None,
            "fps_max": max(fps) if fps else None, "cv": cv,
            "devices": self.devices, "mixed_devices": self.mixed_devices,
            "measurable": measurable,
            "verdict": ("ok" if measurable else
                        "NOT MEASURABLE (mixed devices)" if self.mixed_devices else
                        f"NOT MEASURABLE (CV {cv:.2f} > {CV_UNMEASURABLE})"),
        }


def pick_configurations(db: str, arms: Sequence[str],
                        resolutions: Sequence[Tuple[int, int]]) -> List[Configuration]:
    """One completed run per (arm, resolution), preferring the most recent.

    Which specific run is replayed does not matter for the speed axis as long as
    it is held fixed across trials -- the harness times the same input every
    time, so between-trial variance is the machine, not the content.
    """
    import sqlite3

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT hash, video, width, height, degradation, restorer, inpainter"
        "  FROM runs WHERE n_invariant_failures = 0 AND has_metrics = 1"
    ).fetchall()
    con.close()

    best: Dict[Tuple[str, int, int], Configuration] = {}
    for r in rows:
        fill = r["restorer"] or r["inpainter"]
        if fill not in REPLAYABLE:
            continue
        label = f'{r["degradation"]}+{fill}'
        key = (label, r["width"], r["height"])
        if label not in arms or (r["width"], r["height"]) not in resolutions:
            continue
        run_dir = os.path.join(os.path.dirname(db), r["hash"])
        if not (os.path.isfile(os.path.join(run_dir, "encoded_degraded.mp4"))
                and os.path.isfile(os.path.join(run_dir, "strength_maps.npz"))):
            continue
        if key not in best:
            best[key] = Configuration(label=label, width=r["width"],
                                      height=r["height"], hash=r["hash"])
    return sorted(best.values(), key=lambda c: (c.label, c.width))


def replay_once(run_dir: str, out_dir: str) -> Tuple[float, int, List[str]]:
    """Run the restoration exactly as the component did, and time only that.

    Returns (seconds, frames, devices). Imports are local because this module is
    also used with --dry-run on a box where torch may not be importable.
    """
    import cv2
    import numpy as np

    from presley.components.presley_ai import _restorer_strength_map
    from presley.encode_utils import load_frames_from_video
    from presley.concurrency import last_resolved_devices, reset_resolved_devices
    from presley.degradation import upsample_block_mask
    from presley.sidechannel import load_level_masks

    with open(os.path.join(run_dir, "result.json")) as fh:
        cfg = json.load(fh)["config"]
    restorer = cfg.get("restorer") or cfg.get("inpainter")
    params = cfg.get("restorer_params") or {}
    block_size, width, height = cfg["block_size"], cfg["width"], cfg["height"]

    # `load_frames_from_video` and not cv2.VideoCapture: the transmitted videos
    # are SVT-AV1, which this OpenCV build decodes to zero frames without
    # raising. The component uses this helper, so replaying with anything else
    # would also risk decoding the stream differently than the run did.
    frames = load_frames_from_video(os.path.join(run_dir, "encoded_degraded.mp4"))
    cap = cv2.VideoCapture(os.path.join(run_dir, "encoded_degraded.mp4"))
    framerate = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if not frames:
        raise ValueError(f"decoded 0 frames from {run_dir}/encoded_degraded.mp4")

    smaps = load_level_masks(os.path.join(run_dir, "strength_maps.npz"))
    frames_dir = os.path.join(out_dir, "in")
    restored_dir = os.path.join(out_dir, "out")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(restored_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(frames_dir, f"{i:05d}.png"), frame)

    smap_r = _restorer_strength_map(np.asarray(smaps), restorer,
                                    cfg.get("degradation", ""), params)

    reset_resolved_devices()
    # The clock starts after every input is on disk, so it measures restoration
    # and not the harness's own decode -- the component's own timer does the
    # same, which keeps these numbers comparable to the recorded ones.
    started = time.time()
    if restorer == "realesrgan":
        from presley.restoration import restore_downsampled_with_realesrgan
        restore_downsampled_with_realesrgan(
            frames_dir, restored_dir, smap_r, block_size,
            denoise_strength=params.get("denoise_strength", 1.0),
            tile=params.get("tile", 0), tile_pad=params.get("tile_pad", 10),
            fp32=params.get("fp32", False))
    elif restorer == "nafnet":
        from presley.restoration import restore_with_nafnet_adaptive
        restore_with_nafnet_adaptive(
            frames_dir, restored_dir, smap_r, block_size,
            width=int(params.get("width", 64)),
            weights_path=params.get("weights_path"),
            fp32=bool(params.get("fp32", True)),
            local=bool(params.get("local", True)))
    elif restorer == "propainter":
        from presley.restoration import inpaint_with_propainter
        masks_dir = os.path.join(out_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        for i in range(len(smaps)):
            m = upsample_block_mask((smaps[i] > 0).astype(np.uint8),
                                    block_size, width, height) * 255
            cv2.imwrite(os.path.join(masks_dir, f"{i:05d}.png"), m)
        pp_keys = ("ref_stride", "neighbor_length", "subvideo_length",
                   "raft_iter", "fp16", "resize_ratio")
        inpaint_with_propainter(
            frames_dir, masks_dir, restored_dir, width, height, framerate,
            mask_dilation=0, **{k: params[k] for k in pp_keys if k in params})
    else:
        raise ValueError(f"not replayable: {restorer}")
    elapsed = time.time() - started
    return elapsed, len(frames), last_resolved_devices()


def run_campaign(configs: Sequence[Configuration], trials: int,
                 scratch: str) -> None:
    from presley.gpu_utils import acquisition_conditions

    for cfg in configs:
        run_dir = cfg.hash if os.path.isdir(cfg.hash) else os.path.join(
            "results", cfg.hash)
        for trial in range(1, trials + 1):
            out_dir = tempfile.mkdtemp(prefix=f"w3_{cfg.hash[:8]}_", dir=scratch)
            try:
                seconds, frames, devices = replay_once(run_dir, out_dir)
            finally:
                # Scratch only. Nothing under results/ is ever removed here.
                shutil.rmtree(out_dir, ignore_errors=True)
            cond = acquisition_conditions()
            cfg.trials.append(Trial(
                hash=cfg.hash, label=cfg.label, width=cfg.width,
                height=cfg.height, trial=trial, seconds=seconds, frames=frames,
                devices=devices, gpu_free_mb=cond.get("gpu_free_mb"),
                max_used_frac=cond.get("max_used_frac")))
            print(f"  {cfg.label:26} {cfg.width}x{cfg.height} trial {trial}/{trials}: "
                  f"{frames / seconds:7.3f} fps on {devices or ['?']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="results/presley.db")
    ap.add_argument("--trials", type=int, default=3,
                    help="repeat trials per configuration (plan W3: >=3)")
    ap.add_argument("--arms", nargs="*", default=[
        "downsample+realesrgan", "blur+nafnet", "freeze+propainter",
        "blackout+propainter"])
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="list the configurations that would be measured")
    args = ap.parse_args()

    resolutions = [(640, 360), (1280, 720), (1920, 1080)]
    configs = pick_configurations(args.db, args.arms, resolutions)

    print(f"{len(configs)} configurations, {args.trials} trials each "
          f"= {len(configs) * args.trials} restorations")
    for cfg in configs:
        print(f"  {cfg.label:26} {cfg.width}x{cfg.height}  replaying {cfg.hash}")
    missing = [(a, f"{w}x{h}") for a in args.arms for w, h in resolutions
               if not any(c.label == a and (c.width, c.height) == (w, h)
                          for c in configs)]
    if missing:
        print("\nNo replayable run on disk for (reported, not silently skipped):")
        for arm, res in missing:
            print(f"  {arm:26} {res}")
    if args.dry_run:
        return

    run_campaign(configs, args.trials, args.scratch)

    print(f"\n{'arm':26} {'res':>10} {'n':>2} {'fps median':>11} {'CV':>6}  "
          f"{'devices':16} verdict")
    print("-" * 100)
    summaries = [c.summary() for c in configs]
    for s in summaries:
        print(f"{s['label']:26} {s['resolution']:>10} {s['n_trials']:>2} "
              f"{s['fps_median']:>11.3f} {s['cv']:>6.3f}  "
              f"{','.join(s['devices']):16} {s['verdict']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"summaries": summaries,
                       "trials": [asdict(t) for c in configs for t in c.trials]},
                      fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
