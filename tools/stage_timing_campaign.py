#!/usr/bin/env python3
"""W1e: measure the per-stage cost of each arm, with repeats, on a pinned device.

`stage_times_seconds` gives the breakdown but only one sample per config, on a
shared GPU. A cost *claim* needs repeats under recorded conditions, and repeats
cannot go through `presley-run`: repeat trials of one config collapse to one
experiment hash, so the runner would either skip them or overwrite a citable
result. `_`-prefixed keys are excluded from the hash, so that escape hatch does
not work either.

So this calls the component functions directly into a scratch directory, three
times per cell, and never writes under `results/`. Same principle as
`tools/timing_campaign.py`, which replays the restoration step alone; this
covers the whole pipeline stage by stage.

**Every trial records its device.** A trial that resolves to CPU is reported as
such and never averaged with GPU trials. That mixture is the original defect
that made every pre-2026-08-05 restoration timing unusable, and finer-grained
accounting does not fix it -- only refusing to average does.

Cells that scatter more than CV 0.3 across their repeats are reported as "not
measurable" rather than as a number, matching the existing campaign's rule.

Bounds, stated before reading (revised from the plan after the first
measurement, with the reason recorded there): restoration dominates;
`select` 0.005-0.05 s at an 80x45 grid; `encode` 1-20 s at 360p. An arm whose
`select` exceeds its `encode` is an alarm.

Usage:
    python tools/stage_timing_campaign.py --trials 3 --out docs/w1e_stage_timings.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import statistics
import sys
import tempfile
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

CV_UNMEASURABLE = 0.30

# One cell per arm the article compares. Each is (label, component, overrides).
BASE = {
    "video": None, "width": 640, "height": 360, "codec": "svtav1",
    "codec_params": {"preset": "8", "qp": 50}, "target_bitrate": 0,
}
BLOCKY = {
    "block_size": 8, "alpha": 0.5, "beta": 0.5, "shrink_amount": 0.25,
    "fg_protect": True, "composite_output": True,
}

ARMS = [
    ("baseline (SVT-AV1)", "baselines", {}),
    ("elvis blackout + ProPainter", "elvis",
     {**BLOCKY, "removal_mode": "blackout", "inpainter": "propainter"}),
    ("presley downsample + Real-ESRGAN", "presley_ai",
     {**BLOCKY, "degradation": "downsample", "restorer": "realesrgan",
      "restorer_params": {}}),
    ("presley freeze + ProPainter", "presley_ai",
     {**BLOCKY, "degradation": "freeze", "restorer": "propainter",
      "restorer_params": {}}),
]

VIDEOS = ("bear", "camel", "dog")


def resolve_device() -> str:
    try:
        import torch
        return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def run_once(component: str, cfg: dict, data_root: pathlib.Path) -> dict:
    from presley.components.baselines import run_baseline
    from presley.components.elvis import run_elvis
    from presley.components.presley_ai import run_presley_ai

    fn = {"baselines": run_baseline, "elvis": run_elvis,
          "presley_ai": run_presley_ai}[component]

    scratch = tempfile.mkdtemp(prefix="stagetiming_")
    try:
        started = time.time()
        result = fn(cfg, str(data_root / "dataset"), scratch, str(data_root / "cache"))
        return {
            "stages": result.get("stage_times_seconds", {}),
            "wall": time.time() - started,
            "device": resolve_device(),
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--videos", nargs="*", default=list(VIDEOS))
    ap.add_argument("--out")
    args = ap.parse_args()

    data_root = pathlib.Path(args.data_root).resolve()
    import presley  # noqa: F401  sqlite3-before-torch

    records = []
    for label, component, overrides in ARMS:
        for video in args.videos:
            cfg = {**BASE, **overrides, "component": component, "video": video}
            trials = []
            for t in range(args.trials):
                try:
                    trials.append(run_once(component, dict(cfg), data_root))
                except Exception as exc:
                    print(f"  FAILED {label} {video} trial {t}: {exc!r}", flush=True)
            if not trials:
                continue

            devices = sorted({t["device"] for t in trials})
            if len(devices) > 1:
                # The original defect: never average across device populations.
                print(f"  MIXED DEVICES for {label} {video}: {devices} -- not averaged",
                      flush=True)
                continue

            stages = sorted({s for t in trials for s in t["stages"]})
            per_stage = {}
            for s in stages:
                vals = [t["stages"].get(s, 0.0) for t in trials]
                mean = statistics.fmean(vals)
                sd = statistics.pstdev(vals)
                cv = sd / mean if mean > 0 else 0.0
                per_stage[s] = {
                    "mean_s": round(mean, 4),
                    "cv": round(cv, 3),
                    "measurable": cv <= CV_UNMEASURABLE,
                }
            walls = [t["wall"] for t in trials]
            records.append({
                "arm": label, "component": component, "video": video,
                "device": devices[0], "trials": len(trials),
                "wall_mean_s": round(statistics.fmean(walls), 3),
                "stages": per_stage,
            })
            print(f"{label:34}{video:8}{devices[0]:8}"
                  f"wall {statistics.fmean(walls):7.1f}s  "
                  + " ".join(f"{s}={per_stage[s]['mean_s']:.2f}" for s in stages),
                  flush=True)

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out} ({len(records)} cells)")

    alarms = [r for r in records
              if r["stages"].get("select", {}).get("mean_s", 0)
              > r["stages"].get("encode", {}).get("mean_s", float("inf"))]
    if alarms:
        print("\n!! ALARM: selection cost exceeds encoding cost in "
              f"{len(alarms)} cell(s) -- investigate before reporting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
