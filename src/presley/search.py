"""Search runner for the PRESLEY pipeline."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from presley.config import PresleyConfig
from presley.pipeline import run_presley

GRID_RESULTS_DIR = Path("grid_search_results")
RANDOM_RESULTS_DIR = Path("random_search_results")

_ASPECT_RATIO_MIN = 5 / 4
_ASPECT_RATIO_MAX = 21 / 9

PARAMETER_GRID: Dict[str, List[Any]] = {
    "reference_video": [
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/bear.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/camel.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/dance-jump.mp4",
    ],
    "experiment_dir": ["experiment"],
    "width": [640],
    "height": [360],
    "block_size": [8, 16],
    "shrink_amount": [0.25, 0.5],
    "enable_fvmd": [False],
    "quality_factor": [1.2],
    "removability_alpha": [0.25, 0.75],
    "removability_smoothing_beta": [0.25, 0.75],
    "strength_maps_target_bitrate": [10000, 25000],
    "propainter_resize_ratio": [1.0],
    "propainter_ref_stride": [20],
    "propainter_neighbor_length": [4],
    "propainter_subvideo_length": [30],
    "propainter_mask_dilation": [4],
    "propainter_raft_iter": [20],
    "propainter_fp16": [True],
    "propainter_parallel_chunk_length": [None],
    "propainter_chunk_overlap": [None],
    "e2fgvi_ref_stride": [10],
    "e2fgvi_neighbor_stride": [5],
    "e2fgvi_num_ref": [-1],
    "e2fgvi_mask_dilation": [4],
    "realesrgan_denoise_strength": [1.0],
    "realesrgan_tile": [0],
    "realesrgan_tile_pad": [10],
    "realesrgan_pre_pad": [0],
    "realesrgan_parallel_chunk_length": [None],
    "realesrgan_per_device_workers": [1],
    "instantir_cfg": [7.0],
    "instantir_creative_start": [1.0],
    "instantir_preview_start": [0.0],
    "instantir_batch_size": [3],
    "generate_opencv_benchmarks": [False],
    "metric_stride": [1],
    "fvmd_stride": [5],
    "fvmd_max_frames": [None],
    "fvmd_processes": [16],
    "fvmd_early_stop_delta": [0.01],
    "fvmd_early_stop_window": [50],
    "vmaf_stride": [1],
}

def _slugify(parts: Dict[str, Any]) -> str:
    sanitized_segments: List[str] = []
    for key, value in parts.items():
        text = str(value)
        text = text.replace("/", "-").replace("\\", "-")
        text = text.replace(" ", "-").replace(".", "p")
        text = "".join(ch for ch in text if ch.isalnum() or ch in {"-", "_"})
        sanitized_segments.append(f"{key}-{text}")
    return "_".join(sanitized_segments)

def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)

def _extract_metric_sections(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    metric_sections: Dict[str, Any] = {}
    for key, value in analysis_data.items():
        if isinstance(value, dict) and "foreground" in value and "background" in value:
            metric_sections[key] = value
    return metric_sections

def _is_valid_overrides(overrides: Dict[str, Any]) -> bool:
    width = overrides.get("width")
    height = overrides.get("height")
    block_size = overrides.get("block_size")

    if isinstance(width, int) and width <= 0: return False
    if isinstance(height, int) and height <= 0: return False
    if isinstance(block_size, int) and block_size <= 0: return False
    if isinstance(width, int) and isinstance(block_size, int) and width % block_size != 0: return False
    if isinstance(height, int) and isinstance(block_size, int) and height % block_size != 0: return False
    if isinstance(width, int) and isinstance(height, int):
        if height == 0: return False
        aspect_ratio = width / height
        if aspect_ratio < _ASPECT_RATIO_MIN or aspect_ratio > _ASPECT_RATIO_MAX: return False
    return True

def run_experiment_list(combinations: List[Dict[str, Any]], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    varying_keys = [key for key, values in PARAMETER_GRID.items() if len(values) > 1]
    summary: List[Dict[str, Any]] = []
    total_runs = len(combinations)

    for run_index, overrides in enumerate(combinations, start=1):
        slug_source = {key: overrides[key] for key in varying_keys if key in overrides}
        slug = _slugify(slug_source) or f"run_{run_index:03d}"
        final_dir = results_dir / slug

        config_data = asdict(PresleyConfig())
        config_data["minimal_figures"] = True
        config_data.update(overrides)
        config_data["experiment_dir"] = str(final_dir)
        config = PresleyConfig(**config_data)

        experiment_path = Path(config.experiment_dir)
        _ensure_clean_dir(experiment_path)
        experiment_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[{run_index}/{total_runs}] Running configuration: {overrides}")
        results = run_presley(config)

        analysis_path = experiment_path / "analysis_results.json"
        analysis_data: Dict[str, Any] = results
        if analysis_path.exists():
            with analysis_path.open("r") as fp:
                analysis_data = json.load(fp)
            analysis_data.setdefault("parameters", {}).setdefault("derived", {})["experiment_dir"] = str(experiment_path)
            analysis_data["parameters"]["derived"]["analysis_results_path"] = str(analysis_path)
            analysis_data["experiment_dir"] = str(experiment_path)
            analysis_data["analysis_results_path"] = str(analysis_path)
            analysis_data["search_label"] = slug
            with analysis_path.open("w") as fp:
                json.dump(analysis_data, fp, indent=4)

        metrics_snapshot = _extract_metric_sections(analysis_data)
        timings_snapshot = analysis_data.get("execution_times_seconds", {})
        video_metadata = {
            "video_name": analysis_data.get("video_name"),
            "video_length_seconds": analysis_data.get("video_length_seconds"),
            "video_framerate": analysis_data.get("video_framerate"),
            "video_resolution": analysis_data.get("video_resolution"),
            "block_size": analysis_data.get("block_size"),
            "target_bitrate_bps": analysis_data.get("target_bitrate_bps"),
        }

        summary.append({
            "label": slug,
            "overrides": overrides,
            "analysis_results_path": str(analysis_path),
            "metrics": metrics_snapshot,
            "execution_times_seconds": timings_snapshot,
            "video_metadata": video_metadata,
        })

    summary_path = results_dir / "runs_summary.json"
    with summary_path.open("w") as fp:
        json.dump(summary, fp, indent=4)
    print(f"\nCompleted {total_runs} runs. Summary saved to {summary_path}")

def run_grid() -> None:
    grid_keys = list(PARAMETER_GRID.keys())
    grid_values = [PARAMETER_GRID[key] for key in grid_keys]

    valid_combinations: List[Dict[str, Any]] = []
    skipped = 0

    for combo in itertools.product(*grid_values):
        overrides = {key: combo[idx] for idx, key in enumerate(grid_keys)}
        if not _is_valid_overrides(overrides):
            skipped += 1
            continue
        valid_combinations.append(overrides)

    if not valid_combinations:
        print("No valid parameter combinations found.")
        return
    if skipped:
        print(f"Skipping {skipped} combinations that failed sanity checks.")
    
    run_experiment_list(valid_combinations, GRID_RESULTS_DIR)

def run_random(num_samples: int) -> None:
    grid_keys = list(PARAMETER_GRID.keys())
    grid_values = [PARAMETER_GRID[key] for key in grid_keys]
    
    all_combinations = list(itertools.product(*grid_values))
    random.shuffle(all_combinations)
    
    valid_combinations: List[Dict[str, Any]] = []
    
    for combo in all_combinations:
        if len(valid_combinations) >= num_samples:
            break
        overrides = {key: combo[idx] for idx, key in enumerate(grid_keys)}
        if _is_valid_overrides(overrides):
            valid_combinations.append(overrides)
            
    if not valid_combinations:
        print("No valid parameter combinations found.")
        return
    
    run_experiment_list(valid_combinations, RANDOM_RESULTS_DIR)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run PRESLEY search experiments.")
    parser.add_argument("--mode", type=str, choices=["grid", "random"], required=True, help="Search mode")
    parser.add_argument("--samples", type=int, default=10, help="Number of random samples for random mode")
    args = parser.parse_args()

    if args.mode == "grid":
        run_grid()
    elif args.mode == "random":
        run_random(args.samples)

if __name__ == "__main__":
    main()
