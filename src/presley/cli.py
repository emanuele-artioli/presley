import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, Any

from presley.config import PresleyConfig
from presley.pipeline import run_presley

def _load_config_from_cli() -> PresleyConfig:
    parser = argparse.ArgumentParser(description="Run the PRESLEY pipeline with configurable parameters.")
    parser.add_argument("--config", type=str, help="Path to a JSON file containing PresleyConfig fields.")
    parser.add_argument("--reference-video", type=str, help="Path to the input reference video.")
    parser.add_argument("--width", type=int, help="Target frame width.")
    parser.add_argument("--height", type=int, help="Target frame height.")
    parser.add_argument("--block-size", type=int, help="Processing block size.")
    parser.add_argument("--shrink-amount", type=float, help="Shrink amount for ELVIS.")
    parser.add_argument("--quality-factor", type=float, help="Quality factor for target bitrate calculation.")
    parser.add_argument("--target-bitrate", type=int, help="Override target bitrate in bits per second")
    parser.add_argument("--removability-alpha", type=float, help="Alpha parameter for removability scoring.")
    parser.add_argument("--removability-smoothing-beta", type=float, help="Smoothing beta for removability scoring.")
    parser.add_argument("--encode-preset", type=str, help="FFmpeg preset for encoding (e.g., medium, fast, slow).")
    parser.add_argument("--encode-pix-fmt", type=str, help="Pixel format for encoding (e.g., yuv420p).")
    parser.add_argument("--generate-opencv-benchmarks", dest="generate_opencv_benchmarks", action="store_true", help="Enable OpenCV baseline generation.")
    parser.add_argument("--disable-opencv-benchmarks", dest="generate_opencv_benchmarks", action="store_false", help="Disable OpenCV baseline generation.")
    parser.set_defaults(generate_opencv_benchmarks=None)
    parser.add_argument("--metric-stride", type=int, help="Stride for PSNR/SSIM/LPIPS metrics.")
    parser.add_argument("--fvmd-stride", type=int, help="Stride for FVMD computation.")
    parser.add_argument("--fvmd-max-frames", type=int, help="Maximum frames for FVMD computation.")
    parser.add_argument("--fvmd-processes", type=int, help="Number of FVMD worker processes.")
    parser.add_argument("--fvmd-early-stop-delta", type=float, help="Early stop delta for FVMD.")
    parser.add_argument("--fvmd-early-stop-window", type=int, help="Early stop window for FVMD.")
    parser.add_argument("--vmaf-stride", type=int, help="Stride for VMAF computation.")

    args = parser.parse_args()

    config_data: Dict[str, Any] = asdict(PresleyConfig())

    if args.config:
        with open(args.config, "r") as f:
            file_config = json.load(f)
        config_data.update(file_config)

    overrides = {
        "reference_video": args.reference_video,
        "width": args.width,
        "height": args.height,
        "block_size": args.block_size,
        "shrink_amount": args.shrink_amount,
        "quality_factor": args.quality_factor,
        "target_bitrate_override": args.target_bitrate,
        "removability_alpha": args.removability_alpha,
        "removability_smoothing_beta": args.removability_smoothing_beta,
        "encode_preset": args.encode_preset,
        "encode_pix_fmt": args.encode_pix_fmt,
        "metric_stride": args.metric_stride,
        "fvmd_stride": args.fvmd_stride,
        "fvmd_max_frames": args.fvmd_max_frames,
        "fvmd_processes": args.fvmd_processes,
        "fvmd_early_stop_delta": args.fvmd_early_stop_delta,
        "fvmd_early_stop_window": args.fvmd_early_stop_window,
        "vmaf_stride": args.vmaf_stride,
    }

    for key, value in overrides.items():
        if value is not None:
            config_data[key] = value

    if args.generate_opencv_benchmarks is not None:
        config_data["generate_opencv_benchmarks"] = args.generate_opencv_benchmarks

    return PresleyConfig(**config_data)


def main() -> None:
    config = _load_config_from_cli()
    results = run_presley(config)
    path = results.get("analysis_results_path")
    if path:
        print(f"\nFinal analysis JSON: {path}")

if __name__ == "__main__":
    main()
