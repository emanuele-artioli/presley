from dataclasses import dataclass
from typing import Optional, List, Union

# ---------------------------------------------------------------------------
# Pipeline and reporting label constants
# ---------------------------------------------------------------------------
APPROACH_BASELINE = "Baseline"
APPROACH_PRESLEY_QP = "PRESLEY QP"
APPROACH_ELVIS = "ELVIS"
APPROACH_ELVIS_CV2 = "ELVIS CV2"
APPROACH_ELVIS_PROP = "ELVIS ProPainter"
APPROACH_ELVIS_E2FGVI = "ELVIS E2FGVI"
APPROACH_PRESLEY_REALESRGAN = "PRESLEY RealESRGAN"
APPROACH_PRESLEY_INSTANTIR = "PRESLEY InstantIR"
APPROACH_PRESLEY_LANCZOS = "PRESLEY Lanczos"
APPROACH_PRESLEY_UNSHARP = "PRESLEY Unsharp"

@dataclass
class PresleyConfig:
    reference_video: str = "davis_test/bear.mp4"
    experiment_dir: Optional[str] = None
    analysis_sample_frames: Optional[int] = None
    strength_maps_use_npz: bool = True
    width: int = 640
    height: int = 360
    block_size: int = 8
    shrink_amount: float = 0.25
    quality_factor: float = 1.2
    target_bitrate_override: Optional[int] = None
    removability_alpha: float = 0.5
    removability_smoothing_beta: float = 0.5
    encode_preset: str = "medium"
    encode_pix_fmt: str = "yuv420p"
    
    # Inpainting Config
    propainter_resize_ratio: float = 1.0
    propainter_ref_stride: int = 20
    propainter_neighbor_length: int = 4
    propainter_subvideo_length: int = 40
    propainter_mask_dilation: int = 4
    propainter_raft_iter: int = 20
    propainter_fp16: bool = True
    propainter_devices: Optional[List[Union[int, str]]] = None
    propainter_parallel_chunk_length: Optional[int] = None
    propainter_chunk_overlap: Optional[int] = None
    e2fgvi_ref_stride: int = 10
    e2fgvi_neighbor_stride: int = 5
    e2fgvi_num_ref: int = -1
    e2fgvi_mask_dilation: int = 4
    e2fgvi_devices: Optional[List[Union[int, str]]] = None
    e2fgvi_parallel_chunk_length: Optional[int] = None
    e2fgvi_chunk_overlap: Optional[int] = None
    
    # Super-resolution / Deblurring Config
    realesrgan_denoise_strength: float = 1.0
    realesrgan_tile: int = 0
    realesrgan_tile_pad: int = 10
    realesrgan_pre_pad: int = 0
    realesrgan_fp32: bool = False
    realesrgan_devices: Optional[List[Union[int, str]]] = None
    realesrgan_parallel_chunk_length: Optional[int] = None
    realesrgan_per_device_workers: int = 1
    instantir_cfg: float = 7.0
    instantir_creative_start: float = 1.0
    instantir_preview_start: float = 0.0
    instantir_seed: Optional[int] = 42
    instantir_devices: Optional[List[Union[int, str]]] = None
    instantir_batch_size: int = 4
    instantir_parallel_chunk_length: Optional[int] = None
    
    # Metrics
    generate_opencv_benchmarks: bool = True
    metric_stride: int = 1
    fvmd_stride: int = 1
    fvmd_max_frames: Optional[int] = None
    fvmd_processes: Optional[int] = None
    fvmd_early_stop_delta: float = 0.002
    fvmd_early_stop_window: int = 50
    vmaf_stride: int = 1
    enable_fvmd: bool = True
