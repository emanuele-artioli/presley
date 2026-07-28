"""DC-VSR weight layout + inference gate (Q8).

Kept free of InstantIR/diffusers imports so the wiring smoke tests stay in
the fast pytest tier. ``presley.restoration.restore_downsampled_with_dc_vsr``
re-exports the directory I/O wrapper that calls into this gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

# HF `Janghyeok/dc-vsr` layout (inspected 2026-07-28): UNet EMA only.
_DEFAULT_DC_VSR_WEIGHTS_DIR = Path('weights') / 'dc-vsr'
_DC_VSR_UNET_WEIGHTS_REL = Path('unet_ema') / 'diffusion_pytorch_model.safetensors'
_DC_VSR_UNET_CONFIG_REL = Path('unet_ema') / 'config.json'

_DC_VSR_MISSING_INFERENCE = (
    'DC-VSR inference is not available: Hugging Face repo Janghyeok/dc-vsr '
    'contains only UNet EMA weights (UNetSpatioTemporalConditionModel / '
    'diffusers 0.29.2 safetensors), not a runnable pipeline. Missing: '
    '(1) public SAP/TAP/DSSAG sampling code from arXiv 2502.03502, '
    '(2) VAE + noise scheduler + CLIP/image-encoder bundle the UNet was '
    'trained against (config `_name_or_path` points at a private '
    'svd-vsr checkout), (3) any documented inference entrypoint. '
    'Weights alone are not enough to restore frames. Download layout:\n'
    '  hf download Janghyeok/dc-vsr --local-dir weights/dc-vsr\n'
    'Isolate future deps in a separate conda env — do not upgrade the '
    'pinned `presley` env. See docs/EXPERIMENTS_QUEUED.md Q8.'
)


def resolve_dc_vsr_weights_dir(
    weights_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the local DC-VSR weight tree (HF ``Janghyeok/dc-vsr`` layout)."""
    path = (
        Path(weights_dir).expanduser()
        if weights_dir is not None
        else _DEFAULT_DC_VSR_WEIGHTS_DIR.expanduser()
    )
    unet = path / _DC_VSR_UNET_WEIGHTS_REL
    if unet.is_file():
        return path
    alt = Path.cwd() / 'weights' / 'dc-vsr'
    if weights_dir is None and (alt / _DC_VSR_UNET_WEIGHTS_REL).is_file():
        return alt
    raise FileNotFoundError(
        f'DC-VSR weights not found at {unet}. Download with:\n'
        '  hf download Janghyeok/dc-vsr --local-dir weights/dc-vsr\n'
        '(unset http(s)_proxy if the Hub 403s). Expected files: '
        f'{_DC_VSR_UNET_WEIGHTS_REL} and {_DC_VSR_UNET_CONFIG_REL}.'
    )


def require_dc_vsr_inference_ready(
    *,
    fp32: bool = True,
    weights_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Validate fp16 policy + weights, then refuse to pretend inference works.

    Returns the resolved weights dir only if a future implementation unlocks
    the ``RuntimeError`` below; callers today never get a successful return.
    """
    if not fp32:
        raise ValueError(
            'DC-VSR must run in float32 (fp32=True). Half precision is banned '
            'preemptively (SVD-class SpatioTemporal Softmax attention — same '
            'NaN class as Real-HAT on this stack). Re-evaluate only after a '
            'real forward pass exists; see tests/test_restoration_dc_vsr.py.'
        )
    resolved = resolve_dc_vsr_weights_dir(weights_dir)
    raise RuntimeError(
        _DC_VSR_MISSING_INFERENCE + f' Resolved weights dir: {resolved}.'
    )
