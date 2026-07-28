"""Wiring / honesty-gate tests for the DC-VSR restorer stub (Q8).

Imports only ``presley.dc_vsr`` (no InstantIR / GPU) so this stays in the
fast pytest tier. Full directory I/O for ``restore_downsampled_with_dc_vsr``
is exercised lightly via a temp PNG once the gate is unlocked — today the
gate always raises, which is the behaviour under test.
"""
from pathlib import Path

import numpy as np
import pytest

from presley.dc_vsr import (
    _DC_VSR_UNET_WEIGHTS_REL,
    require_dc_vsr_inference_ready,
    resolve_dc_vsr_weights_dir,
)


def _touch_fake_unet(weights_root: Path) -> Path:
    unet = weights_root / _DC_VSR_UNET_WEIGHTS_REL
    unet.parent.mkdir(parents=True, exist_ok=True)
    unet.write_bytes(b'not-a-real-safetensors')
    return weights_root


def test_dc_vsr_rejects_fp16():
    """Half precision banned preemptively (Softmax NaN class, like Real-HAT)."""
    with pytest.raises(ValueError, match='float32'):
        require_dc_vsr_inference_ready(fp32=False, weights_dir='/nonexistent')


def test_dc_vsr_missing_weights_raises_file_not_found(tmp_path):
    missing = tmp_path / 'empty-dc-vsr'
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match='hf download Janghyeok/dc-vsr'):
        resolve_dc_vsr_weights_dir(missing)


def test_dc_vsr_weights_present_still_blocks_missing_inference(tmp_path):
    """Honesty rule: weights alone must not pretend the restorer works."""
    root = _touch_fake_unet(tmp_path / 'dc-vsr')
    with pytest.raises(RuntimeError, match='inference is not available'):
        require_dc_vsr_inference_ready(fp32=True, weights_dir=root)


def test_dc_vsr_resolve_finds_unet_layout(tmp_path):
    root = _touch_fake_unet(tmp_path / 'dc-vsr')
    assert resolve_dc_vsr_weights_dir(root) == root


def test_restore_downsampled_with_dc_vsr_raises_cleanly(tmp_path):
    """Directory wrapper validates I/O then hits the same RuntimeError gate."""
    # Lazy import: restoration pulls InstantIR at module scope. Prefer the
    # thin gate above for CI; this path only runs when the pinned env is
    # importable (same as other restoration contract tests).
    restoration = pytest.importorskip(
        'presley.restoration',
        reason='needs the pinned `presley` conda env (instantir/basicsr)',
    )
    import cv2

    frames_dir = tmp_path / 'in'
    out_dir = tmp_path / 'out'
    frames_dir.mkdir()
    out_dir.mkdir()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_dir / '00000.png'), frame)
    maps = np.zeros((1, 2, 2), dtype=np.int32)
    weights = _touch_fake_unet(tmp_path / 'dc-vsr')

    with pytest.raises(RuntimeError, match='SAP/TAP/DSSAG'):
        restoration.restore_downsampled_with_dc_vsr(
            str(frames_dir),
            str(out_dir),
            maps,
            block_size=8,
            weights_dir=weights,
            fp32=True,
        )
