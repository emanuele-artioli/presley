"""Contract tests for NAFNet restoration I/O (no real weights required).

`presley.restoration` imports InstantIR at module level, so this file is
skipped on CI's lean env via importorskip — same pattern as
`tests/test_restoration_bsrgan.py`.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

restoration = pytest.importorskip(
    "presley.restoration",
    reason="needs the pinned `presley` conda env (instantir at module import)",
)

from presley.nafnet_arch import build_gopro_nafnet


def test_nafnet_deblur_bgr_returns_same_shape_and_dtype(tmp_path, rng):
    """Stub random-init NAFNet: output contract is original HxWx3 uint8 BGR."""
    net = build_gopro_nafnet(32, local=False)
    net.eval()
    frame = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    out = restoration._nafnet_deblur_bgr(net, frame, torch.device("cpu"), fp32=True)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_restore_with_nafnet_adaptive_copies_when_no_blur(tmp_path, rng, monkeypatch):
    """Zero blur map must short-circuit without loading weights."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(in_dir / "00000.png"), frame)
    blur_maps = np.zeros((1, 2, 2), dtype=np.int32)

    def _boom(*_a, **_k):
        raise AssertionError("get_nafnet must not be called when blur_maps are all zero")

    monkeypatch.setattr(restoration, "get_nafnet", _boom)
    restoration.restore_with_nafnet_adaptive(str(in_dir), str(out_dir), blur_maps, block_size=8)
    written = list(out_dir.glob("*.png"))
    assert len(written) == 1
    loaded = cv2.imread(str(written[0]))
    np.testing.assert_array_equal(loaded, frame)


def test_restore_with_nafnet_adaptive_pastes_back_untouched_blocks(tmp_path, rng, monkeypatch):
    """After the full-frame forward, blur_map==0 blocks must match the input
    (InstantIR/unsharp contract) — otherwise clean BG is scored as a restorer
    failure."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    # 16x16 → 2x2 blocks of size 8; only top-left is "blurred".
    frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(in_dir / "00000.png"), frame)
    blur_maps = np.array([[[1, 0], [0, 0]]], dtype=np.int32)

    class _Stub:
        def __call__(self, x):
            # Return a constant so any pasted-back region is obvious.
            return torch.zeros_like(x)

    def _fake_get_nafnet(*_a, **_k):
        return _Stub()

    monkeypatch.setattr(restoration, "get_nafnet", _fake_get_nafnet)
    # Bypass real deblur: write a solid color so untouched paste-back is testable.
    def _fake_deblur(_model, frame_bgr, _device, *, fp32=False):
        return np.full_like(frame_bgr, 7)

    monkeypatch.setattr(restoration, "_nafnet_deblur_bgr", _fake_deblur)
    restoration.restore_with_nafnet_adaptive(str(in_dir), str(out_dir), blur_maps, block_size=8)
    out = cv2.imread(str(out_dir / "00000.png"))
    # Top-left 8x8 came from the stub (all 7s); the other three blocks are original.
    np.testing.assert_array_equal(out[0:8, 0:8], np.full((8, 8, 3), 7, dtype=np.uint8))
    np.testing.assert_array_equal(out[0:8, 8:16], frame[0:8, 8:16])
    np.testing.assert_array_equal(out[8:16, 0:8], frame[8:16, 0:8])
    np.testing.assert_array_equal(out[8:16, 8:16], frame[8:16, 8:16])


def test_resolve_nafnet_weights_missing_raises():
    with pytest.raises(FileNotFoundError, match="NAFNet weights"):
        restoration._resolve_nafnet_weights(64, weights_path="/tmp/does-not-exist-nafnet.pth")


def test_get_nafnet_rejects_fp16():
    """Regression: CUDA half() on NAFNet overflowed LayerNorm2d/SCA and
    produced rainbow garbage (Q5 bear BG-PSNR 10.9 vs tx 22.8). Official
    megvii path is float32; we refuse fp32=False rather than silently ruin
    a run."""
    with pytest.raises(ValueError, match="float32"):
        restoration.get_nafnet(torch.device("cpu"), fp32=False)


def test_nafnet_deblur_rejects_fp16(rng):
    net = build_gopro_nafnet(32, local=False)
    net.eval()
    frame = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="fp32"):
        restoration._nafnet_deblur_bgr(net, frame, torch.device("cpu"), fp32=False)
