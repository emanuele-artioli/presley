"""Output-contract tests for the Real-HAT-GAN conditioned restorer.

Mirrors ``tests/test_restoration_bsrgan.py``: adaptive-pyramid bookkeeping
with a stub 2x upsampler (no weights/GPU required for the shape/dtype
contract). The fp16 ban is a pure-Python ValueError check against
``_instantiate_real_hat_gan_upsampler`` and does not load weights when the
weights file is absent — we only assert the guard fires.
"""
import cv2
import numpy as np
import pytest
import torch

restoration = pytest.importorskip(
    "presley.restoration",
    reason="needs the pinned `presley` conda env (instantir/basicsr/realesrgan)",
)

pytestmark = pytest.mark.gpu


def _double(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)


def test_upscale_real_hat_gan_adaptive_returns_original_resolution_and_dtype(rng):
    block_size = 8
    frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    downscale_map = np.array([[1, 0], [0, 0]], dtype=np.int32)

    restored = restoration.upscale_real_hat_gan_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )

    assert restored.shape == frame.shape
    assert restored.dtype == frame.dtype


def test_upscale_real_hat_gan_adaptive_handles_no_degraded_blocks(rng):
    block_size = 8
    frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    downscale_map = np.zeros((2, 2), dtype=np.int32)

    restored = restoration.upscale_real_hat_gan_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )

    assert restored.shape == frame.shape
    assert restored.dtype == frame.dtype


def test_upscale_real_hat_gan_adaptive_requires_an_upsample_fn():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    downscale_map = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(TypeError):
        restoration.upscale_real_hat_gan_adaptive(frame, downscale_map, 8)


def test_upscale_real_hat_gan_adaptive_matches_realesrgan_given_the_same_stub():
    block_size = 8
    frame = np.random.default_rng(20260728).integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    downscale_map = np.array([[1, 0], [0, 0]], dtype=np.int32)

    hat_out = restoration.upscale_real_hat_gan_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )
    realesrgan_out = restoration.upscale_realesrgan_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )
    np.testing.assert_array_equal(hat_out, realesrgan_out)


def test_real_hat_gan_rejects_fp16():
    """Half precision NaNs Softmax on this arch — ban it like NAFNet fp16."""
    with pytest.raises(ValueError, match="float32"):
        restoration._instantiate_real_hat_gan_upsampler(
            torch.device("cpu"), fp32=False
        )
