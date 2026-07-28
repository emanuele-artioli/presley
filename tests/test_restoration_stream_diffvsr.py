"""Output-contract tests for Stream-DiffVSR adaptive / map paste (stubbed).

Mirrors ``tests/test_restoration_real_hat_gan.py``: adaptive-pyramid
bookkeeping with a stub 2× upsampler, plus directory restore that pastes
untouched blocks. Marked ``gpu`` only because ``presley.restoration`` imports
InstantIR at module level (same as Real-HAT / NAFNet restoration tests).
"""
import cv2
import numpy as np
import pytest

restoration = pytest.importorskip(
    "presley.restoration",
    reason="needs the pinned `presley` conda env (instantir/basicsr/realesrgan)",
)

pytestmark = pytest.mark.gpu


def _double(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)


def test_upscale_stream_diffvsr_adaptive_returns_original_resolution_and_dtype(rng):
    block_size = 8
    frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    downscale_map = np.array([[1, 0], [0, 0]], dtype=np.int32)

    restored = restoration.upscale_stream_diffvsr_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )

    assert restored.shape == frame.shape
    assert restored.dtype == frame.dtype


def test_upscale_stream_diffvsr_adaptive_matches_realesrgan_given_the_same_stub():
    block_size = 8
    frame = np.random.default_rng(20260728).integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    downscale_map = np.array([[1, 0], [0, 0]], dtype=np.int32)

    sdv_out = restoration.upscale_stream_diffvsr_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )
    realesrgan_out = restoration.upscale_realesrgan_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )
    np.testing.assert_array_equal(sdv_out, realesrgan_out)


def test_restore_downsampled_with_stream_diffvsr_stub_pastes_untouched(tmp_path, rng):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(in_dir / "00000.png"), frame)
    downscale_maps = np.array([[[1, 0], [0, 0]]], dtype=np.int32)

    restoration.restore_downsampled_with_stream_diffvsr(
        str(in_dir),
        str(out_dir),
        downscale_maps,
        block_size=8,
        upsample_fn=_double,
    )
    out = cv2.imread(str(out_dir / "00000.png"))
    assert out is not None
    assert out.shape == frame.shape
    # Untouched blocks (map==0) must match the transmitted frame exactly.
    np.testing.assert_array_equal(out[0:8, 8:16], frame[0:8, 8:16])
    np.testing.assert_array_equal(out[8:16, 0:8], frame[8:16, 0:8])
    np.testing.assert_array_equal(out[8:16, 8:16], frame[8:16, 8:16])


def test_restore_downsampled_with_stream_diffvsr_copies_when_no_degradation(tmp_path, rng):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(in_dir / "00000.png"), frame)
    downscale_maps = np.zeros((1, 2, 2), dtype=np.int32)

    restoration.restore_downsampled_with_stream_diffvsr(
        str(in_dir), str(out_dir), downscale_maps, block_size=8
    )
    loaded = cv2.imread(str(out_dir / "00000.png"))
    np.testing.assert_array_equal(loaded, frame)
