"""Output-contract tests for the BSRGAN conditioned restorer.

`presley.restoration` unconditionally imports `instantir` at module level
(it's the shared home for every restoration backend, not just this one), so
it can only be *imported* inside the full pinned `presley` conda env -- CI's
fast tier installs CPU torch plus only the handful of pure-Python deps the
tested modules need (see .github/workflows/ci.yml's comment), never the
forked model repos. `pytest.importorskip` turns "not installed" into a skip
at collection time rather than an ImportError that would break the whole
suite, so this file is inert on CI and exercised for real wherever the pinned
env is available (this host's `presley` conda env has it).

No real BSRGAN weights are loaded here -- `upsample_fn` is a deterministic
stand-in 2x resize. The thing worth checking without a GPU is the
stage-contract: the adaptive pyramid produces an image at the original
resolution with the original dtype, and it shares the exact block-level
bookkeeping already exercised (informally) by realesrgan, since both now
delegate to the same `_adaptive_block_pyramid_upscale` core.
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
    """Deterministic stand-in 2x upsampler -- no model weights required to
    exercise the block-level bookkeeping around it."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)


def test_upscale_bsrgan_adaptive_returns_original_resolution_and_dtype(rng):
    """Same contract as `upscale_realesrgan_adaptive`: whatever the per-block
    downscale factors were, the restored image comes back at the *original*
    resolution and dtype -- that's what lets `restore_frames_bsrgan` composite
    it back against the transmitted frame without a resize."""
    block_size = 8
    frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    # 2x2 block grid; top-left block was downsampled 2x (map=1), rest untouched.
    downscale_map = np.array([[1, 0], [0, 0]], dtype=np.int32)

    restored = restoration.upscale_bsrgan_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )

    assert restored.shape == frame.shape
    assert restored.dtype == frame.dtype


def test_upscale_bsrgan_adaptive_handles_no_degraded_blocks(rng):
    """A downscale map that's all zeros (nothing degraded) is the routine's
    edge case: max_factor collapses to 1, so the loop must still run at least
    once and return a same-shape image rather than dividing by zero or
    short-circuiting."""
    block_size = 8
    frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    downscale_map = np.zeros((2, 2), dtype=np.int32)

    restored = restoration.upscale_bsrgan_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )

    assert restored.shape == frame.shape
    assert restored.dtype == frame.dtype


def test_upscale_bsrgan_adaptive_requires_an_upsample_fn():
    """Unlike Real-ESRGAN (which falls back to a bundled CLI/subprocess),
    BSRGAN has no such fallback wired up in this environment -- calling it
    without a model must fail loudly rather than silently no-op."""
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    downscale_map = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(TypeError):
        restoration.upscale_bsrgan_adaptive(frame, downscale_map, 8)


def test_upscale_bsrgan_adaptive_matches_realesrgan_given_the_same_stub():
    """Both restorers now delegate to the shared `_adaptive_block_pyramid_upscale`
    core -- given an identical stub 2x upsampler, they must produce identical
    output for identical input. This is the "drop-in, not a redesign" contract
    the task asked for, made concrete."""
    block_size = 8
    frame = np.random.default_rng(20260724).integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    downscale_map = np.array([[1, 0], [0, 0]], dtype=np.int32)

    bsrgan_out = restoration.upscale_bsrgan_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )
    realesrgan_out = restoration.upscale_realesrgan_adaptive(
        frame, downscale_map, block_size, upsample_fn=_double
    )

    np.testing.assert_array_equal(bsrgan_out, realesrgan_out)
