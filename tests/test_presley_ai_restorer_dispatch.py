"""Dispatch/config-table tests for presley_ai's restorer capability table.

These import only `presley.components.presley_ai`, which -- unlike
`presley.restoration` -- does no module-level heavy-model imports (its own
imports are preprocessing/encode_utils/degradation/sidechannel, all pure
numpy/cv2), so they run in the fast CI tier without a GPU or the pinned
conda env. The actual restoration call (`restore_downsampled_with_bsrgan`) is
imported lazily inside `run_presley_ai`'s dispatch branch precisely so this
stays testable without the heavy stack -- see `tests/test_restoration_bsrgan.py`
for the output-contract coverage of that function.
"""
import numpy as np

from presley.components.presley_ai import (
    RESTORER_DEGRADATIONS,
    INPAINT_DEGRADATIONS,
    _STRENGTH_CLAMP,
    _restorer_strength_map,
)


def test_bsrgan_is_registered_alongside_realesrgan():
    """Second conditioned GAN/CNN restorer for the paper's
    restoration-comparison ablation -- same allowed-degradation set as
    realesrgan (both are conditioned restorers that consume `downsample` plus
    the hole degradations, and interpret the map the same way)."""
    assert "bsrgan" in RESTORER_DEGRADATIONS
    assert RESTORER_DEGRADATIONS["bsrgan"] == ("downsample",) + INPAINT_DEGRADATIONS
    assert RESTORER_DEGRADATIONS["bsrgan"] == RESTORER_DEGRADATIONS["realesrgan"]


def test_bsrgan_rejects_a_degradation_it_cannot_interpret():
    """bsrgan only knows the log2-downscale-factor map; blur/noise use
    different units (sharpening rounds / variance) that would silently
    corrupt its `np.power(2, map)` computation if allowed through."""
    assert "blur" not in RESTORER_DEGRADATIONS["bsrgan"]
    assert "noise" not in RESTORER_DEGRADATIONS["bsrgan"]


def test_bsrgan_strength_map_is_clamped_like_realesrgan():
    """Both restorers read the map as a log2 downscale factor and share the
    same clamp -- the clamp exists so a mis-paired degradation (e.g. noise,
    whose map goes up to noise_variance) can't blow up cv2.resize with an
    astronomical factor (see `_restorer_strength_map`'s caller comment)."""
    raw = np.array([[0, 1, 5, 10]])
    clamped = _restorer_strength_map(raw, "bsrgan", "downsample", {})
    assert clamped.max() <= _STRENGTH_CLAMP["bsrgan"]
    np.testing.assert_array_equal(
        clamped,
        _restorer_strength_map(raw, "realesrgan", "downsample", {}),
    )


def test_bsrgan_hole_degradations_use_the_binary_rounds_recipe():
    """Same recipe as every other conditioned/in-painting restorer for the
    bridge degradations: binary hole map * requested rounds, not the raw
    (irrelevant) transmitted strength value."""
    raw = np.array([[0, 5, 0, 9]])  # freeze/mean_fill maps are binary in practice
    clamped = _restorer_strength_map(raw, "bsrgan", "freeze", {"rounds": 2})
    np.testing.assert_array_equal(clamped, [[0, 2, 0, 2]])


def test_nafnet_is_registered_alongside_instantir():
    """CNN deblur gauge for blur transport (Q5) — same allowed degradations
    as InstantIR (blur + hole fills)."""
    assert "nafnet" in RESTORER_DEGRADATIONS
    assert RESTORER_DEGRADATIONS["nafnet"] == ("blur",) + INPAINT_DEGRADATIONS
    assert RESTORER_DEGRADATIONS["nafnet"] == RESTORER_DEGRADATIONS["instantir"]


def test_nafnet_rejects_downsample():
    """NAFNet is a deblur network; its strength map is blur rounds, not
    log2 downscale factors."""
    assert "downsample" not in RESTORER_DEGRADATIONS["nafnet"]
    assert "noise" not in RESTORER_DEGRADATIONS["nafnet"]


def test_nafnet_strength_map_clamped_like_instantir():
    raw = np.array([[0, 1, 5, 10]])
    clamped = _restorer_strength_map(raw, "nafnet", "blur", {})
    assert clamped.max() <= _STRENGTH_CLAMP["nafnet"]
    np.testing.assert_array_equal(
        clamped,
        _restorer_strength_map(raw, "instantir", "blur", {}),
    )


def test_real_hat_gan_is_registered_alongside_realesrgan():
    """Q4 recent SR GAN — same allowed-degradation set as realesrgan/bsrgan."""
    assert "real_hat_gan" in RESTORER_DEGRADATIONS
    assert RESTORER_DEGRADATIONS["real_hat_gan"] == ("downsample",) + INPAINT_DEGRADATIONS
    assert RESTORER_DEGRADATIONS["real_hat_gan"] == RESTORER_DEGRADATIONS["realesrgan"]


def test_real_hat_gan_rejects_blur():
    assert "blur" not in RESTORER_DEGRADATIONS["real_hat_gan"]
    assert "noise" not in RESTORER_DEGRADATIONS["real_hat_gan"]


def test_real_hat_gan_strength_map_clamped_like_realesrgan():
    raw = np.array([[0, 1, 5, 10]])
    clamped = _restorer_strength_map(raw, "real_hat_gan", "downsample", {})
    assert clamped.max() <= _STRENGTH_CLAMP["real_hat_gan"]
    np.testing.assert_array_equal(
        clamped,
        _restorer_strength_map(raw, "realesrgan", "downsample", {}),
    )


def test_stream_diffvsr_is_registered_alongside_realesrgan():
    """Q7 diffusion VSR — same allowed-degradation set as realesrgan/hat."""
    assert "stream_diffvsr" in RESTORER_DEGRADATIONS
    assert RESTORER_DEGRADATIONS["stream_diffvsr"] == ("downsample",) + INPAINT_DEGRADATIONS
    assert RESTORER_DEGRADATIONS["stream_diffvsr"] == RESTORER_DEGRADATIONS["realesrgan"]


def test_stream_diffvsr_rejects_blur():
    assert "blur" not in RESTORER_DEGRADATIONS["stream_diffvsr"]
    assert "noise" not in RESTORER_DEGRADATIONS["stream_diffvsr"]


def test_stream_diffvsr_strength_map_clamped_like_realesrgan():
    raw = np.array([[0, 1, 5, 10]])
    clamped = _restorer_strength_map(raw, "stream_diffvsr", "downsample", {})
    assert clamped.max() <= _STRENGTH_CLAMP["stream_diffvsr"]
    np.testing.assert_array_equal(
        clamped,
        _restorer_strength_map(raw, "realesrgan", "downsample", {}),
    )


def test_stream_diffvsr_hole_degradations_use_binary_rounds():
    raw = np.array([[0, 5, 0, 9]])
    clamped = _restorer_strength_map(raw, "stream_diffvsr", "freeze", {"rounds": 2})
    np.testing.assert_array_equal(clamped, [[0, 2, 0, 2]])


def test_dc_vsr_is_registered_alongside_realesrgan():
    """Q8 diffusion VSR quality arm — same allowed degradations as SR GANs."""
    assert "dc_vsr" in RESTORER_DEGRADATIONS
    assert RESTORER_DEGRADATIONS["dc_vsr"] == ("downsample",) + INPAINT_DEGRADATIONS
    assert RESTORER_DEGRADATIONS["dc_vsr"] == RESTORER_DEGRADATIONS["realesrgan"]


def test_dc_vsr_rejects_blur():
    assert "blur" not in RESTORER_DEGRADATIONS["dc_vsr"]
    assert "noise" not in RESTORER_DEGRADATIONS["dc_vsr"]


def test_dc_vsr_strength_map_clamped_like_realesrgan():
    raw = np.array([[0, 1, 5, 10]])
    clamped = _restorer_strength_map(raw, "dc_vsr", "downsample", {})
    assert clamped.max() <= _STRENGTH_CLAMP["dc_vsr"]
    np.testing.assert_array_equal(
        clamped,
        _restorer_strength_map(raw, "realesrgan", "downsample", {}),
    )
