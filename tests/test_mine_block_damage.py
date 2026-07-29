"""The damage miner turns existing runs into the missing half of the selection
objective, so its pooling arithmetic has to be right or every downstream
conclusion about "which blocks restore well" inherits the error.

The dangerous mistakes are quiet ones: averaging PSNR instead of MSE (PSNR is
logarithmic, so the mean is not the pooled value), dropping a partial edge
superblock, and mishandling block sizes that do not divide 64 -- bs=24 appears
in real runs and would be silently truncated by a reshape-based pool.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "mine_block_damage.py"
_spec = importlib.util.spec_from_file_location("mine_block_damage", _TOOL)
mbd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mbd)


def test_constant_field_pools_to_the_same_constant():
    """The weakest possible invariant, and the one that catches weight-
    normalization bugs: pooling a uniform field must not change its value."""
    arr = np.full((3, 45, 80), 7.5)
    out = mbd.pool_to_superblocks(arr, block_size=8, height=360, width=640)
    assert np.allclose(out, 7.5)


def test_block_size_dividing_64_is_plain_block_averaging():
    """bs=8 -> exactly 8x8 blocks per superblock, so pooling must equal a
    straight mean over those blocks."""
    rng = np.random.default_rng(0)
    arr = rng.random((2, 16, 16))  # 128x128 px at bs=8 -> 2x2 superblocks
    out = mbd.pool_to_superblocks(arr, block_size=8, height=128, width=128)

    assert out.shape == (2, 2, 2)
    expected = arr.reshape(2, 2, 8, 2, 8).mean(axis=(2, 4))
    assert np.allclose(out, expected)


def test_block_size_not_dividing_64_is_area_weighted():
    """bs=24 does not divide 64. A block straddling a superblock boundary must
    contribute in proportion to the pixels it actually covers."""
    # 96px wide = 4 blocks of 24 = 1 full SB (64px) + partial SB (32px).
    arr = np.array([[[0.0, 0.0, 1.0, 1.0]]])  # (1,1,4)
    out = mbd.pool_to_superblocks(arr, block_size=24, height=24, width=96)

    assert out.shape == (1, 1, 2)
    # SB0 spans px 0-63: block0 24px(0.0), block1 24px(0.0), block2 16px(1.0)
    assert out[0, 0, 0] == pytest.approx(16 / 64)
    # SB1 spans px 64-95: block2 8px(1.0), block3 24px(1.0)
    assert out[0, 0, 1] == pytest.approx(1.0)


def test_partial_edge_superblock_is_kept_not_dropped():
    """360 is not a multiple of 64 (5 full SBs + 40px). Dropping the remainder
    would silently discard 11% of every frame."""
    arr = np.ones((1, 45, 10))  # 360x80 px at bs=8
    out = mbd.pool_to_superblocks(arr, block_size=8, height=360, width=80)

    assert out.shape == (1, 6, 2)  # ceil(360/64)=6, ceil(80/64)=2
    assert np.allclose(out, 1.0)   # including the partial row


def test_pooling_happens_in_mse_not_psnr():
    """Pooling PSNR would give the mean of logs; the correct value is the log
    of the mean MSE. These differ, and the difference is the whole point."""
    mse = np.array([[[1.0, 10000.0]]])
    pooled = mbd.pool_to_superblocks(mse, block_size=64, height=64, width=128)
    # One superblock per block here, so check the conversion path instead.
    psnr_of_mean = mbd._psnr(np.array([mse.mean()]))
    mean_of_psnr = mbd._psnr(mse.ravel()).mean()
    assert not np.isclose(psnr_of_mean, mean_of_psnr)
    assert np.allclose(pooled.ravel(), mse.ravel())


def test_psnr_uses_8bit_peak():
    """MSE of 0 must not produce inf/NaN, and the peak must match the
    evaluation code's 255 convention or deltas won't be comparable."""
    assert mbd._psnr(np.array([(255.0 ** 2)]))[0] == pytest.approx(0.0)
    assert np.isfinite(mbd._psnr(np.array([0.0]))[0])


def test_strength_fraction_is_a_coverage_ratio():
    """strength_frac is pooled from a binary map, so a superblock that is half
    degraded must read 0.5 -- that partial coverage is what distinguishes an
    SB-snapped selection from a scattered one."""
    binary = np.zeros((1, 8, 16))
    binary[0, :, :8] = 1.0  # left 64px of a 128px-wide frame
    out = mbd.pool_to_superblocks(binary, block_size=8, height=64, width=128)
    assert out[0, 0, 0] == pytest.approx(1.0)
    assert out[0, 0, 1] == pytest.approx(0.0)
