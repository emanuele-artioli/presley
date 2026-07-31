"""Superblock geometry and the claim-(b) probe's summary logic.

The probe's GPU half is a real `presley_ai` run and is not re-tested here. What
is tested is everything that could silently produce a plausible-but-wrong
spread: where superblock boundaries fall, that MSE (not PSNR) is what gets
averaged, and that the degraded/untouched split means what the claim says.
"""

import numpy as np
import pytest

from presley.blockdamage import (
    SB,
    pool_to_superblocks,
    pool_weights,
    psnr_from_mse,
    superblock_mse,
)
from presley.components.probe_block_damage import DEGRADED_FRAC, _percentiles


# --- geometry ------------------------------------------------------------------


def test_a_partial_edge_superblock_keeps_its_true_area():
    """360 rows is 5 full superblocks plus a 40-row strip, not 5 superblocks.

    Dropping the strip would quietly exclude the bottom 11% of every frame from
    the damage statistic.
    """
    w = pool_weights(n_blocks=45, block_size=8, extent=360)
    assert w.shape[0] == 6
    assert w[-1].sum() == 40


def test_a_block_size_that_does_not_divide_64_is_split_exactly():
    """bs=24 appears in real runs: block 2 spans pixels [48,72), crossing the boundary."""
    w = pool_weights(n_blocks=4, block_size=24, extent=96)
    assert w[0, 2] == 16   # [48,64)
    assert w[1, 2] == 8    # [64,72)


def test_superblock_mse_averages_squared_error_not_psnr():
    """One bright block and one dark one must not average as decibels.

    A PSNR-space average of the two is ~34 dB; the MSE-space answer is ~24 dB.
    Averaging PSNR across blocks is the single easiest way to produce a wrong
    spread that still looks like a plausible number.
    """
    ref = np.zeros((SB, 2 * SB, 3), dtype=np.uint8)
    dec = ref.copy()
    dec[:, SB:] = 100  # right superblock is badly wrong, left is perfect

    mse = superblock_mse([ref], [dec])
    assert mse.shape == (1, 1, 2)
    assert mse[0, 0, 0] == 0.0
    assert mse[0, 0, 1] == pytest.approx(100.0 ** 2)


def test_pooling_a_uniform_map_is_the_identity_on_its_value():
    arr = np.full((3, 45, 80), 0.25)
    pooled = pool_to_superblocks(arr, block_size=8, height=360, width=640)
    assert pooled.shape == (3, 6, 10)
    assert np.allclose(pooled, 0.25)


def test_psnr_of_a_perfect_block_is_finite():
    """Zero MSE must not become inf and poison every percentile downstream."""
    assert np.isfinite(psnr_from_mse(np.array([0.0]))).all()


# --- the summary the claim is made from ----------------------------------------


def test_the_headline_is_a_within_run_dispersion():
    """p90-p10 over superblocks, which is what claim (b)'s 6.2/8.2 dB asserts."""
    values = np.arange(0.0, 101.0)  # p10=10, p90=90
    summary = _percentiles(values)
    assert summary["spread_p90_p10"] == pytest.approx(80.0)
    assert summary["n"] == 101


def test_an_empty_group_reports_no_percentiles_rather_than_nan():
    """A run where nothing was degraded has no spread -- not a spread of nan."""
    assert _percentiles(np.array([])) == {"n": 0}


def test_a_partly_degraded_superblock_is_in_neither_group():
    """The split has a gap by design.

    An SB straddling the selection boundary is genuinely part-degraded; putting
    it in `degraded` inflates the untouched-side contamination check, and
    putting it in `untouched` inflates the headline spread.
    """
    frac = np.array([0.0, 0.3, 0.5, 0.9])
    degraded = frac > DEGRADED_FRAC
    untouched = frac == 0.0
    assert degraded.tolist() == [False, False, False, True]
    assert untouched.tolist() == [True, False, False, False]
    assert not (degraded & untouched).any()
