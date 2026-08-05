"""Rate control and QP mapping.

`derive_rate_control` is what lets the invariant tier check the fixed-QP hard
rule mechanically. Every degradation claim in the paper depends on the run
having been fixed-QP/CRF rather than VBR, and 25/25 matched VBR pairs have
shown that a VBR degradation curve measures the encoder, not the method — so
mislabelling a mode here would let a VBR run be cited as evidence.
"""

import numpy as np
import pytest

from presley import encode_utils
from presley.encode_utils import calculate_target_bitrate, derive_rate_control, scores_to_qp_offsets


@pytest.mark.parametrize(
    "codec, params, expected",
    [
        ("x265", {"qp": 30}, "cqp"),
        ("x265", {}, "vbr_2pass"),
        ("x264", {"qp": 30}, "vbr_2pass"),  # x264 path has no constant-QP mode
        ("kvazaar", {}, "vbr_1pass"),
        # Added 2026-08-03 with the fixed-QP kvazaar baseline path. Before it,
        # kvazaar could ONLY be encoded --bitrate, so tab:roi's fixed-QP ROI
        # arms had only VBR baselines to compare against -- and kvazaar VBR
        # overshoots targets by 30-45%, the same band as the table's headline
        # "saving". Mislabelling either of these as VBR would re-hide that.
        ("kvazaar", {"qp": 30}, "cqp"),
        ("kvazaar", {"rate_control": "cqp"}, "cqp"),
        ("svtav1", {"qp": 40}, "crf"),  # rc=0:q=N with aq-mode=2 is CRF, not CQP
        ("svtav1", {}, "vbr_1pass"),
        ("unknown_codec", {"qp": 1}, "n/a"),
    ],
)
def test_rate_control_per_codec(codec, params, expected):
    assert derive_rate_control(codec, params) == expected


def test_codec_name_is_case_insensitive():
    assert derive_rate_control("X265", {"qp": 30}) == "cqp"


def test_missing_params_are_treated_as_absent():
    assert derive_rate_control("x265", None) == "vbr_2pass"


@pytest.mark.parametrize(
    "roi_method, expected",
    [
        ("kvazaar", "cqp"),
        ("svtav1", "crf"),
        ("x265_aq", "vbr_2pass"),
    ],
)
def test_roi_method_decides_the_mode_regardless_of_codec_params(roi_method, expected):
    """The ROI helpers binary-search their own fixed QP, ignoring codec_params."""
    assert derive_rate_control("x265", {"qp": 30}, roi_method=roi_method) == expected


@pytest.mark.parametrize(
    "roi_method", ["presley_downsample", "presley_blur", "presley_noise", "presley_qp"]
)
def test_presley_degradations_are_reported_as_vbr(roi_method):
    """`qp` here is a degradation knob, not an encoder flag — the encode is VBR.

    Reading that key as constant-QP would label these runs fixed-QP and let the
    invariant check pass on exactly the VBR configuration it exists to reject.
    """
    assert derive_rate_control("x265", {"qp": 30}, roi_method=roi_method) == "vbr_2pass"
    assert derive_rate_control("kvazaar", {"qp": 30}, roi_method=roi_method) == "vbr_1pass"


def test_unknown_roi_method_is_not_guessed():
    assert derive_rate_control("x265", {"qp": 30}, roi_method="something_new") == "n/a"


# --- QP offsets ----------------------------------------------------------------
# Scores arrive as [frames, blocks_y, blocks_x]; the mapping mean-centres each
# frame so the offsets move bits around without changing the total budget.


def test_qp_offsets_stay_within_the_requested_range():
    scores = np.array([[[0.0, 0.5, 1.0]]])
    offsets = scores_to_qp_offsets(scores, qp_range=15)

    assert offsets.min() >= -15
    assert offsets.max() <= 15


def test_more_removable_blocks_get_a_higher_qp():
    """Higher removability must mean coarser quantisation, never the reverse.

    A sign flip here would spend *more* bits on the blocks the method judged
    least important — inverting Goal 1 while still producing plausible numbers.
    """
    offsets = scores_to_qp_offsets(np.array([[[0.0, 1.0]]]), qp_range=15)
    assert offsets.flat[1] > offsets.flat[0]


def test_offsets_are_bit_neutral_within_a_frame():
    """Mean-centring is the mechanism: offsets must sum to ~0 per frame.

    Without it the skew of DAVIS removability scores makes almost every offset
    negative, telling the encoder to spend more bits nearly everywhere.
    """
    scores = np.array([[[0.1, 0.2, 0.2, 0.9]]])
    offsets = scores_to_qp_offsets(scores, qp_range=15)

    assert offsets.sum(axis=(1, 2)) == pytest.approx(0.0, abs=1e-5)


def test_uniform_scores_produce_no_offsets():
    """If every block is equally removable there is nothing to reallocate."""
    offsets = scores_to_qp_offsets(np.full((1, 4, 4), 0.5), qp_range=15)
    assert np.all(offsets == 0.0)


def test_target_bitrate_grows_with_resolution():
    small = calculate_target_bitrate(640, 360, 30.0)
    large = calculate_target_bitrate(1920, 1080, 30.0)
    assert large > small


def test_target_bitrate_scales_with_the_quality_factor():
    base = calculate_target_bitrate(1920, 1080, 30.0, quality_factor=1.0)
    high = calculate_target_bitrate(1920, 1080, 30.0, quality_factor=2.0)
    assert high > base


# --- SVT-AV1 tune (F3) -------------------------------------------------------
# The encoder's own default is tune=1 (PSNR), so every encode predating this
# parameter optimized an objective the paper does not claim. These pin the
# backward-compatibility contract: omitting tune must not change the command.

def test_svtav1_qp_omits_tune_by_default(monkeypatch):
    """Adding the parameter must not alter a single existing encode -- results/
    holds 694 runs keyed by a hash of their config, and a changed command line
    would silently invalidate all of them."""
    seen = {}

    def fake_run(cmd, **kw):
        seen['cmd'] = cmd
        class R: returncode = 0
        return R()

    monkeypatch.setattr(encode_utils.subprocess, 'run', fake_run)
    encode_utils.encode_video_svtav1_qp('in.mp4', 'out.mp4', 24.0, 43)

    params = seen['cmd'][seen['cmd'].index('-svtav1-params') + 1]
    assert params == 'rc=0:q=43'
    assert 'tune' not in params


def test_svtav1_qp_passes_tune_when_set(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen['cmd'] = cmd
        class R: returncode = 0
        return R()

    monkeypatch.setattr(encode_utils.subprocess, 'run', fake_run)
    encode_utils.encode_video_svtav1_qp('in.mp4', 'out.mp4', 24.0, 43, tune=0)

    params = seen['cmd'][seen['cmd'].index('-svtav1-params') + 1]
    assert params == 'rc=0:q=43:tune=0'


def test_svtav1_qp_tune_zero_is_not_treated_as_unset(monkeypatch):
    """tune=0 is VQ -- the one value we actually want to test -- and is falsy.
    A truthiness check instead of an `is None` check would silently drop it."""
    seen = {}

    def fake_run(cmd, **kw):
        seen['cmd'] = cmd
        class R: returncode = 0
        return R()

    monkeypatch.setattr(encode_utils.subprocess, 'run', fake_run)
    encode_utils.encode_video_svtav1_qp('in.mp4', 'out.mp4', 24.0, 43, tune=0)
    assert 'tune=0' in seen['cmd'][seen['cmd'].index('-svtav1-params') + 1]
