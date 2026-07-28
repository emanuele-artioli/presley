"""Regression: InstantIR num_inference_steps must be configurable.

Historical CLAIM(tab:conditioned) blur+instantir runs used steps=1 (hardcoded).
Q2 corrected smokes pass steps via restorer_params; the default stays 1 so
existing hashes remain reproducible. See docs/EXPERIMENTS_QUEUED.md audit.
"""
import inspect

import pytest

pytest.importorskip("instantir", reason="needs the pinned presley conda env")

from presley.restoration import restore_with_instantir_adaptive


def test_instantir_adaptive_exposes_num_inference_steps_defaulting_to_one():
    sig = inspect.signature(restore_with_instantir_adaptive)
    param = sig.parameters["num_inference_steps"]
    assert param.default == 1


def test_instantir_adaptive_rejects_non_positive_steps():
    with pytest.raises(ValueError, match="num_inference_steps"):
        restore_with_instantir_adaptive(
            "/tmp/nonexistent_in",
            "/tmp/nonexistent_out",
            blur_maps=__import__("numpy").zeros((1, 1, 1), dtype=int),
            block_size=8,
            num_inference_steps=0,
        )
