"""Fast-tier wiring tests for Stream-DiffVSR glue (no GPU / no vendor checkout)."""
from unittest import mock

import cv2
import numpy as np
import pytest

from presley import stream_diffvsr as sdv


def test_validate_fp32_policy_allows_fp16_by_default():
    """Wave-1 policy: try fp16 (fp32=False) until Softmax/LN NaNs are proven."""
    sdv.validate_fp32_policy(fp32=False)
    sdv.validate_fp32_policy(fp32=True)


def test_validate_fp32_policy_rejects_when_fp16_marked_unsafe(monkeypatch):
    monkeypatch.setattr(sdv, "_STREAM_DIFFVSR_FP16_UNSAFE", True)
    with pytest.raises(ValueError, match="float32"):
        sdv.validate_fp32_policy(fp32=False)


def test_assert_finite_bgr_rejects_nans():
    bad = np.full((4, 4, 3), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite|fp32=True"):
        sdv.assert_finite_bgr(bad)


def test_build_seq_layout_writes_lr_under_seq(tmp_path, rng):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    src = frames_dir / "00000.png"
    cv2.imwrite(str(src), frame)
    in_path = sdv.build_seq_layout([src], tmp_path / "stage", seq_name="seq", scale=4)
    lr_files = sorted((in_path / "seq").glob("frame_*.png"))
    assert len(lr_files) == 1
    lr = cv2.imread(str(lr_files[0]))
    assert lr.shape == (4, 4, 3)


def test_locate_upstream_output_dir_nested(tmp_path):
    in_path = tmp_path / "in"
    out_path = tmp_path / "out"
    nested = out_path / "in" / "seq"
    nested.mkdir(parents=True)
    (nested / "frame_0001.png").write_bytes(b"x")
    assert sdv.locate_upstream_output_dir(out_path, in_path, seq_name="seq") == nested


def test_run_stream_diffvsr_inference_builds_expected_cmd(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "inference.py").write_text("# stub\n", encoding="utf-8")
    in_path = tmp_path / "in"
    seq = in_path / "seq"
    seq.mkdir(parents=True)
    (seq / "frame_0001.png").write_bytes(b"x")
    out_path = tmp_path / "out"
    nested = out_path / "in" / "seq"
    nested.mkdir(parents=True)
    (nested / "frame_0001.png").write_bytes(b"y")

    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sdv.subprocess, "run", _fake_run)
    result = sdv.run_stream_diffvsr_inference(
        in_path,
        out_path,
        model_id="Jamichsu/Stream-DiffVSR",
        num_inference_steps=4,
        fp32=False,
        vendor_dir=vendor,
        python_executable="/usr/bin/python3",
        cache_dir=tmp_path / "hf_cache",
    )
    assert result == nested
    assert captured["cmd"][0] == "/usr/bin/python3"
    assert "--num_inference_steps" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--num_inference_steps") + 1] == "4"
    assert captured["env"][sdv._ENV_FP16] == "1"
    assert captured["env"]["HF_HOME"] == str((tmp_path / "hf_cache").resolve())


def test_run_stream_diffvsr_inference_sets_fp16_env_off_when_fp32(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "inference.py").write_text("# stub\n", encoding="utf-8")
    in_path = tmp_path / "in"
    (in_path / "seq").mkdir(parents=True)
    out_path = tmp_path / "out"
    nested = out_path / "in" / "seq"
    nested.mkdir(parents=True)
    (nested / "a.png").write_bytes(b"y")
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sdv.subprocess, "run", _fake_run)
    sdv.run_stream_diffvsr_inference(
        in_path, out_path, fp32=True, vendor_dir=vendor, python_executable="python"
    )
    assert captured["env"][sdv._ENV_FP16] == "0"
