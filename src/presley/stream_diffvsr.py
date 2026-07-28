"""Stream-DiffVSR glue for the ``stream_diffvsr`` conditioned restorer (Q7).

Upstream (https://github.com/jamichss/Stream-DiffVSR) is a 4× auto-regressive
diffusion VSR. Its deps (diffusers / xformers pins, often a newer torch) must
**not** be pip-installed into the pinned ``presley`` conda env. Wire inference
via subprocess against an isolated checkout + env:

  git clone https://github.com/jamichss/Stream-DiffVSR.git vendor/stream-diffvsr
  # separate conda env from upstream requirements.yml / requirements-cu12.txt
  # then point restorer_params.python at that env's interpreter, and
  # restorer_params.vendor_dir (or $STREAM_DIFFVSR_ROOT) at the checkout.

Weights come from HF ``Jamichsu/Stream-DiffVSR`` (auto-fetched by upstream
``inference.py`` into the HF cache; optional ``cache_dir`` / HF_HOME).

fp16 policy
-----------
Prefer fp16 on CUDA for latency (``fp32=False``). If Softmax / LayerNorm NaNs
appear in restored frames, reject ``fp32=False`` the same way Real-HAT / NAFNet
do — set ``_STREAM_DIFFVSR_FP16_UNSAFE = True`` after Wave-2 confirmation, or
rely on the per-run NaN check in ``assert_finite_bgr``. Upstream ``inference.py``
does not expose a dtype flag; our runner passes ``STREAM_DIFFVSR_FP16=1`` when
``fp32=False`` so a patched/vendored entrypoint can honour it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

DEFAULT_MODEL_ID = "Jamichsu/Stream-DiffVSR"
DEFAULT_NUM_INFERENCE_STEPS = 4
STREAM_DIFFVSR_SCALE = 4  # SD x4 upscaler backbone

# Flip to True after Wave-2 GPU smokes prove systemic Softmax/LN NaNs under half.
_STREAM_DIFFVSR_FP16_UNSAFE = False

_ENV_ROOT = "STREAM_DIFFVSR_ROOT"
_ENV_PYTHON = "STREAM_DIFFVSR_PYTHON"
_ENV_FP16 = "STREAM_DIFFVSR_FP16"


def validate_fp32_policy(fp32: bool) -> None:
    """Enforce the documented fp16 policy before launching inference."""
    if not fp32 and _STREAM_DIFFVSR_FP16_UNSAFE:
        raise ValueError(
            "Stream-DiffVSR must run in float32 (fp32=True). Half precision "
            "produces NaN activations in Softmax/LayerNorm on this stack; "
            "see tests/test_stream_diffvsr.py and docs/EXPERIMENTS_QUEUED.md Q7."
        )


def assert_finite_bgr(frame: np.ndarray, *, context: str = "Stream-DiffVSR") -> None:
    """Reject half-precision runs whose outputs already contain NaN/Inf."""
    arr = np.asarray(frame)
    if not np.isfinite(arr).all():
        raise ValueError(
            f"{context} produced non-finite pixels (NaN/Inf). Half precision "
            "(fp32=False) is unsafe on this stack — re-run with fp32=True "
            "(same rejection pattern as Real-HAT-GAN / NAFNet)."
        )


def resolve_vendor_dir(vendor_dir: Optional[Union[str, Path]] = None) -> Path:
    """Locate the Stream-DiffVSR checkout (inference.py + pipeline/)."""
    candidates: List[Path] = []
    if vendor_dir is not None:
        candidates.append(Path(vendor_dir).expanduser())
    env = os.environ.get(_ENV_ROOT)
    if env:
        candidates.append(Path(env).expanduser())
    # Common local layouts relative to the repo / cwd.
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # src/presley/ → repo
    candidates.extend(
        [
            repo_root / "vendor" / "stream-diffvsr",
            Path.cwd() / "vendor" / "stream-diffvsr",
            Path.home() / "emanuele" / "Models" / "Stream-DiffVSR",
        ]
    )
    for path in candidates:
        if (path / "inference.py").is_file():
            return path.resolve()
    searched = ", ".join(str(p) for p in candidates) or "(none)"
    raise FileNotFoundError(
        "Stream-DiffVSR checkout not found (need inference.py). Clone with "
        "`git clone https://github.com/jamichss/Stream-DiffVSR.git "
        "vendor/stream-diffvsr` and pass restorer_params.vendor_dir, or set "
        f"${_ENV_ROOT}. Searched: {searched}"
    )


def resolve_python(python_executable: Optional[Union[str, Path]] = None) -> str:
    """Interpreter for the isolated Stream-DiffVSR env (not the pinned presley env)."""
    if python_executable is not None:
        return str(Path(python_executable).expanduser())
    env = os.environ.get(_ENV_PYTHON)
    if env:
        return str(Path(env).expanduser())
    # Fall back to current interpreter — only useful if deps happen to match.
    return sys.executable


def build_seq_layout(
    frame_paths: Sequence[Path],
    staging_root: Union[str, Path],
    *,
    seq_name: str = "seq",
    scale: int = STREAM_DIFFVSR_SCALE,
) -> Path:
    """Write LR frames under ``staging_root/in/<seq_name>/frame_XXXX.png``.

    Upstream expects ``in_path/seq_name/frame_*.png``. Returns the ``in_path``
    directory to pass as ``--in_path``.
    """
    staging_root = Path(staging_root)
    in_path = staging_root / "in"
    seq_dir = in_path / seq_name
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    for idx, src in enumerate(frame_paths, start=1):
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read frame for Stream-DiffVSR: {src}")
        h, w = bgr.shape[:2]
        lr_w = max(1, w // scale)
        lr_h = max(1, h // scale)
        lr = cv2.resize(bgr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        out_name = f"frame_{idx:04d}.png"
        if not cv2.imwrite(str(seq_dir / out_name), lr):
            raise RuntimeError(f"Failed to write LR frame {seq_dir / out_name}")
    return in_path


def locate_upstream_output_dir(out_path: Path, in_path: Path, seq_name: str = "seq") -> Path:
    """Map upstream's nested out layout back to the sequence folder.

    ``inference.py`` writes to ``out_path / <in_path.name> / <seq_name> /``.
    """
    nested = out_path / in_path.name / seq_name
    flat = out_path / seq_name
    if nested.is_dir():
        return nested
    if flat.is_dir():
        return flat
    # Last resort: any directory under out_path that has PNGs.
    for child in sorted(out_path.rglob("*.png")):
        return child.parent
    raise FileNotFoundError(
        f"Stream-DiffVSR produced no frames under {out_path} "
        f"(expected {nested} or {flat})."
    )


def run_stream_diffvsr_inference(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    fp32: bool = False,
    vendor_dir: Optional[Union[str, Path]] = None,
    python_executable: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Invoke upstream ``inference.py`` via subprocess; return the seq output dir."""
    validate_fp32_policy(fp32)
    vendor = resolve_vendor_dir(vendor_dir)
    python = resolve_python(python_executable)
    in_path = Path(in_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    inference_script = vendor / "inference.py"
    cmd = [
        python,
        str(inference_script),
        "--model_id",
        model_id,
        "--in_path",
        str(in_path),
        "--out_path",
        str(out_path),
        "--num_inference_steps",
        str(int(num_inference_steps)),
    ]
    env = os.environ.copy()
    env[_ENV_FP16] = "0" if fp32 else "1"
    if cache_dir is not None:
        cache = str(Path(cache_dir).expanduser().resolve())
        env["HF_HOME"] = cache
        env["HUGGINGFACE_HUB_CACHE"] = str(Path(cache) / "hub")
    # Keep vendor imports resolving from the checkout.
    env["PYTHONPATH"] = (
        str(vendor) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(vendor),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Stream-DiffVSR inference failed (exit "
            f"{result.returncode}). Prefer an isolated env — do not pip-upgrade "
            "torch/diffusers inside the pinned `presley` conda env.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    # Discover seq name from in_path children.
    seqs = sorted(p.name for p in in_path.iterdir() if p.is_dir())
    seq_name = seqs[0] if seqs else "seq"
    return locate_upstream_output_dir(out_path, in_path, seq_name=seq_name)
