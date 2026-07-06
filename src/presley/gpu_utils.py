"""GPU resource preflight for the shared, no-root GPU server.

This box is shared: other users' training jobs routinely hold 25-30 GB on each
GPU, so a naive `cuda:0` launch OOMs. Before dispatching a GPU component
(presley_ai / elvis) the runner calls `preflight_gpu()` to (1) pin
CUDA_VISIBLE_DEVICES to the GPU with the most free memory and (2) suggest a
VRAM-safe InstantIR batch_size. Everything degrades gracefully to a no-op when
nvidia-smi or CUDA is absent (CPU box, or user already pinned a device).
"""
import os
import subprocess
from typing import List, Optional, Tuple


def gpu_free_memory() -> List[Tuple[int, int, int]]:
    """Return [(index, free_mb, total_mb), ...] from nvidia-smi, or [] if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return []
        rows = []
        for line in out.stdout.strip().splitlines():
            idx, free, total = (p.strip() for p in line.split(","))
            rows.append((int(idx), int(free), int(total)))
        return rows
    except Exception:
        return []


def pick_gpu(min_free_mb: int = 2000) -> Optional[Tuple[int, int]]:
    """Pick the GPU with the most free memory. Returns (index, free_mb) or None.

    None means no GPU has at least min_free_mb free — the caller should warn and
    either wait or fall back rather than launch into a near-certain OOM.
    """
    gpus = gpu_free_memory()
    if not gpus:
        return None
    idx, free, _ = max(gpus, key=lambda g: g[1])
    if free < min_free_mb:
        return None
    return idx, free


def suggest_instantir_batch_size(free_mb: int, cap: int = 4) -> int:
    """Largest InstantIR batch_size expected to fit in free_mb, clamped to [1, cap].

    Fit from measured peaks on this repo's InstantIR path: batch_size=1 ~16.7 GB,
    batch_size=4 ~28 GB -> roughly 13 GB base + 3.8 GB/batch. We size to 90% of
    free VRAM as a safety margin against fragmentation and transient spikes.
    """
    base_mb, per_batch_mb = 13000, 3800
    budget = 0.90 * free_mb
    b = int((budget - base_mb) // per_batch_mb)
    return max(1, min(cap, b))


def preflight_gpu(component_name: str, experiment: dict, min_free_mb: int = 2000) -> None:
    """Pin the least-loaded GPU and fill in a VRAM-safe InstantIR batch_size.

    Mutates `experiment` in place: sets restorer_params.batch_size for
    presley_ai/instantir when the user hasn't specified one. Respects an
    externally-set CUDA_VISIBLE_DEVICES (does not override a deliberate pin).
    No-op for non-GPU components and when nvidia-smi/CUDA is unavailable.
    """
    if component_name not in ("presley_ai", "elvis"):
        return

    picked = pick_gpu(min_free_mb)
    if picked is None:
        gpus = gpu_free_memory()
        detail = ", ".join(f"gpu{i}:{f}MB free" for i, f, _ in gpus) or "no GPUs visible"
        print(f"  [preflight] WARNING: no GPU with >= {min_free_mb}MB free ({detail}); "
              f"launching anyway — OOM risk.")
        return
    idx, free = picked

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        print(f"  [preflight] pinned CUDA_VISIBLE_DEVICES={idx} ({free}MB free).")
    else:
        print(f"  [preflight] CUDA_VISIBLE_DEVICES already set to "
              f"{os.environ['CUDA_VISIBLE_DEVICES']}; leaving it (best free was gpu{idx}, {free}MB).")

    # Suggest a safe InstantIR batch size only when the user didn't pin one.
    if component_name == "presley_ai" and experiment.get("restorer", "").lower() == "instantir":
        rp = experiment.setdefault("restorer_params", {})
        if "batch_size" not in rp:
            b = suggest_instantir_batch_size(free)
            rp["batch_size"] = b
            print(f"  [preflight] InstantIR batch_size auto-set to {b} for {free}MB free.")
