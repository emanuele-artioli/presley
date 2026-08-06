"""Acquisition conditions — when a run happened and how busy the box was.

Added 2026-08-03. Timings in this corpus were unattributable: ProPainter's
restoration time at 640x360 splits into two clusters ~10-40x apart with no
recorded config explaining it, and with neither a timestamp nor GPU occupancy
stored, a code change could not be told apart from a co-tenant's job on this
shared server. The sibling failure is the withdrawn Wave 2C throughput table,
where a median pooled across two acquisition batches produced a value no run
ever achieved.

These cover the only things that can go wrong in code we own: that the
annotation records what it promises, and that it can never take a finished run
down with it.
"""

import pytest

from presley import gpu_utils


def test_acquisition_conditions_records_time_and_gpu_state(monkeypatch):
    monkeypatch.setattr(gpu_utils, "gpu_free_memory",
                        lambda: [(0, 10_000, 49_140), (1, 40_000, 49_140)])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    out = gpu_utils.acquisition_conditions()

    assert out["gpu_visible"] == "1"
    assert len(out["gpus"]) == 2
    # Busiest card is gpu0 at (49140-10000)/49140 -- the co-tenancy proxy that
    # tells a later reader whether a slow timing was our code or someone else's.
    assert out["max_used_frac"] == pytest.approx(0.796, abs=0.001)
    assert out["finished_utc"].endswith("+00:00")


def test_acquisition_conditions_never_raises_on_a_cpu_box(monkeypatch):
    """A timing annotation must not be able to fail a finished run -- losing an
    hours-long restoration to bookkeeping is the failure this guards."""
    def boom():
        raise RuntimeError("nvidia-smi missing")

    monkeypatch.setattr(gpu_utils, "gpu_free_memory", boom)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    out = gpu_utils.acquisition_conditions()

    assert out["gpu_visible"] is None
    assert "gpus" not in out
    assert out["finished_utc"]


# --- what the restoration actually ran on (W3) --------------------------------

def test_resolved_devices_are_recorded_so_a_timing_has_a_device():
    """CPU fallback is enabled at every restorer call site and is silent: a run
    with no visible CUDA device does not fail, it runs ~30x slower. A timing
    whose device is unknown is not a measurement."""
    from presley.concurrency import (_resolve_device_list, last_resolved_devices,
                                     reset_resolved_devices)

    reset_resolved_devices()
    assert last_resolved_devices() == []

    _resolve_device_list(["cpu"])

    assert last_resolved_devices() == ["cpu"]


def test_no_restoration_reads_as_unknown_not_as_cpu():
    """A `none` control resolves no device at all. Reporting that as CPU would
    invent the very fact this record exists to establish."""
    from presley.concurrency import last_resolved_devices, reset_resolved_devices

    reset_resolved_devices()

    assert last_resolved_devices() == []


def test_one_run_cannot_inherit_the_previous_runs_device():
    from presley.concurrency import (_resolve_device_list, last_resolved_devices,
                                     reset_resolved_devices)

    _resolve_device_list(["cpu"])
    assert last_resolved_devices() == ["cpu"]

    reset_resolved_devices()

    assert last_resolved_devices() == []
