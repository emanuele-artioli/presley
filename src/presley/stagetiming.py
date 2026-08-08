"""Per-stage wall-clock accounting, shared by every component.

The article wants to say where the time goes, and until now it could not. Only
*restoration* was timed cleanly. Preprocessing ran before the clock started and
was invisible; selection, degradation, side-channel packing and encoding were
all summed into one `encoding_time_seconds`; and `selection_time_seconds`
existed on `presley_ai` alone and had populated **zero** of 1159 results,
because it was added after every one of them had run.

So the components record a `stage_times_seconds` dict against one vocabulary
(`STAGES`). A shared vocabulary is the point: "encode" has to mean the same
thing in `baselines` and in `presley_ai` or the arms cannot be put on one axis,
which is the whole purpose of the figure this feeds.

**This is an output-only field.** `compute_experiment_hash` consumes the
experiment config and never the result, so recording it moves no hash,
invalidates no cached run, and cannot change what an existing experiment
measures.

Two things it deliberately does *not* do:

* **It does not replace repeat trials.** One run gives one sample of a noisy
  quantity on a shared GPU. A timing *claim* needs repeats under recorded
  conditions, which is `tools/timing_campaign.py`'s job; this supplies the
  breakdown that campaign replays, not the campaign.
* **It does not record a device.** Timing is only comparable within one device
  population, and the components already record the device they resolved. A
  timing that mixes a silent CPU fallback with GPU runs is the defect that made
  every pre-2026-08-05 restoration number unusable, and no amount of
  finer-grained accounting fixes it.
"""

from __future__ import annotations

import contextlib
import time
from typing import Dict, Iterator, List

# The vocabulary. Every component uses these names or none: a stage that means
# something different in two components cannot be plotted on one axis.
#
# Order is pipeline order, which is also the order a stacked bar should read in.
STAGES: List[str] = [
    "preprocess",    # frame extraction and rescale to the target resolution
    "score",         # EVCA complexity + segmentation masks -> removability
    "select",        # choosing WHICH blocks to degrade
    "degrade",       # applying the degradation to the selected blocks
    "encode",        # the codec call itself
    "sidechannel",   # packing and writing the transmitted strength map
    "decode",        # decoding the transmitted video, client side
    "restore",       # the generative model
    "composite",     # passthrough compositing of transmitted and generated
]

# `preprocess` and `score` are cached across experiments, so a run that hits a
# warm cache reports ~0 for them. That is honest -- it is what that run spent --
# but it means the two are a property of the corpus rather than of the method,
# and a cost figure must say which it is showing.
CACHED_STAGES = frozenset({"preprocess", "score"})


class StageTimer:
    """Accumulates wall-clock per named stage.

    Re-entering a stage adds to it rather than replacing it, because several
    stages run once per frame inside a loop over the sequence.
    """

    def __init__(self) -> None:
        self._totals: Dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        if stage not in STAGES:
            # A typo would otherwise appear as a silent extra stage and quietly
            # break the sum against total_time_seconds.
            raise KeyError(f"unknown stage {stage!r}; expected one of {STAGES}")
        start = time.time()
        try:
            yield
        finally:
            self._totals[stage] = self._totals.get(stage, 0.0) + (time.time() - start)

    def add(self, stage: str, seconds: float) -> None:
        """Record a stage timed by hand, e.g. around a subprocess call."""
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; expected one of {STAGES}")
        self._totals[stage] = self._totals.get(stage, 0.0) + seconds

    def as_dict(self) -> Dict[str, float]:
        """Stages that actually ran, in pipeline order.

        Absent stages are omitted rather than zero-filled: `baselines` has no
        restoration step at all, and a 0.0 there would read as "restoration was
        free" instead of "there was none".
        """
        return {stage: self._totals[stage] for stage in STAGES if stage in self._totals}

    def total(self) -> float:
        return sum(self._totals.values())
