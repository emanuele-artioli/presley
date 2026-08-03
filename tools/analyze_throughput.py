#!/usr/bin/env python
"""Wave 2C -- throughput, and cost attached to the operating map.

Three things, in the order they must be done:

1. **Close the svtav1 720p/1080p alarm.** Median encode throughput pooled over
   a cell said 3.2 fps at 720p and 28.9 fps at 1080p. Faster at higher
   resolution is impossible, so the pooling is wrong, not the physics. This
   tool splits every cell into *acquisition batches* and refuses to quote a
   median over a cell whose batches are disjoint in fps.

2. **A throughput row per configuration, with conditions.** Rows are per
   (codec, resolution, preset, batch), carrying n, median, IQR and the QP
   range, because on a shared box a pooled median is a mixture, not a centre.

3. **Cost attached to each map recommendation**, reusing Wave 1's scoring so
   the numbers cannot drift from the map they annotate.

Read-only against results/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_operating_map as wave1  # noqa: E402

DB = "/home/itec/emanuele/presley/results/presley.db"
RESULTS = "/home/itec/emanuele/presley/results"

#: Two runs belong to the same acquisition batch if they land within this many
#: seconds of each other. Batches in this corpus are minutes apart internally
#: and days apart from each other, so the threshold is not delicate.
BATCH_GAP_S = 6 * 3600

#: Source framerate for every clip in the corpus.
SOURCE_FPS = 24.0


# --------------------------------------------------------------------------
# Robust statistics
# --------------------------------------------------------------------------

def median(values: Sequence[float]) -> float:
    return statistics.median(values)


def iqr(values: Sequence[float]) -> float:
    """Interquartile range. Zero for fewer than 2 points."""
    if len(values) < 2:
        return 0.0
    xs = sorted(values)
    n = len(xs)

    def q(frac: float) -> float:
        pos = frac * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])

    return q(0.75) - q(0.25)


def cluster_batches(times: Sequence[float], gap_s: float = BATCH_GAP_S) -> List[int]:
    """Assign each timestamp a batch id, splitting wherever the sorted gap
    exceeds ``gap_s``. Returns ids in the caller's original order."""
    if not times:
        return []
    order = sorted(range(len(times)), key=lambda i: times[i])
    ids = [0] * len(times)
    batch = 0
    prev = times[order[0]]
    for i in order:
        if times[i] - prev > gap_s:
            batch += 1
        ids[i] = batch
        prev = times[i]
    return ids


def ranges_disjoint(ranges: Sequence[Tuple[float, float]]) -> bool:
    """True when no two (lo, hi) intervals overlap -- i.e. the groups do not
    merely differ in centre, they do not share any value at all. That is the
    signature of a mixture whose pooled median describes no group."""
    rs = sorted(ranges)
    for a, b in zip(rs, rs[1:]):
        if b[0] <= a[1]:
            return False
    return True


# --------------------------------------------------------------------------
# Timing records
# --------------------------------------------------------------------------

@dataclass
class Timing:
    hash: str
    component: str
    video: str
    codec: Optional[str]
    width: int
    height: int
    preset: Optional[str]
    qp: Any
    frames: Optional[int]
    encode_s: Optional[float]
    restore_s: Optional[float]
    transport: Optional[str]
    fill: Optional[str]
    mtime: float = 0.0

    @property
    def encode_fps(self) -> Optional[float]:
        if not self.encode_s or not self.frames:
            return None
        return self.frames / self.encode_s

    @property
    def restore_fps(self) -> Optional[float]:
        if not self.restore_s or not self.frames:
            return None
        return self.frames / self.restore_s

    @property
    def wall_fps(self) -> Optional[float]:
        total = (self.encode_s or 0.0) + (self.restore_s or 0.0)
        if not total or not self.frames:
            return None
        return self.frames / total


def run_mtime(results_root: str, h: str) -> float:
    """Wall-clock time the run was produced, for batch clustering.

    metrics.json is rewritten by bulk re-evaluation passes, so it is not a
    reliable acquisition time. The *earliest* mtime among a run's artifacts is
    closer to when it was actually encoded.
    """
    d = os.path.join(results_root, h)
    if not os.path.isdir(d):
        return 0.0
    stamps = [os.path.getmtime(p)
              for p in glob.glob(os.path.join(d, "*"))
              if os.path.isfile(p)]
    return min(stamps) if stamps else 0.0


def load_timings(db: str, results_root: str = RESULTS) -> List[Timing]:
    """Fixed-QP, invariant-clean runs with their logged timings."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT hash, doc, component, video, width, height, codec, qp,"
        "       degradation, restorer, inpainter, video_frames"
        "  FROM runs"
        " WHERE rate_control IN (?, ?) AND n_invariant_failures = 0",
        wave1.FIXED_QP,
    ).fetchall()
    con.close()

    out: List[Timing] = []
    for r in rows:
        doc = json.loads(r["doc"])
        cfg = doc.get("config") or {}
        params = cfg.get("codec_params") or {}
        out.append(Timing(
            hash=r["hash"], component=r["component"], video=r["video"],
            codec=r["codec"], width=r["width"], height=r["height"],
            preset=str(params.get("preset")) if params.get("preset") is not None else None,
            qp=r["qp"] if r["qp"] is not None else params.get("qp"),
            frames=r["video_frames"],
            encode_s=doc.get("encoding_time_seconds"),
            restore_s=doc.get("restoration_time_seconds") or None,
            transport=r["degradation"] or cfg.get("removal_mode"),
            fill=r["restorer"] or r["inpainter"],
            mtime=run_mtime(results_root, r["hash"]),
        ))
    return out


# --------------------------------------------------------------------------
# Encode-throughput cells
# --------------------------------------------------------------------------

@dataclass
class BatchRow:
    batch: int
    n: int
    med: float
    lo: float
    hi: float
    spread: float
    qps: List[Any]
    when: float


@dataclass
class EncodeCell:
    codec: Optional[str]
    width: int
    height: int
    preset: Optional[str]
    batches: List[BatchRow] = field(default_factory=list)
    verdict: str = ""
    reason: str = ""

    @property
    def n(self) -> int:
        return sum(b.n for b in self.batches)


def build_encode_cells(timings: Sequence[Timing]) -> List[EncodeCell]:
    """Group baseline encodes by (codec, resolution, preset), split into
    acquisition batches, and rule on whether a pooled median may be quoted."""
    groups: Dict[Tuple, List[Timing]] = defaultdict(list)
    for t in timings:
        if t.component != "baselines" or t.encode_fps is None:
            continue
        groups[(t.codec, t.width, t.height, t.preset)].append(t)

    cells: List[EncodeCell] = []
    for key, ts in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ids = cluster_batches([t.mtime for t in ts])
        by_batch: Dict[int, List[Timing]] = defaultdict(list)
        for bid, t in zip(ids, ts):
            by_batch[bid].append(t)

        cell = EncodeCell(codec=key[0], width=key[1], height=key[2], preset=key[3])
        for bid in sorted(by_batch):
            members = by_batch[bid]
            fps = [t.encode_fps for t in members]
            m = median(fps)
            cell.batches.append(BatchRow(
                batch=bid, n=len(members), med=m, lo=min(fps), hi=max(fps),
                spread=iqr(fps) / m if m else 0.0,
                qps=sorted({t.qp for t in members if t.qp is not None}),
                when=median([t.mtime for t in members]),
            ))
        cell.verdict, cell.reason = judge_cell(cell)
        cells.append(cell)
    return cells


def judge_cell(cell: EncodeCell) -> Tuple[str, str]:
    """May this cell's pooled median be published as a throughput figure?

    The alarm this answers: pooling batches whose fps ranges do not overlap
    produces a median that describes none of them and can invert the ordering
    between resolutions.
    """
    if cell.n < 3:
        return "not measurable", f"n={cell.n}, too few runs for a median"
    if len(cell.batches) > 1:
        spans = [(b.lo, b.hi) for b in cell.batches]
        if ranges_disjoint(spans):
            desc = "; ".join(f"batch {b.batch}: n={b.n}, {b.lo:.1f}-{b.hi:.1f} fps"
                             for b in cell.batches)
            return ("not measurable",
                    f"{len(cell.batches)} batches with non-overlapping fps ({desc}) "
                    "-- the pooled median falls in a gap no run occupies")
    worst = max(b.spread for b in cell.batches)
    if worst > 1.0:
        return "wide", f"IQR/median={worst:.2f} within a batch; contention-limited"
    return "ok", ""


# --------------------------------------------------------------------------
# QP-matched resolution comparison (the alarm's actual test)
# --------------------------------------------------------------------------

def qp_matched_by_resolution(timings: Sequence[Timing], codec: str,
                             videos: Sequence[str],
                             qps: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Encode fps by height, holding codec, video set and QP set fixed.

    This is the comparison the pooled cells could not make: the cells differed
    in which QPs they contained, and QP moves encode time by more than
    resolution does in this corpus.
    """
    out: Dict[int, List[float]] = defaultdict(list)
    for t in timings:
        if (t.component != "baselines" or t.codec != codec
                or t.video not in videos or t.qp not in qps
                or t.encode_fps is None):
            continue
        out[t.height].append(t.encode_fps)
    return {h: {"n": len(v), "median": median(v), "lo": min(v), "hi": max(v)}
            for h, v in sorted(out.items()) if v}


def qp_effect(timings: Sequence[Timing], codec: str, height: int) -> Dict[Any, Dict[str, Any]]:
    """Encode fps by QP at one resolution -- the confounder, measured."""
    out: Dict[Any, List[float]] = defaultdict(list)
    for t in timings:
        if (t.component != "baselines" or t.codec != codec
                or t.height != height or t.qp is None or t.encode_fps is None):
            continue
        out[t.qp].append(t.encode_fps)
    return {q: {"n": len(v), "median": median(v)} for q, v in sorted(out.items())}


# --------------------------------------------------------------------------
# Restoration throughput
# --------------------------------------------------------------------------

def restoration_by_fill(timings: Sequence[Timing], width: int = 640,
                        height: int = 360) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Timing]] = defaultdict(list)
    for t in timings:
        if t.width != width or t.height != height:
            continue
        if t.restore_fps is None or t.fill in (None, "none"):
            continue
        groups[t.fill].append(t)

    rows = []
    for fill, ts in groups.items():
        fps = [t.restore_fps for t in ts]
        m = median(fps)
        share = [t.restore_s / (t.restore_s + (t.encode_s or 0.0))
                 for t in ts if t.restore_s and (t.restore_s + (t.encode_s or 0.0))]
        rows.append({
            "fill": fill, "n": len(ts), "median_fps": m,
            "lo": min(fps), "hi": max(fps), "spread": iqr(fps) / m if m else 0.0,
            "restore_share": median(share) if share else None,
            "videos": len({t.video for t in ts}),
        })
    return sorted(rows, key=lambda r: -r["median_fps"])


# --------------------------------------------------------------------------
# Cost attached to the map
# --------------------------------------------------------------------------

def map_with_cost(db: str, objective: str) -> Dict[str, Any]:
    """Wave 1's map, with the winner's throughput and the cheapest arm that is
    quality-tied to it, so cost is a visible axis rather than a footnote."""
    runs = wave1.load(db)
    by_op = wave1.score_arms(runs)
    cells = wave1.build_map(by_op, objective)

    fps_by_label: Dict[str, List[float]] = defaultdict(list)
    for arms in by_op.values():
        for a in arms:
            if a.fps:
                fps_by_label[a.label].append(a.fps)

    decided = [c for c in cells if c.winner is not None]
    rows = []
    for c in decided:
        arms = by_op[c.op]
        win = next((a for a in arms if a.label == c.winner), None)
        if win is None:
            continue
        elig = wave1.eligible(arms)
        # Quality-tied alternatives: within one BG-LPIPS JND of the winner.
        tied = [a for a in elig
                if abs(a.bg_lpips - win.bg_lpips) <= wave1.BG_LPIPS_JND and a.fps]
        faster = max(tied, key=lambda a: a.fps) if tied else None
        rows.append({
            "op": wave1._fmt_op(c.op),
            "winner": c.winner,
            "winner_fps": win.fps,
            "cheapest_tied": faster.label if faster else None,
            "cheapest_tied_fps": faster.fps if faster else None,
            "speedup": (faster.fps / win.fps) if faster and win.fps else None,
        })
    return {
        "objective": objective,
        "decided": len(decided),
        "rows": rows,
        "fps_by_label": {k: median(v) for k, v in sorted(fps_by_label.items())},
    }


def dominated_count(db: str) -> Tuple[int, int]:
    runs = wave1.load(db)
    by_op = wave1.score_arms(runs)
    arms = [a for v in by_op.values() for a in v]
    return sum(1 for a in arms if a.cost_dominated), len(arms)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    timings = load_timings(args.db, args.results)
    cells = build_encode_cells(timings)

    print("=" * 78)
    print("1. ENCODE THROUGHPUT -- per cell, split into acquisition batches")
    print("=" * 78)
    print(f"{'codec':9} {'res':10} {'preset':7} {'b':>2} {'n':>3} "
          f"{'med fps':>8} {'range':>15} {'QPs':>14}  verdict")
    for c in cells:
        for b in c.batches:
            print(f"{str(c.codec):9} {c.width}x{c.height:<5} {str(c.preset):7} "
                  f"{b.batch:>2} {b.n:>3} {b.med:>8.2f} "
                  f"{b.lo:>6.2f}-{b.hi:<8.2f} "
                  f"{(str(min(b.qps)) + '-' + str(max(b.qps))) if b.qps else 'n/a':>14}"
                  f"  {c.verdict if b is c.batches[0] else ''}")
        if c.reason:
            print(f"    -> {c.reason}")

    print()
    print("=" * 78)
    print("2. THE ALARM: svtav1 by resolution, QP- and video-matched")
    print("=" * 78)
    matched = qp_matched_by_resolution(timings, "svtav1", ("bear", "camel"),
                                       (43, 50, 58, 62))
    for h, v in matched.items():
        print(f"  {h:>5}p  n={v['n']:<3} median {v['median']:>7.2f} fps  "
              f"({v['lo']:.2f}-{v['hi']:.2f})")
    print("\n  QP effect at 640x360 (the confounder):")
    for q, v in qp_effect(timings, "svtav1", 360).items():
        print(f"    QP {q:<4} n={v['n']:<3} median {v['median']:>7.2f} fps")

    print()
    print("=" * 78)
    print("3. RESTORATION THROUGHPUT at 640x360")
    print("=" * 78)
    rest = restoration_by_fill(timings)
    print(f"{'fill':18} {'n':>4} {'vids':>5} {'med fps':>8} {'range':>16} "
          f"{'IQR/med':>8} {'restore share':>14}")
    for r in rest:
        share = f"{100 * r['restore_share']:.0f}%" if r["restore_share"] else "n/a"
        print(f"{r['fill']:18} {r['n']:>4} {r['videos']:>5} {r['median_fps']:>8.2f} "
              f"{r['lo']:>7.2f}-{r['hi']:<8.2f} {r['spread']:>8.2f} {share:>14}")

    print()
    print("=" * 78)
    print("4. COST ATTACHED TO THE MAP")
    print("=" * 78)
    dom, total = dominated_count(args.db)
    print(f"  cost-dominated arms: {dom}/{total}")
    payload: Dict[str, Any] = {"encode_cells": [], "matched": matched,
                               "restoration": rest,
                               "dominated": [dom, total], "map": {}}
    for obj in ("quality_first", "rate_first"):
        m = map_with_cost(args.db, obj)
        payload["map"][obj] = m
        swaps = [r for r in m["rows"] if r["speedup"] and r["speedup"] > 1.05]
        print(f"\n  {obj}-first: {m['decided']} decided cells; "
              f"{len(swaps)} have a quality-tied arm that is faster")
        for r in swaps[:12]:
            print(f"    {r['op']:34} {r['winner']:24} {r['winner_fps']:6.2f} fps "
                  f"-> {str(r['cheapest_tied']):24} {r['cheapest_tied_fps']:6.2f} fps "
                  f"({r['speedup']:.1f}x)")

    for c in cells:
        payload["encode_cells"].append({
            "codec": c.codec, "width": c.width, "height": c.height,
            "preset": c.preset, "verdict": c.verdict, "reason": c.reason,
            "batches": [vars(b) for b in c.batches],
        })
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
