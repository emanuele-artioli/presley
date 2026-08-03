#!/usr/bin/env python
"""Wave 2A -- does any content attribute predict transport choice?

Pre-registration, decision rule and bounds: docs/WAVE2A_CONTENT_AXIS.md.
Read that first; this tool only executes what is written there.

Three targets, from the quality-first operating map built by
``tools/build_operating_map.py``:

  T1  which arm wins among the separable cells  -- declared untestable by
      construction (Wave 1: 18 of 19 cells name the same arm). The tool
      *checks* that degeneracy rather than assuming it, and refuses to run a
      test against it.
  T2  separable vs tie, among contested cells.
  T3  no-eligible-arm, among all cells.

Four candidate attributes (k=4, fixed in the pre-registration), all from EVCA
scores already cached -- no GPU, no new runs:

  A1  motion magnitude                     mean TC over all blocks/frames
  A2  hole-region temporal instability     mean TC over foreground blocks
  A3  background texture energy            mean SC over non-foreground blocks
  A4  residual-information proxy           mean of EVCA's per-frame B column

The unit of analysis is the **video**, never the cell: attributes are constant
within a video, so cells are not independent observations.

Usage:
    python tools/analyze_content_axis.py --map map.json \
        --cache /home/itec/emanuele/presley/cache \
        --annotations /home/itec/emanuele/presley/dataset/annotations
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# --- pre-registered constants -------------------------------------------------

N_CANDIDATES = 4  # Holm correction uses this regardless of how many compute
MIN_VIDEOS_FOR_SIGNIFICANCE = 6  # hard rule 2b
RHO_EFFECT_THRESHOLD = 0.6
ALPHA = 0.05
N_PERMUTATIONS = 10_000
SEED = 0
FG_BLOCK_COVERAGE = 0.5
BLOCK_SIZE = 8

ATTRIBUTES = ("A1_motion", "A2_hole_instability", "A3_bg_texture", "A4_residual_info")


# --- statistics ---------------------------------------------------------------


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks, ties shared. Equivalent to scipy's 'average' method."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    sorted_vals = arr[order]
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho. NaN when either side is constant (rho undefined)."""
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_p(x: Sequence[float], y: Sequence[float], seed: int = SEED,
                  n_perm: int = N_PERMUTATIONS) -> tuple[float, float]:
    """Two-tailed permutation p for Spearman rho. Returns (rho, p)."""
    rho = spearman(x, y)
    if np.isnan(rho):
        return rho, float("nan")
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=float)
    hits = 0
    for _ in range(n_perm):
        if abs(spearman(x, rng.permutation(y_arr))) >= abs(rho) - 1e-12:
            hits += 1
    # +1/+1 so p is never exactly 0: a permutation test cannot license p=0.
    return rho, (hits + 1) / (n_perm + 1)


def holm(pvalues: Sequence[float], n_tests: int | None = None) -> list[float]:
    """Holm-Bonferroni adjusted p-values, monotone, clipped to 1.

    ``n_tests`` defaults to len(pvalues); the pre-registration fixes it at the
    number of *candidates*, so a candidate that failed to compute still costs
    correction rather than silently making the survivors look better.
    """
    m = len(pvalues) if n_tests is None else n_tests
    if m < len(pvalues):
        raise ValueError("n_tests cannot be smaller than the number of p-values")
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# --- content attributes -------------------------------------------------------


def _read_block_csv(path: str) -> np.ndarray:
    """EVCA per-block CSV: one row per block, one column per frame."""
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def foreground_block_mask(mask: np.ndarray, n_blocks: int, width: int, height: int,
                          block_size: int = BLOCK_SIZE,
                          coverage: float = FG_BLOCK_COVERAGE) -> np.ndarray:
    """Boolean per-block foreground flag, in EVCA's row-major block order.

    ``mask`` is a binary array at any resolution; it is nearest-resampled onto
    the (height/block, width/block) grid of block coverage fractions.
    """
    cols = width // block_size
    rows = n_blocks // cols
    if rows == 0:
        raise ValueError(f"{n_blocks} blocks is fewer than one row of {cols}")
    src_h, src_w = mask.shape
    ys = (np.arange(rows * block_size) * src_h // (rows * block_size)).clip(0, src_h - 1)
    xs = (np.arange(cols * block_size) * src_w // (cols * block_size)).clip(0, src_w - 1)
    resized = (mask[np.ix_(ys, xs)] > 0).astype(float)
    per_block = resized.reshape(rows, block_size, cols, block_size).mean(axis=(1, 3))
    flags = per_block.ravel() > coverage
    return flags[:n_blocks] if len(flags) >= n_blocks else np.pad(
        flags, (0, n_blocks - len(flags)))


def cache_dir_for(video: str, cache_root: str, preferred: str = "640x360") -> str | None:
    """Pre-registered choice: <basename>_640x360_bs8 if present, else the only _bs8."""
    # DAVIS clips sit directly under cache/; MOSEv2 and YouTube-VOS clips are
    # nested one level down (cache/mosev2/<clip>_<WxH>_bs8), so the video's own
    # path prefix has to be honoured rather than flattened to a basename.
    parent = os.path.join(cache_root, os.path.dirname(video))
    base = video.split("/")[-1]
    if not os.path.isdir(parent):
        return None
    candidates = sorted(
        d for d in os.listdir(parent)
        if d.startswith(base + "_") and d.endswith(f"_bs{BLOCK_SIZE}")
        and os.path.isfile(os.path.join(parent, d, "evca_SC_blocks.csv"))
    )
    if not candidates:
        return None
    for d in candidates:
        if preferred in d:
            return os.path.join(parent, d)
    return os.path.join(parent, candidates[0])


def first_annotation(video: str, annotations_root: str) -> np.ndarray | None:
    d = os.path.join(annotations_root, video)
    if not os.path.isdir(d):
        return None
    frames = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    if not frames:
        return None
    from PIL import Image  # local import: only needed when annotations are read

    return np.array(Image.open(os.path.join(d, frames[0])))


def content_attributes(video: str, cache_root: str, annotations_root: str) -> dict:
    """The four pre-registered attributes for one video. Missing -> None."""
    out = {a: None for a in ATTRIBUTES}
    out["source"] = None
    cdir = cache_dir_for(video, cache_root)
    if cdir is None:
        return out
    out["source"] = os.path.basename(cdir)
    dims = os.path.basename(cdir).split("_")[-2]
    width, height = (int(v) for v in dims.split("x"))

    sc = _read_block_csv(os.path.join(cdir, "evca_SC_blocks.csv"))
    tc = _read_block_csv(os.path.join(cdir, "evca_TC_blocks.csv"))
    out["A1_motion"] = float(tc.mean())

    raw_path = os.path.join(cdir, "evca_EVCA_reference_raw.csv")
    if os.path.isfile(raw_path):
        raw = np.loadtxt(raw_path, delimiter=",", skiprows=1, ndmin=2)
        out["A4_residual_info"] = float(raw[:, 0].mean())

    mask = first_annotation(video, annotations_root)
    if mask is not None:
        fg = foreground_block_mask(mask, tc.shape[0], width, height)
        if fg.any():
            out["A2_hole_instability"] = float(tc[fg].mean())
        if (~fg).any():
            out["A3_bg_texture"] = float(sc[~fg].mean())
    return out


# --- targets ------------------------------------------------------------------


@dataclass
class TargetRates:
    """Per-video outcome rate for one target, plus the counts behind it."""

    name: str
    rates: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def target_rates(cells: Iterable[dict], numerator: set[str], denominator: set[str],
                 name: str) -> TargetRates:
    res = TargetRates(name=name)
    hits: dict[str, int] = {}
    tot: dict[str, int] = {}
    for cell in cells:
        if cell["verdict"] not in denominator:
            continue
        video = cell["op"][0]
        tot[video] = tot.get(video, 0) + 1
        hits[video] = hits.get(video, 0) + (1 if cell["verdict"] in numerator else 0)
    for video, n in tot.items():
        res.rates[video] = hits[video] / n
        res.counts[video] = n
    return res


def winner_degeneracy(cells: Iterable[dict]) -> dict:
    """T1's structural check: is there any variance in *which* arm wins?"""
    winners: dict[str, int] = {}
    for cell in cells:
        if cell.get("separable"):
            winners[cell["winner"]] = winners.get(cell["winner"], 0) + 1
    total = sum(winners.values())
    top = max(winners.values()) if winners else 0
    return {
        "winners": winners,
        "n_separable": total,
        "modal_share": (top / total) if total else float("nan"),
        "degenerate": total > 0 and (top / total) >= 0.9,
    }


# --- reporting ----------------------------------------------------------------


def analyse_target(rates: TargetRates, attrs: dict[str, dict], seed: int = SEED,
                   n_perm: int = N_PERMUTATIONS) -> list[dict]:
    rows = []
    raw_p, keys = [], []
    for attr in ATTRIBUTES:
        videos = [v for v in sorted(rates.rates)
                  if attrs.get(v, {}).get(attr) is not None]
        x = [attrs[v][attr] for v in videos]
        y = [rates.rates[v] for v in videos]
        rho, p = permutation_p(x, y, seed=seed, n_perm=n_perm) if len(videos) >= 3 \
            else (float("nan"), float("nan"))
        rows.append({"attribute": attr, "n_videos": len(videos), "rho": rho, "p": p})
        if not np.isnan(p):
            raw_p.append(p)
            keys.append(attr)
    if raw_p:
        adj = holm(raw_p, n_tests=N_CANDIDATES)
        by_key = dict(zip(keys, adj))
        for row in rows:
            row["p_holm"] = by_key.get(row["attribute"], float("nan"))
    else:
        for row in rows:
            row["p_holm"] = float("nan")
    for row in rows:
        row["predictive"] = bool(
            row["n_videos"] >= MIN_VIDEOS_FOR_SIGNIFICANCE
            and not np.isnan(row["rho"])
            and abs(row["rho"]) >= RHO_EFFECT_THRESHOLD
            and row["p_holm"] < ALPHA
        )
        row["suggestive"] = bool(
            not row["predictive"] and not np.isnan(row["rho"])
            and abs(row["rho"]) >= RHO_EFFECT_THRESHOLD
        )
        row["alarm_rho"] = bool(not np.isnan(row["rho"]) and abs(row["rho"]) > 0.9)
    return rows


def dataset_confound(attrs: dict[str, dict], videos: Sequence[str]) -> dict[str, float]:
    """Rank correlation of each attribute with the DAVIS / newer-clip split."""
    out = {}
    for attr in ATTRIBUTES:
        vs = [v for v in videos if attrs.get(v, {}).get(attr) is not None]
        if len(vs) < 3:
            out[attr] = float("nan")
            continue
        x = [attrs[v][attr] for v in vs]
        y = [1.0 if "/" in v else 0.0 for v in vs]
        out[attr] = spearman(x, y)
    return out


def restrict(rates: TargetRates, min_cells: int = 1,
             drop: Sequence[str] = ()) -> TargetRates:
    """Pre-registered robustness subsets: drop thin or dominant videos."""
    out = TargetRates(name=rates.name)
    for video, n in rates.counts.items():
        if n >= min_cells and video not in drop:
            out.rates[video] = rates.rates[video]
            out.counts[video] = n
    return out


def coverage_confound(rates: TargetRates, attrs: dict[str, dict]) -> dict[str, float]:
    """Is an attribute really predicting how many cells a video happens to have?

    Not pre-registered; added after A1 fired, and reported as such. A video's
    cell count is a history of what was run, not a property of its content, so
    an attribute that tracks it is not a content predictor.
    """
    videos = sorted(rates.rates)
    counts = [rates.counts[v] for v in videos]
    out = {"outcome_vs_cellcount": spearman(counts, [rates.rates[v] for v in videos])}
    for attr in ATTRIBUTES:
        vs = [v for v in videos if attrs.get(v, {}).get(attr) is not None]
        out[attr] = spearman([attrs[v][attr] for v in vs],
                             [rates.counts[v] for v in vs])
    return out


def _fmt(value: float, spec: str = "6.3f") -> str:
    return "   n/a" if value is None or (isinstance(value, float) and np.isnan(value)) \
        else format(value, spec)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="JSON from build_operating_map.py")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--json", help="write the full result table here")
    args = ap.parse_args(argv)

    with open(args.map) as fh:
        cells = json.load(fh)["cells"]["quality_first"]

    print("=" * 78)
    print("WAVE 2A -- content axis. Pre-registration: docs/WAVE2A_CONTENT_AXIS.md")
    print("=" * 78)

    deg = winner_degeneracy(cells)
    print("\nT1 -- which arm wins (structural check, no test run)")
    for arm, n in sorted(deg["winners"].items(), key=lambda kv: -kv[1]):
        print(f"    {arm:34s} {n:3d}")
    print(f"    modal share {deg['modal_share']:.3f} over {deg['n_separable']} "
          f"separable cells -> {'DEGENERATE' if deg['degenerate'] else 'has variance'}")
    if deg["degenerate"]:
        print("    No attribute is tested against T1: the outcome is a constant.")

    t2 = target_rates(cells, {"separable"}, {"separable", "tie_within_threshold"},
                      "T2 separable-vs-tie")
    t3 = target_rates(cells, {"no_eligible_arm"},
                      {c["verdict"] for c in cells}, "T3 no-eligible-arm")

    videos = sorted({c["op"][0] for c in cells})
    attrs = {v: content_attributes(v, args.cache, args.annotations) for v in videos}

    missing = [v for v in videos if attrs[v]["A1_motion"] is None]
    print(f"\nAttributes computed for {len(videos) - len(missing)}/{len(videos)} videos")
    if missing:
        print(f"    no EVCA cache: {', '.join(missing)}")

    results = {"t1": deg, "targets": {}, "attributes": attrs}
    for target in (t2, t3):
        rows = analyse_target(target, attrs, n_perm=args.permutations)
        results["targets"][target.name] = {
            "rates": target.rates, "counts": target.counts, "rows": rows}
        print(f"\n{target.name}: n={len(target.rates)} videos, "
              f"{sum(target.counts.values())} cells")
        print(f"    {'attribute':22s} {'n':>3s} {'rho':>7s} {'p':>8s} "
              f"{'p_holm':>8s}  verdict")
        for row in rows:
            verdict = ("PREDICTIVE" if row["predictive"] else
                       "suggestive-only (underpowered)" if row["suggestive"] else
                       "negative")
            if row["n_videos"] < MIN_VIDEOS_FOR_SIGNIFICANCE:
                verdict = "descriptive only (n<6 videos)"
            flag = "  !ALARM |rho|>0.9" if row["alarm_rho"] else ""
            print(f"    {row['attribute']:22s} {row['n_videos']:3d} "
                  f"{_fmt(row['rho'], '7.3f')} {_fmt(row['p'], '8.4f')} "
                  f"{_fmt(row['p_holm'], '8.4f')}  {verdict}{flag}")

    print("\nRobustness (pre-registered): unequal cell counts per video")
    results["robustness"] = {}
    for target in (t2, t3):
        for label, subset in (
            ("videos with >=2 cells", restrict(target, min_cells=2)),
            ("without bear+camel", restrict(target, drop=("bear", "camel"))),
        ):
            rows = analyse_target(subset, attrs, n_perm=args.permutations)
            results["robustness"][f"{target.name} / {label}"] = rows
            print(f"  {target.name} -- {label} (n={len(subset.rates)} videos)")
            for row in rows:
                if row["n_videos"] >= 3:
                    print(f"      {row['attribute']:22s} n={row['n_videos']:2d} "
                          f"rho={_fmt(row['rho'], '6.3f')} "
                          f"p_holm={_fmt(row['p_holm'], '6.4f')}")

    print("\nConfound -- attribute vs cell count (added after A1 fired, not "
          "pre-registered)")
    for target in (t2, t3):
        cov = coverage_confound(target, attrs)
        results.setdefault("coverage_confound", {})[target.name] = cov
        print(f"  {target.name}: outcome vs cell count "
              f"rho={_fmt(cov['outcome_vs_cellcount'], '6.3f')}")
        for attr in ATTRIBUTES:
            print(f"      {attr:22s} rho={_fmt(cov[attr], '6.3f')}")

    conf = dataset_confound(attrs, videos)
    print("\nConfound -- attribute vs dataset provenance (DAVIS=0, MOSEv2/YT-VOS=1)")
    for attr, rho in conf.items():
        print(f"    {attr:22s} rho={_fmt(rho, '6.3f')}")
    results["dataset_confound"] = conf

    print("\n" + "=" * 78)
    print("ADJUDICATION -- a fired attribute must also survive its own checks")
    print("=" * 78)
    surviving = []
    for name, target in results["targets"].items():
        cov = results["coverage_confound"][name]
        robust = results["robustness"][f"{name} / videos with >=2 cells"]
        robust_by_attr = {r["attribute"]: r for r in robust}
        for row in target["rows"]:
            if not row["predictive"]:
                continue
            sub = robust_by_attr.get(row["attribute"], {})
            holds = bool(sub.get("n_videos", 0) >= 3 and not np.isnan(sub.get("p_holm", np.nan))
                         and sub["p_holm"] < ALPHA)
            cleaner = abs(cov[row["attribute"]]) < abs(row["rho"]) * 0.75
            print(f"  {name} / {row['attribute']}: rho={row['rho']:.3f}")
            print(f"      survives the >=2-cells subset: "
                  f"{'yes' if holds else 'NO'} "
                  f"(rho={_fmt(sub.get('rho', float('nan')), '.3f')}, "
                  f"p_holm={_fmt(sub.get('p_holm', float('nan')), '.4f')})")
            print(f"      cleaner than the cell-count confound: "
                  f"{'yes' if cleaner else 'NO'} "
                  f"(attr vs cell count rho={cov[row['attribute']]:.3f}, "
                  f"outcome vs cell count rho={cov['outcome_vs_cellcount']:.3f})")
            if holds and cleaner:
                surviving.append(f"{name}/{row['attribute']}")
            else:
                print("      -> WITHDRAWN: the association is not separable from "
                      "how many operating points the video happens to have been "
                      "run at, which is run history, not content.")
    if not surviving:
        print("  nothing fired, or everything that fired was withdrawn.")

    for label, rows in results["robustness"].items():
        n_fire = sum(r["predictive"] for r in rows)
        if n_fire >= 3:
            print(f"  !ALARM ({label}): {n_fire} attributes fired at once -- the "
                  "pre-registered reading is a shared confound among collinear "
                  "EVCA means, not independent discoveries. Not a finding.")

    print("\n" + "=" * 78)
    print("VERDICT: " + ("predictive: " + ", ".join(surviving) if surviving
                         else "NEGATIVE -- the map stays an empirical lookup table"))
    print("=" * 78)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
