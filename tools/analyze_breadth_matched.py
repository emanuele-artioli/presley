#!/usr/bin/env python3
"""Breadth on a matched corpus — data for `fig:breadth`.

The earlier breadth figure compared 33 ELVIS clips against 11 PRESLEY ones,
which invites the reader to compare two different corpora. This restricts to
the videos where the pristine baseline, ELVIS and PRESLEY all exist at both
QP 32 and 37, so every clip contributes to both configurations and the two are
read off the same set.

Fixed-QP only, runs with a non-empty `invariant_failures` skipped, rate billed
on the actual encoded bitrate. Writes Figures/breadth.json.
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "68e8b6bb11d0dd9e62a67aef" / "Figures" / "breadth.json"
QPS = (32, 37)


def family(v):
    if v.startswith("mosev2/"):
        return "MOSEv2 (held out)"
    if v.startswith("youtube_vos/"):
        return "YouTube-VOS (held out)"
    return "DAVIS"


def collect():
    R = collections.defaultdict(dict)
    for d in (ROOT / "results").iterdir():
        f = d / "result.json"
        if not f.is_file():
            continue
        try:
            j = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if j.get("invariant_failures"):
            continue
        c = j.get("config") or {}
        qp = (c.get("codec_params") or {}).get("qp")
        if (c.get("codec") != "x265" or c.get("width") != 640
                or c.get("height") != 360 or qp not in QPS):
            continue
        fg = (j.get("metrics") or {}).get("foreground") or {}
        rec = {"br": j.get("actual_bitrate_bps"), "fg_lpips": fg.get("lpips_mean")}
        key = (c.get("video"), qp)
        if c.get("component") == "baselines":
            R["base"][key] = rec
        elif c.get("component") == "elvis":
            R["elvis"][key] = rec
        elif (c.get("component") == "presley_ai" and c.get("degradation") == "downsample"
              and c.get("restorer") == "realesrgan"):
            R["presley"][key] = rec
    return R


def main() -> int:
    R = collect()
    vids = {v for v, _ in R["base"]}
    matched = sorted(v for v in vids
                     if all((v, q) in R[k] for q in QPS
                            for k in ("base", "elvis", "presley")))
    out = {"n_clips": len(matched), "qps": list(QPS),
           "by_family": dict(collections.Counter(family(v) for v in matched)),
           "quantity": ("median over QP of (configuration bitrate - baseline bitrate) "
                        "/ baseline bitrate, percent, paired within (video, QP)"),
           "arms": {}}
    for k, label in (("elvis", "ELVIS (block removal)"),
                     ("presley", "PRESLEY (downsample)")):
        per = []
        for v in matched:
            ds = [(R[k][(v, q)]["br"] - R["base"][(v, q)]["br"])
                  / R["base"][(v, q)]["br"] * 100.0 for q in QPS]
            fgd = [R[k][(v, q)]["fg_lpips"] - R["base"][(v, q)]["fg_lpips"]
                   for q in QPS
                   if R[k][(v, q)]["fg_lpips"] is not None
                   and R["base"][(v, q)]["fg_lpips"] is not None]
            per.append({"video": v, "family": family(v),
                        "bitrate_delta_pct": round(statistics.median(ds), 1),
                        "fg_lpips_delta": round(statistics.median(fgd), 4) if fgd else None})
        saving = sum(1 for p in per if p["bitrate_delta_pct"] < 0)
        out["arms"][label] = {
            "per_clip": per, "n_clips": len(per), "clips_saving_bits": saving,
            "clips_costing_bits": len(per) - saving,
            "median_pct": round(statistics.median(p["bitrate_delta_pct"] for p in per), 1),
            "saving_with_foreground_cost": [
                p["video"] for p in per
                if p["bitrate_delta_pct"] < 0 and (p["fg_lpips_delta"] or 0) > 0],
        }
        print(f"{label:24s} n={len(per)}  saves {saving}/{len(per)}  "
              f"median {out['arms'][label]['median_pct']:+.1f}%")
    OUT.write_text(json.dumps({"data": out, "figure": "breadth"}, indent=1,
                              sort_keys=True) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
