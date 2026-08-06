#!/usr/bin/env python3
"""The four-link chain, each link anchored to its OWN encoder — D1. DESCRIPTIVE.

`baseline -> roi -> elvis -> presley_ai` cannot be measured as a chain on any
shared operating point, and the reason is structural rather than a sampling gap:
the only encoder whose native ROI moves bits measurably is kvazaar, and
`elvis`/`presley_ai` never run kvazaar. Zero operating points carry all four
components.

So each link is expressed as a delta against **its own encoder's pristine
fixed-QP baseline**, and the normalized deltas are set side by side. Two rules
make that defensible:

  * Operating points are matched on a CODEC-INDEPENDENT coordinate -- the
    pristine baseline's own delivered quality. Never on QP (x265 32 is not
    svtav1 43 is not a kvazaar binary-searched base QP) and never on absolute
    bitrate, which IS the codec-efficiency difference we are trying to remove.
  * Only the deltas are compared, never raw values across encoders.

## What this can and cannot support

CAN: "within its own encoder, each stage does X." Three separately-anchored
effects reported over the same content.

CANNOT, and these phrasings are banned in the paper:
  - "the chain holds" as an ordering
  - "each stage adds" / "cumulative" / "compounding" / "end-to-end gain of X%"
  - "PRESLEY beats codec ROI by X%"

Composition is untested AND mechanistically doubtful: ROI moves bits TOWARD the
foreground, while ELVIS and PRESLEY sacrifice background to do the same thing.
Stacking them would double-charge one background budget.

## Why there is no p-value here

The chain exists on six videos. At n=6 the exact two-tailed sign test floors at
p=0.031, which admits a Holm family of exactly ONE test -- and a 4-link x 2-axis
table is eight. The chain therefore cannot carry a significance claim, and does
not need to: "scope" asks which content and which operating points a behaviour
holds on, which is a descriptive question. The significance budget is spent
where it buys something, on the n=13 matched-rate result.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bd_rate import BDError, bd_rate, overlap_fraction  # noqa: E402

RESULTS = REPO_ROOT / "results"
JND = {"psnr": 0.5, "lpips": 0.05}

# The six videos on which the four-link chain exists at all.
CHAIN_VIDEOS = ("bear", "camel", "dog", "india", "pigs", "tennis")

# One arm per component, so a "link" is one configuration rather than an average
# over every configuration that happened to run at that operating point.
ARMS = {
    "elvis": {"removal_mode": "blackout", "inpainter": "propainter",
              "fg_protect": True, "shrink_amount": 0.25},
    "presley_ai": {"degradation": "downsample", "restorer": "realesrgan",
                   "fg_protect": True, "shrink_amount": 0.25},
}


def load():
    """{(component, video, codec): {key: doc}} over citable fixed-QP runs."""
    out = collections.defaultdict(dict)
    for p in RESULTS.glob("*/result.json"):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        c = d.get("config") or {}
        if c.get("video") not in CHAIN_VIDEOS:
            continue
        if d.get("rate_control") not in ("cqp", "crf") or d.get("invariant_failures"):
            continue
        comp = c.get("component")
        codec = c.get("codec") or c.get("roi_method")
        if comp == "roi":
            if c.get("roi_method") != "kvazaar":
                continue
            if (c.get("alpha"), c.get("beta"), c.get("block_size")) != (0.5, 0.5, 8):
                continue
            key = c.get("target_bitrate")
        elif comp == "baselines":
            key = c.get("target_bitrate") if codec == "kvazaar" else (c.get("codec_params") or {}).get("qp")
        elif comp in ("elvis", "presley_ai"):
            if c.get("block_size") != 8:
                continue
            # Pin the arm. Filtering on component alone averages over every
            # degradation and restorer that ever ran at that QP -- which showed
            # up as 7 matched points on a 4-rung ladder before this was added.
            if any(c.get(k) != v for k, v in ARMS[comp].items()):
                continue
            key = (c.get("codec_params") or {}).get("qp")
        else:
            continue
        # Resolution belongs in the key. bear and camel also have 1920x1080 and
        # 1280x720 runs at the same QPs, and mixing a 1080p arm with a 360p
        # baseline yields a confident, meaningless delta.
        out[(comp, c["video"], codec, c.get("width"), c.get("height"))][key] = d
    return out


def m(d, region, name):
    return ((d.get("metrics", {}).get(region, {}) or {}).get(name))


def rate(d):
    v = d.get("transmitted_size_bytes")
    if v:
        return v * 8 / (d["video_frames"] / d["video_framerate"]) / 1000.0
    return d["actual_bitrate_bps"] / 1000.0


def link_delta(arm, base, keys):
    """Per-operating-point deltas of an arm against its own-codec baseline."""
    rows = []
    for k in keys:
        a, b = arm.get(k), base.get(k)
        if not (a and b):
            continue
        fg_a, fg_b = m(a, "foreground", "psnr_mean"), m(b, "foreground", "psnr_mean")
        bg_a, bg_b = m(a, "background", "psnr_mean"), m(b, "background", "psnr_mean")
        if None in (fg_a, fg_b, bg_a, bg_b):
            continue
        rows.append({
            "coord": m(b, "background", "lpips_mean"),   # baseline-only, so uncontaminated
            "dbits": (rate(a) - rate(b)) / rate(b) * 100.0,
            "dfg": fg_a - fg_b,
            "dbg": bg_a - bg_b,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    data = load()

    print("=" * 88)
    print("THE FOUR-LINK CHAIN -- each link vs ITS OWN encoder's pristine baseline")
    print("DESCRIPTIVE: n=6, at the significance floor. No p-values, by design.")
    print("All links at 640x360; other renditions are excluded, not mixed in.")
    print("=" * 88)

    W, H = 640, 360      # every chain link is measured at this rendition
    links = [
        ("roi vs baseline (kvazaar)", "roi", "baselines", "kvazaar", "kvazaar"),
        ("elvis vs baseline (svtav1)", "elvis", "baselines", "svtav1", "svtav1"),
        ("presley_ai vs baseline (svtav1)", "presley_ai", "baselines", "svtav1", "svtav1"),
        ("elvis vs baseline (x265)", "elvis", "baselines", "x265", "x265"),
        ("presley_ai vs baseline (x265)", "presley_ai", "baselines", "x265", "x265"),
    ]

    summary = {}
    for label, arm_c, base_c, arm_k, base_k in links:
        print(f"\n--- {label} ---")
        print(f"  {'video':10}{'n_pts':>6}{'d bits %':>11}{'d FG dB':>10}"
              f"{'d BG dB':>10}{'FG perceptible?':>17}")
        per_video = []
        for v in CHAIN_VIDEOS:
            arm = data.get((arm_c, v, arm_k, W, H), {})
            base = data.get((base_c, v, base_k, W, H), {})
            keys = sorted(set(arm) & set(base), key=lambda x: (x is None, x))
            rows = link_delta(arm, base, keys)
            if not rows:
                continue
            n = len(rows)
            db = sum(r["dbits"] for r in rows) / n
            dfg = sum(r["dfg"] for r in rows) / n
            dbg = sum(r["dbg"] for r in rows) / n
            per_video.append((v, n, db, dfg, dbg))
            print(f"  {v:10}{n:>6}{db:>+11.1f}{dfg:>+10.2f}{dbg:>+10.2f}"
                  f"{('YES' if abs(dfg) >= JND['psnr'] else 'sub-JND'):>17}")
        if not per_video:
            print("  (no matched operating points)")
            continue
        nv = len(per_video)
        mb = sorted(r[2] for r in per_video)[nv // 2]
        mfg = sorted(r[3] for r in per_video)[nv // 2]
        mbg = sorted(r[4] for r in per_video)[nv // 2]
        pos_fg = sum(1 for r in per_video if r[3] > 0)
        perceptible = sum(1 for r in per_video if abs(r[3]) >= JND["psnr"])
        print(f"  {'MEDIAN':10}{nv:>6}{mb:>+11.1f}{mfg:>+10.2f}{mbg:>+10.2f}"
              f"{f'{perceptible}/{nv} videos':>17}")
        print(f"     FG improves on {pos_fg}/{nv} videos; "
              f"perceptible (>= {JND['psnr']} dB) on {perceptible}/{nv}")
        summary[label] = dict(n_videos=nv, median_dbits=mb, median_dfg=mfg,
                              median_dbg=mbg, fg_positive=pos_fg,
                              fg_perceptible=perceptible)

    print("\n" + "=" * 88)
    print("READING THIS TABLE")
    print("  Each block is a delta against a DIFFERENT encoder's own baseline.")
    print("  Compare the deltas; never the raw values, and never subtract one")
    print("  block from another -- that would be a composition claim, and")
    print("  composition is untested. ROI moves bits TOWARD the foreground while")
    print("  ELVIS/PRESLEY sacrifice background to do the same thing, so stacking")
    print("  them would double-charge one background budget.")
    print("  n=6 floors the sign test at p=0.031 -> a Holm family of ONE. This is")
    print("  reported descriptively and carries no significance claim.")
    print("=" * 88)

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
