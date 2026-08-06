"""Selective vs. uniform downsampling at matched rate — HOLE(sec:downsample-vs-uniform).

Referee 2 asked why PRESLEY's per-block downsampling differs from, or beats,
the uniform downscale-then-super-resolve that Netflix and YouTube already use.
The paper has only ever answered that architecturally. This is the measurement.

Both arms share the transport, the encoder, the restorer and the metrics; the
ONLY variable is whether the degradation is spatially selected. The uniform arm
is the same code path with the selection budget opened to every block and
foreground protection off, so per-block strength is pixel-identical between
arms (verified: `filter_frame_downsample` with a full-coverage selection is
byte-identical to an explicit uniform level).

Scope limit that must travel with any text derived from this: the uniform arm
transmits full-resolution-but-lowpassed frames, not a genuine lower-rendition
bitstream with fewer coded pixels. So this answers "does selectivity beat
uniformity with the transport held identical" -- NOT "does PRESLEY beat a real
industry rendition ladder."

Bounds are pre-registered in docs/PLAN_DOWNSAMPLE_VS_UNIFORM.md and are checked
here rather than eyeballed. A bound that fires is REPORTED as fired; it is never
re-fitted to accommodate the result.
"""
from __future__ import annotations

import json
import pathlib
import sys
from math import comb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from bd_rate import bd_rate, overlap_fraction  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
VIDEOS = ("bear", "camel", "dog", "india", "pigs", "tennis")

# A BD integral over a sliver of shared rate range is arithmetically valid and
# practically meaningless (see scripts/bd_rate.overlap_fraction). Curves below
# this are reported separately and excluded from the verdict, never averaged in.
MIN_OVERLAP = 0.50

# Pre-registered, docs/PLAN_DOWNSAMPLE_VS_UNIFORM.md section 4.
# Sign: BD-rate of SELECTIVE anchored on UNIFORM; negative = selective wins.
BANDS = {
    "bd_fg":  dict(best=-60.0, worst=+25.0, alarm=(-80.0, +50.0)),
    "bd_bg":  dict(best=-20.0, worst=+70.0, alarm=(-50.0, +120.0)),
    # FG-LPIPS delta is UNIFORM MINUS SELECTIVE; positive = uniform worse.
    "d_fglpips": dict(best=+0.15, worst=-0.02, alarm=(-0.05, +0.25)),
    "d_bits": dict(best=-60.0, worst=-15.0, alarm=(-100.0, 0.0)),
}


# The selective arm is PINNED BY HASH, from docs/PLAN_DOWNSAMPLE_VS_UNIFORM.md 3.1.
# It must not be re-derived by config query: up to 21 runs share a
# (video, qp, component, block_size, fg_protect, shrink_amount) key, spanning
# freeze/blur/noise/mean_fill degradations and a dozen restorers, so a query
# silently returns whichever the filesystem yielded last. Doing exactly that
# produced a non-monotonic rate ladder and a nonsense +711% BD-rate on bear.
SELECTIVE = {
    "bear":   ((43, "ceac3559f8af0c3f"), (51, "e2cb6bed165d69b1"),
               (58, "660bfa8e58ad4dc0"), (61, "aa00beae3eca0b00")),
    "camel":  ((42, "a1dc7f1557c09867"), (50, "c29c94b5c290f208"),
               (58, "fde83c22fc01c85c"), (62, "b06092cca2df6726")),
    "dog":    ((43, "dc189c206da6cd4d"), (50, "36d973b3d6662b9b"),
               (58, "c85b79fa26560bc3"), (62, "5d046a46384dd4de")),
    "india":  ((43, "474fbb90d358caf0"), (50, "a1fab6d2e641f27a"),
               (58, "93657ec7fe2751d9"), (62, "33c8f6d4dfd857e7")),
    "pigs":   ((43, "340195a69b34b413"), (50, "9cdf895adbad37bd"),
               (58, "b23228908f7d34a7"), (62, "065853196970874b")),
    "tennis": ((43, "6d169af7b8a90f24"), (50, "69f428bd0178726a"),
               (58, "8b2b11e91badd1e8"), (62, "1ec9cbffcdbf0988")),
}


# Per-video uniform rungs, where the default ladder does not bracket the
# selective range. The uniform arm codes 25-33% cheaper at the same QP, so on
# bear its default ladder (43/51/58/61) topped out at 353 kbps against a
# selective range reaching 614 -- only 0.47 overlap, below MIN_OVERLAP.
#
# Recalibrating rungs per video is the established practice here (tab:av1-breadth
# did the same so its baseline PSNR range would bracket the incumbents'). The
# replacement was chosen on a stated rule, NOT by maximising overlap after the
# fact: among all four-rung ladders that bracket [132, 614] kbps, take the one
# most evenly spaced in log-rate. That is (35, 43, 51, 58) -- the original ladder
# with its bottom rung swapped for a new qp35 run -- at log-evenness 1.31 and
# overlap 0.83. Three other ladders tie on overlap; this one wins on spacing.
UNIFORM_RUNGS = {"bear": (35, 43, 51, 58)}


def _read(h):
    p = RESULTS / h / "result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load():
    sel, uni = {}, {}
    for v, rungs in SELECTIVE.items():
        for qp, h in rungs:
            d = _read(h)
            if d is None or d.get("invariant_failures") or d.get("rate_control") != "crf":
                continue
            sel[(v, qp)] = d
    # The uniform arm is unambiguous: sa==80 with fg_protect off exists only here.
    for p in RESULTS.glob("*/result.json"):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        c = d.get("config") or {}
        if c.get("component") != "presley_ai" or c.get("codec") != "svtav1":
            continue
        if c.get("video") not in VIDEOS or c.get("block_size") != 8:
            continue
        if d.get("rate_control") != "crf" or d.get("invariant_failures"):
            continue
        if c.get("fg_protect") is False and c.get("shrink_amount") == 80:
            v, qp = c["video"], (c.get("codec_params") or {}).get("qp")
            allowed = UNIFORM_RUNGS.get(v)
            if allowed is not None and qp not in allowed:
                continue
            uni[(v, qp)] = d
    return sel, uni


def check_monotonic(store, label):
    """A ladder whose rate rises with QP is mispaired, not informative."""
    bad = []
    for v in VIDEOS:
        ks = sorted([k for k in store if k[0] == v], key=lambda k: k[1])
        rates = [rate(store[k]) for k in ks]
        if any(b >= a for a, b in zip(rates, rates[1:])):
            bad.append(f"{label}/{v}: " + ", ".join(f"qp{k[1]}={r:.0f}" for k, r in zip(ks, rates)))
    return bad


def rate(d):
    # presley_ai bills the transmitted bitstream, not the container.
    return d["transmitted_size_bytes"] * 8 / (d["video_frames"] / d["video_framerate"]) / 1000.0


def m(d, region, key):
    return d["metrics"][region][key]


def sign_p(k, n):
    """Exact two-tailed sign test. One-tailed p here would be the trap that
    hard rule 2b's significance layer exists to stop: 5/5 is p=0.0625, not 0.031."""
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) * 0.5 ** n for i in range(lo + 1)))


def status(name, value):
    b = BANDS[name]
    lo, hi = b["alarm"]
    if not (lo <= value <= hi):
        return "ALARM"
    inner = sorted((b["best"], b["worst"]))
    return "in band" if inner[0] <= value <= inner[1] else "outside band"


def main() -> int:
    sel, uni = load()
    common = sorted(set(sel) & set(uni))
    if not common:
        print("no paired cells found")
        return 1

    missing = [k for k in common
               if m(sel[k], "foreground", "lpips_mean") is None
               or m(uni[k], "foreground", "lpips_mean") is None]
    if missing:
        print(f"ABORT — {len(missing)} cells still lack FG-LPIPS; run the backfill first.")
        return 1

    # Gate: a ladder must be monotonic in rate, and the two arms must share
    # enough range for a BD integral to mean anything. Both are checked BEFORE
    # any number is reported, because both failed silently on the first pass.
    nonmono = check_monotonic(sel, "selective") + check_monotonic(uni, "uniform")
    if nonmono:
        print("ABORT — non-monotonic rate ladder (mispaired runs):")
        print("\n".join("  " + b for b in nonmono))
        return 1

    print("Selective vs uniform downsampling, SVT-AV1 fixed QP, 640x360, bs8.")
    print("BD-rate: selective anchored on uniform, NEGATIVE = selective wins.")
    print("d-FG-LPIPS: uniform minus selective, POSITIVE = uniform's FG is worse.\n")

    fired, rows, low_overlap = [], [], []
    for v in VIDEOS:
        # BD-rate compares two CURVES and does not require matched QPs -- each
        # arm may sit on its own recalibrated ladder, exactly as tab:av1-breadth
        # does. Per-rung deltas DO require matched QP and use the common rungs
        # only. Conflating the two silently dropped bear, whose uniform ladder
        # was recalibrated and therefore shares only three QPs with selective.
        s_keys = sorted([k for k in sel if k[0] == v], key=lambda k: k[1])
        u_keys = sorted([k for k in uni if k[0] == v], key=lambda k: k[1])
        if len(s_keys) < 4 or len(u_keys) < 4:
            print(f"{v}: selective {len(s_keys)} rungs / uniform {len(u_keys)} rungs, skipped")
            continue
        s_r = [rate(sel[k]) for k in s_keys]
        u_r = [rate(uni[k]) for k in u_keys]
        # BD-rate wants quality increasing; LPIPS is lower-is-better, so negate.
        s_fg = [-m(sel[k], "foreground", "lpips_mean") for k in s_keys]
        u_fg = [-m(uni[k], "foreground", "lpips_mean") for k in u_keys]
        s_bg = [-m(sel[k], "background", "lpips_mean") for k in s_keys]
        u_bg = [-m(uni[k], "background", "lpips_mean") for k in u_keys]

        ov = overlap_fraction(u_r, s_r)
        try:
            bd_fg = bd_rate(u_r, u_fg, s_r, s_fg)
            bd_bg = bd_rate(u_r, u_bg, s_r, s_bg)
        except Exception as e:                      # noqa: BLE001
            print(f"{v}: BD failed ({e})")
            continue
        if ov < MIN_OVERLAP:
            low_overlap.append((v, ov, bd_fg, bd_bg))
            continue

        shared = sorted(set(k[1] for k in s_keys) & set(k[1] for k in u_keys))
        if shared:
            d_lp = sum(m(uni[(v, q)], "foreground", "lpips_mean")
                       - m(sel[(v, q)], "foreground", "lpips_mean") for q in shared) / len(shared)
            d_bits = sum((rate(uni[(v, q)]) - rate(sel[(v, q)])) / rate(sel[(v, q)])
                         for q in shared) / len(shared) * 100
        else:
            d_lp = d_bits = float("nan")

        rows.append((v, bd_fg, bd_bg, d_lp, d_bits, ov, len(shared)))
        for nm, val in (("bd_fg", bd_fg), ("bd_bg", bd_bg),
                        ("d_fglpips", d_lp), ("d_bits", d_bits)):
            if val == val:                          # skip NaN
                st = status(nm, val)
                if st != "in band":
                    fired.append((v, nm, val, st))

    print(f"{'video':8s}{'BD-rate FG':>13}{'BD-rate BG':>13}{'d FG-LPIPS':>13}"
          f"{'d bits @QP':>13}{'overlap':>9}{'n@QP':>6}")
    for v, a, b, c, e, ov, ns in rows:
        print(f"{v:8s}{a:>12.1f}%{b:>12.1f}%{c:>13.4f}{e:>12.1f}%{ov:>9.2f}{ns:>6}")

    if rows:
        n = len(rows)
        n_sel_fg = sum(1 for r in rows if r[1] < 0)
        n_sel_bg = sum(1 for r in rows if r[2] < 0)
        print(f"\nselective wins FG BD-rate on {n_sel_fg}/{n} videos, "
              f"BG on {n_sel_bg}/{n}")
        print(f"  FG exact two-tailed sign p = {sign_p(n_sel_fg, n):.4f}")
        print(f"  BG exact two-tailed sign p = {sign_p(n_sel_bg, n):.4f}")
        print(f"  floor at n={n} is p={sign_p(0, n):.4f} (a clean sweep); "
              f"anything short of that cannot reach 0.05")

        # POST-HOC, and labelled as such: this hypothesis was generated by these
        # data, not pre-registered, so its concordance is hypothesis-GENERATING.
        # It is not a significance claim and must not be reported as one until
        # an independent set tests it.
        conc = sum(1 for r in rows
                   if (abs(r[3]) > 0.05 and r[1] < 0) or (abs(r[3]) <= 0.05 and r[1] > 0))
        print(f"\nPOST-HOC observation (NOT pre-registered, NOT a significance claim):")
        print(f"  selective wins iff uniform's FG damage is supra-JND: "
              f"{conc}/{n} concordant")
        print(f"  This hypothesis was generated by these data. Quoting its p-value")
        print(f"  would be candidate shopping. It needs an independent set to test.")

    print("\nPre-registered bound status:")
    if not fired:
        print("  all quantities inside their pre-registered bands")
    for v, nm, val, st in fired:
        print(f"  {st}: {v} {nm} = {val:.4f}   band={BANDS[nm]}")
    print("\nA fired bound is recorded as fired. Do not re-fit the band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
