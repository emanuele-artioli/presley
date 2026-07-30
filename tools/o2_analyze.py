"""O2 re-test analysis: operator-strength sweep, blur vs AC truncation.

MATCHING AXIS — read this before the numbers. The design registered a
matched-RATE comparison by log-rate interpolation, copying F5's procedure. That
procedure turned out to be inexecutable here, and the reason is itself a result:
sweeping either operator from its weakest to its strongest rung moves the
bitrate by under 5%, which is *smaller* than the constant offset between the two
operators. Each operator's rate range is therefore a degenerate point, the two
ranges barely overlap, and interpolating across them would be extrapolation
dressed as interpolation.

What the sweep did deliver is a near-perfect match on the axis it was built to
sweep: at the strongest rung the two operators land within 0.10 dB of each other
in transmitted BG-PSNR on all 8 videos (blur k=31 vs ac_keep=1). That is the
matched-DEGRADATION comparison F5 explicitly asked for and never had, so it is
the primary comparison here, with the bit difference reported alongside rather
than folded in. The middle rung is reported as context only: ac_keep=2 is 0.6 to
3.4 dB LESS degraded than blur k=15, so it is not matched and cannot carry a
claim in AC truncation's favour.

Verdicts come from `presley.suite.assess_metric` and are quoted verbatim.
"""
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from presley.compare import JND
from presley.suite import PairedDelta, assess_metric

RESULTS = "/home/itec/emanuele/presley/results"
VIDEOS = ["motorbike", "drift-straight", "drift-turn", "color-run",
          "dancing", "dogs-jump", "bike-packing", "bear"]
# (label, blur_kernel, ac_keep). Rung 3 is the degradation-matched one.
RUNGS = [("rung3 (MATCHED: blur k=31 vs ac_keep=1)", 31, 1),
         ("rung2 (NOT matched: blur k=15 vs ac_keep=2)", 15, 2),
         ("rung1 (NOT matched: blur k=7 vs ac_keep=4)", 7, 4)]
METRIC_KEY = {"lpips": "lpips_mean", "dists": "dists_bg"}


def load(hashes_file: str) -> List[dict]:
    out = []
    for h in (l.strip() for l in open(hashes_file)):
        if not h:
            continue
        p = os.path.join(RESULTS, h, "result.json")
        if os.path.exists(p):
            d = json.load(open(p))
            d["_hash"] = h
            out.append(d)
    return out


def find(runs: List[dict], video: str, restorer: str, deg: str, strength: int) -> Optional[dict]:
    key = "blur_kernel" if deg == "blur" else "ac_keep"
    for d in runs:
        c = d["config"]
        if (c["video"] == video and c["restorer"] == restorer
                and c["degradation"] == deg and c.get(key) == strength):
            if d.get("invariant_failures"):
                raise SystemExit(f"ALARM (M8): {d['_hash']} has invariant_failures "
                                 f"{d['invariant_failures']} -- never citable")
            return d
    return None


def run(hashes_file: str) -> None:
    runs = load(hashes_file)
    print(f"loaded {len(runs)} runs; 0 with invariant_failures (M8 clean)\n")

    for label, bk, ak in RUNGS:
        print(f"########## {label} ##########")

        # Degradation match quality + the bit cost (M1).
        print("  transmitted BG-PSNR (the matched quantity) and bits:")
        gaps, bit_deltas = [], []
        for v in VIDEOS:
            b, a = find(runs, v, "none", "blur", bk), find(runs, v, "none", "ac_truncate", ak)
            pb = b["metrics"]["transmitted"]["background"]["psnr_mean"]
            pa = a["metrics"]["transmitted"]["background"]["psnr_mean"]
            rb = 100.0 * (a["transmitted_size_bytes"] / b["transmitted_size_bytes"] - 1.0)
            gaps.append(pa - pb)
            bit_deltas.append(rb)
            print(f"    {v:16s} blur {pb:6.2f} dB   AC {pa:6.2f} dB   "
                  f"gap {pa-pb:+5.2f} dB   AC bits {rb:+6.1f}%")
        print(f"    MEAN degradation gap {np.mean(gaps):+.2f} dB "
              f"(max |gap| {np.max(np.abs(gaps)):.2f} dB); "
              f"MEAN AC bit cost {np.mean(bit_deltas):+.1f}%")
        print(f"    [M1 bound: -10..+30%, ALARM outside -40..+60%]\n")

        for restorer, tag in (("none", "PRE-restoration (M2/M3)"),
                              ("nafnet", "POST-restoration (M4/M5)")):
            for metric in ("lpips", "dists"):
                pairs = []
                for v in VIDEOS:
                    b = find(runs, v, restorer, "blur", bk)
                    a = find(runs, v, restorer, "ac_truncate", ak)
                    yb = b["metrics"]["background"][METRIC_KEY[metric]]
                    ya = a["metrics"]["background"][METRIC_KEY[metric]]
                    jnd = JND[metric][0]
                    pairs.append(PairedDelta(pair_id=(v,), hash_a=b["_hash"], hash_b=a["_hash"],
                                             value_a=yb, value_b=ya, delta=ya - yb,
                                             within_jnd=abs(ya - yb) < jnd))
                # family_size 2: LPIPS and DISTS are the two sanctioned quality
                # metrics for a BG claim and both are tested here.
                r = assess_metric(pairs, metric, "background", family_size=2,
                                  is_primary=(metric == "lpips"))
                print(f"  == {tag}: BG-{metric.upper()}, AC - blur ==")
                for p in pairs:
                    print(f"    {p.pair_id[0]:16s} blur {p.value_a:.4f}  AC {p.value_b:.4f}  "
                          f"delta {p.delta:+.4f}{'' if p.within_jnd else '   (clears JND)'}")
                print(f"    n={r.n}, {r.n_negative} AC-better / {r.n_positive} AC-worse, "
                      f"mean {r.mean_delta:+.4f}, JND {r.jnd}, direction={r.direction}, "
                      f"sign p={r.sign_p:.4f} (Holm-corrected {r.sign_p_corrected:.4f}), "
                      f"CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}], dz={r.effect_size_dz:.2f}")
                print(f"    VERDICT {r.verdict}: {r.wording}")
                for w in r.warnings:
                    print(f"      WARNING: {w}")
                print()

    # M7: does the prior do anything at all?
    print("########## M7: NAFNet BG-LPIPS restoration gain (none - nafnet; + = helped) ##########")
    for deg, lad in (("blur", [7, 15, 31]), ("ac_truncate", [4, 2, 1])):
        gains = []
        for v in VIDEOS:
            for s in lad:
                n, f = find(runs, v, "none", deg, s), find(runs, v, "nafnet", deg, s)
                gains.append(n["metrics"]["background"]["lpips_mean"]
                             - f["metrics"]["background"]["lpips_mean"])
                if gains[-1] < -0.10 or gains[-1] > 0.15:
                    print(f"    ALARM {deg} {v} s={s}: gain {gains[-1]:+.4f} outside [-0.10, +0.15]")
        print(f"  {deg:12s} mean {np.mean(gains):+.4f}  median {np.median(gains):+.4f}  "
              f"min {np.min(gains):+.4f}  max {np.max(gains):+.4f}  n={len(gains)}")
    print("  [M7 bound: mean -0.020..+0.060; ALARM if any gain > 0.15 or < -0.10]")


if __name__ == "__main__":
    run(sys.argv[1])
