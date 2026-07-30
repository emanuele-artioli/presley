#!/usr/bin/env python3
"""S1b decision: Arm B (oracle damage-aware) vs Arm A (naive graded) and Arm C (binary).

Verdicts come from presley.suite.assess_metric so the mandated wording is
quoted, never paraphrased. Bounds were pre-registered in
docs/WAVE1_FALSIFIERS.md before any of these numbers existed.
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/home/itec/emanuele/presley/.claude/worktrees/agent-adc96e2b4b84b5357/src")

from presley.compare import JND, REGION_METRIC_KEYS  # noqa: E402
from presley.suite import PairedDelta, assess_metric, holm_adjust, sign_test_p  # noqa: E402

RESULTS = "/home/itec/emanuele/presley/results"
VIDEOS = ["bear", "motorbike", "drift-straight", "drift-turn",
          "color-run", "dancing", "dogs-jump", "bike-packing"]

ARM_A = {"bear": "c4c57169c454d595", "motorbike": "b5e34e9535e95f55",
         "drift-straight": "4e37c0044e896863", "drift-turn": "7f24116293981c8c",
         "color-run": "df95fc53aa9daa06", "dancing": "37a5ba4672377d19",
         "dogs-jump": "a65c763b51ecdee1", "bike-packing": "cb8e4b838abf0631"}
ARM_C = {"bear": "6330a540950be357", "motorbike": "79d911465fed8e66",
         "drift-straight": "649a949e211996b4", "drift-turn": "b881ce63a6c6b4cf",
         "color-run": "ff1d96789ede7861", "dancing": "9fde9ecef2bd01e5",
         "dogs-jump": "2c12da814dc87f63", "bike-packing": "b70dab8d332719f2"}


def load(h):
    return json.load(open(os.path.join(RESULTS, h, "result.json")))


def arm_b_hashes():
    out = {}
    for p in glob.glob(os.path.join(RESULTS, "*", "result.json")):
        d = json.load(open(p))
        if d.get("config", {}).get("downsample_level_map"):
            out[d["config"]["video"]] = os.path.basename(os.path.dirname(p))
    return out


def metric_value(d, region, metric):
    key = REGION_METRIC_KEYS.get(region, {}).get(metric)
    if key is None:
        return None
    m = d.get("metrics", {})
    node = m.get(region, m)
    v = node.get(key)
    return None if v is None else float(v)


def compare(name, base_map, test_map, region="background"):
    print(f"\n{'='*74}\n{name}  (region={region})\n{'='*74}")
    rates = []
    for v in VIDEOS:
        a, b = load(base_map[v]), load(test_map[v])
        ra, rb = a["actual_bitrate_bps"], b["actual_bitrate_bps"]
        rates.append((v, ra, rb, 100 * (rb - ra) / ra))
    print(f"{'video':16}{'base kbps':>11}{'test kbps':>11}{'delta':>9}")
    for v, ra, rb, pc in rates:
        print(f"{v:16}{ra/1000:>11.1f}{rb/1000:>11.1f}{pc:>8.2f}%")
    dr = [r[3] for r in rates]
    npos = sum(1 for x in dr if x > 0)
    print(f"mean rate delta {sum(dr)/len(dr):+.2f}%   "
          f"{npos}/{len(dr)} higher   two-tailed sign p="
          f"{sign_test_p(npos, len(dr)-npos):.4f}")

    results, pvals = [], []
    for metric in [m for m in REGION_METRIC_KEYS.get(region, {}) if m in JND and JND[m][2]]:
        pairs = []
        for v in VIDEOS:
            ha, hb = base_map[v], test_map[v]
            va = metric_value(load(ha), region, metric)
            vb = metric_value(load(hb), region, metric)
            if va is None or vb is None:
                continue
            jnd = JND[metric][0]
            pairs.append(PairedDelta((v,), ha, hb, va, vb, vb - va, abs(vb - va) < jnd))
        if len(pairs) < 2:
            print(f"  {metric:8} SKIPPED -- only {len(pairs)} usable pairs "
                  f"(missing metric: backfill needed)")
            continue
        r = assess_metric(pairs, metric, region, family_size=1,
                          is_primary=(metric == "lpips"))
        results.append(r)
        if r.sign_p is not None:
            pvals.append(r.sign_p)
    for r in results:
        star = " [PRIMARY]" if r.is_primary else ""
        print(f"\n  {r.metric}{star}: mean delta {r.mean_delta:+.4f}  "
              f"n={r.n} ({r.n_positive}+/{r.n_negative}-)  dir={r.direction}  "
              f"sign_p={r.sign_p}  corrected={r.sign_p_corrected}  "
              f"clears_jnd={r.clears_jnd}  pairs_clearing={r.pairs_clearing_jnd}")
        print(f"    verdict: {r.verdict}")
        print(f"    wording: {r.wording}")
    return results


def main():
    ab = arm_b_hashes()
    missing = [v for v in VIDEOS if v not in ab]
    if missing:
        print(f"MISSING Arm-B runs for: {missing}")
        return 1
    print("Arm B hashes:", json.dumps(ab, indent=1))
    for v in VIDEOS:
        d = load(ab[v])
        inv = d.get("invariant_failures")
        if inv:
            print(f"  !! {v} has invariant_failures: {inv} -- NOT CITABLE")
    compare("Arm B vs Arm A  (matched histogram: does damage-aware assignment beat score?)",
            ARM_A, ab)
    compare("Arm B vs Arm C  (THE DECISION: does the oracle beat plain binary?)",
            ARM_C, ab)
    compare("Arm B vs Arm C -- foreground (must be untouched)", ARM_C, ab, region="foreground")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
