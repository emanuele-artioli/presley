#!/usr/bin/env python3
"""W1g: does correcting the selection objective buy anything?

The article diagnoses the selection score as mis-specified -- it models bit
cost and has no term for how well a block survives restoration, and the two are
positively correlated, so it preferentially degrades what restores worst. This
races the correction against the incumbent.

  control    'removability' -- the existing score
  corrected  'restorability' -- that score divided by predicted damage

Everything else is held fixed: same blur, same hard foreground exclusion, same
budget, same encoder, same QP rungs, same restorer. Verified on bear that both
arms degrade exactly the same number of blocks.

Pre-registered before the numbers were read (PLAN_SUBMISSION_PREP.md 3.3):
BD-rate of corrected against control in -20%..+25%, sign not predicted. One
thing WAS known in advance and constrains the reading: the corrected rule
starts down on the rate axis (+3.6% bits on bear at QP 50 measured end to end),
so a win has to come from restoration quality rather than from bits.

Reported on both regions, because they answer different questions:

  BG-LPIPS  did the corrected rule actually protect the blocks that restore
            badly? This is the quantity the correction targets.
  FG-LPIPS  did it cost foreground quality? Both arms hard-exclude the
            foreground, so a large move here would indicate something other
            than selection changed and is a reason to distrust the run.

n=6 is the significance floor: an exact two-tailed sign test cannot reach 0.05
below it, and 6/6 gives exactly p=0.031.

Usage:
    python tools/analyze_selection_race.py --data-root .
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from math import comb

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bd_rate import BDError, bd_rate  # noqa: E402

BD_PLAUSIBLE = (-20.0, 25.0)

# Every ranking variant tried against this control, losers included. The Holm
# family is sized on this, not on the number that happened to look good.
CANDIDATES_TRIED = 1


def sign_p(k: int, n: int) -> float:
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) * 0.5 ** n for i in range(lo + 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    args = ap.parse_args()
    data_root = pathlib.Path(args.data_root).resolve()

    import yaml

    import presley  # noqa: F401
    from presley import db as _db
    from presley.runner import compute_experiment_hash

    entries = yaml.safe_load(open(data_root / "config" / "w1g_selection_race.yaml"))["experiments"]
    arms, flagged, missing = {}, [], []
    for cfg in entries:
        rule = cfg.get("selection_rule", "removability")
        d = _db.load_run(str(data_root / "results"), compute_experiment_hash(cfg))
        if d is None:
            missing.append((cfg["video"], rule, cfg["codec_params"]["qp"]))
            continue
        if d.get("invariant_failures"):
            flagged.append((cfg["video"], rule, cfg["codec_params"]["qp"]))
            continue
        arms.setdefault((rule, cfg["video"]), []).append(d)

    if missing:
        print(f"MISSING {len(missing)} run(s); the race is incomplete:")
        for v, r, qp in sorted(missing)[:10]:
            print(f"   {v} {r} qp{qp}")
    if flagged:
        print(f"EXCLUDED {len(flagged)} run(s) as not citable (invariant_failures):")
        for v, r, qp in sorted(flagged):
            print(f"   {v} {r} qp{qp}")
    print()

    videos = sorted({v for _, v in arms})
    alarms = []
    results = {}

    for region, label in (("background", "BG-LPIPS"), ("foreground", "FG-LPIPS")):
        print("=" * 72)
        print(f"BD-rate on {label}: corrected vs control "
              f"(negative = corrected needs fewer bits)")
        print("=" * 72)
        vals = {}
        for v in videos:
            a = arms.get(("removability", v), [])
            b = arms.get(("restorability", v), [])
            if len(a) < 4 or len(b) < 4:
                print(f"{v:12}{'n<4 rungs':>14}")
                continue
            try:
                val = bd_rate([x["actual_bitrate_bps"] for x in a],
                              [x["metrics"][region]["lpips_mean"] for x in a],
                              [x["actual_bitrate_bps"] for x in b],
                              [x["metrics"][region]["lpips_mean"] for x in b],
                              lower_is_better=True)
            except (BDError, KeyError, TypeError) as exc:
                print(f"{v:12}{'undefined':>14}   ({type(exc).__name__})")
                continue
            vals[v] = val
            flag = "" if BD_PLAUSIBLE[0] <= val <= BD_PLAUSIBLE[1] else "  !"
            if flag:
                alarms.append((label, v, val))
            print(f"{v:12}{val:>+13.1f}%{flag}")

        if vals:
            n = len(vals)
            wins = sum(1 for x in vals.values() if x < 0)
            p = sign_p(wins, n)
            p_holm = min(1.0, p * CANDIDATES_TRIED)
            print(f"\n  n={n}  median {np.median(list(vals.values())):+.1f}%  "
                  f"corrected better on {wins}/{n}  "
                  f"sign p={p:.4f}  p_Holm={p_holm:.4f}")
            if n < 6:
                print("  UNDERPOWERED: n<6 cannot reach 0.05 on an exact "
                      "two-tailed sign test, whatever the split.")
            results[label] = {"n": n, "wins": wins, "p": p, "p_holm": p_holm,
                              "median": float(np.median(list(vals.values()))),
                              "per_video": {k: round(x, 1) for k, x in vals.items()}}
        print()

    if alarms:
        print(f"!! ALARM -- outside the pre-registered band "
              f"[{BD_PLAUSIBLE[0]:+.0f}%, {BD_PLAUSIBLE[1]:+.0f}%]:")
        for label, v, val in alarms:
            print(f"   {label} {v}: {val:+.1f}%")
        print("Investigate before reporting as a finding.")
    else:
        print(f"no cell outside the pre-registered band "
              f"[{BD_PLAUSIBLE[0]:+.0f}%, {BD_PLAUSIBLE[1]:+.0f}%].")

    print("\nReading constraint stated in advance: the corrected rule starts DOWN\n"
          "on the rate axis (+3.6% bits on bear at QP 50), so any win must come\n"
          "from restoration quality rather than from bits, and a null here means\n"
          "the correction does not pay for the rate it costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
