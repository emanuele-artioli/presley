#!/usr/bin/env python3
"""W1f: does the bitrate saving survive at higher resolution?

BD-rate of the PRESLEY arm against the pristine baseline, per (video,
resolution), on the masked perceptual metrics the article's verdicts rest on
(BG-LPIPS for generative restoration, FG-LPIPS for foreground protection).
Then an exact two-tailed sign test across videos *within* each resolution --
n=6 is the significance floor, so each rung is a claim rather than a
description, but only just.

Three things this reports rather than hides:

**The rungs are not in matched regimes.** At fixed QP the higher resolutions
run at roughly half the bits per pixel, i.e. MORE starved -- which is the
regime that favours the method. Any growth of the saving with resolution is
confounded with that and is reported as such, never attributed to resolution
alone. `build_resolution_ladder.py --check` prints the bpp table.

**Runs with a non-empty `invariant_failures` are excluded and named.** Five
camel runs at 720p/1080p trip the saturation check. Dropping them silently
would leave a hole in exactly one clip at exactly the resolutions under test.
Both totals are printed so the exclusion's effect is visible.

**BD-rate needs overlapping quality ranges.** Where the two curves do not
overlap the cell is reported as undefined, not as zero.

Usage:
    python tools/analyze_resolution_ladder.py --data-root .
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from math import comb

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bd_rate import BDError, bd_rate  # noqa: E402

# Pre-registered before the numbers were read (PLAN_SUBMISSION_PREP.md 3.2).
# A cell outside this band is an ALARM to investigate, not a finding to report.
BD_PLAUSIBLE = (-35.0, 10.0)

# `banded` marks the metric the pre-registered range was actually stated for.
#
# The band belongs to Goal 1 -- the bitrate saving at matched FOREGROUND
# quality. Applying it to BG-LPIPS is a category error, and it fired on 13 of
# 18 cells the first time this ran. BD-rate on BG-LPIPS against a PRISTINE
# baseline asks how many extra bits PRESLEY needs to match the baseline's
# background; PRESLEY degrades that background on purpose and restores it
# generatively, so a positive value there is the accepted, priced cost of
# relocation rather than a defect. It is reported, and never banded.
METRICS = (
    ("foreground", "lpips_mean", "FG-LPIPS", True, True),
    ("background", "lpips_mean", "BG-LPIPS (priced cost, not banded)", True, False),
)


def sign_p(k: int, n: int) -> float:
    """Exact two-tailed sign test. n<=5 cannot reach 0.05 at any split."""
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) * 0.5 ** n for i in range(lo + 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--include-flagged", action="store_true",
                    help="include runs with non-empty invariant_failures (NOT citable)")
    args = ap.parse_args()

    data_root = pathlib.Path(args.data_root).resolve()
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import yaml

    import presley  # noqa: F401
    from presley import db as _db
    from presley.runner import compute_experiment_hash

    def load_arm(path):
        entries = yaml.safe_load(open(data_root / path))["experiments"]
        out, flagged = {}, []
        for cfg in entries:
            d = _db.load_run(str(data_root / "results"), compute_experiment_hash(cfg))
            if d is None:
                continue
            if d.get("invariant_failures") and not args.include_flagged:
                flagged.append((cfg["video"], cfg["width"], cfg["height"],
                                cfg["codec_params"]["qp"]))
                continue
            key = (cfg["video"], cfg["width"], cfg["height"])
            out.setdefault(key, []).append(d)
        return out, flagged

    base, base_flagged = load_arm("config/w1f_ladder_baselines.yaml")
    pres, pres_flagged = load_arm("config/w1f_ladder_presley.yaml")

    if pres_flagged or base_flagged:
        print("EXCLUDED as not citable (non-empty invariant_failures):")
        for v, w, h, qp in sorted(pres_flagged + base_flagged):
            print(f"   {v} {w}x{h} qp{qp}")
        print()

    resolutions = sorted({k[1:] for k in pres}, key=lambda wh: wh[0])
    alarms = []

    for region, metric, label, lower_better, banded in METRICS:
        print("=" * 78)
        print(f"BD-rate on {label}  (negative = PRESLEY needs fewer bits for equal quality)")
        print("=" * 78)
        print(f"{'video':14}" + "".join(f"{w}x{h}".rjust(16) for w, h in resolutions))

        per_res = {wh: [] for wh in resolutions}
        for video in sorted({k[0] for k in pres}):
            cells = []
            for wh in resolutions:
                key = (video,) + wh
                b, p = base.get(key, []), pres.get(key, [])
                if len(b) < 4 or len(p) < 4:
                    cells.append(f"{'n<4':>16}")
                    continue
                try:
                    val = bd_rate(
                        [x["actual_bitrate_bps"] for x in b],
                        [x["metrics"][region][metric] for x in b],
                        [x["actual_bitrate_bps"] for x in p],
                        [x["metrics"][region][metric] for x in p],
                        lower_is_better=lower_better)
                except (BDError, KeyError, TypeError):
                    cells.append(f"{'no overlap':>16}")
                    continue
                per_res[wh].append(val)
                flag = ""
                if banded and not (BD_PLAUSIBLE[0] <= val <= BD_PLAUSIBLE[1]):
                    flag = " !"
                    alarms.append((label, video, wh, val))
                cells.append(f"{val:>+14.1f}%{flag}")
            print(f"{video:14}" + "".join(cells))

        print(f"\n{'':14}{'n':>6}{'median':>10}{'wins':>8}{'sign p':>10}")
        for wh in resolutions:
            vals = per_res[wh]
            if not vals:
                continue
            n = len(vals)
            wins = sum(1 for v in vals if v < 0)
            print(f"{f'{wh[0]}x{wh[1]}':14}{n:>6}{np.median(vals):>+10.1f}"
                  f"{f'{wins}/{n}':>8}{sign_p(wins, n):>10.4f}")
        print()

    if alarms:
        print("!! ALARM -- outside the pre-registered band "
              f"[{BD_PLAUSIBLE[0]:+.0f}%, {BD_PLAUSIBLE[1]:+.0f}%]:")
        for label, video, wh, val in alarms:
            print(f"   {label} {video} {wh[0]}x{wh[1]}: {val:+.1f}%")
        print("\nInvestigate implementation / eval / data before reporting any of these\n"
              "as a finding. Do not cite until the alarm is closed or the band is\n"
              "revised with a stated reason.")
    else:
        print(f"no cell outside the pre-registered band "
              f"[{BD_PLAUSIBLE[0]:+.0f}%, {BD_PLAUSIBLE[1]:+.0f}%].")

    print("\nREGIME CAVEAT: at fixed QP the higher rungs run at roughly half the bits\n"
          "per pixel, i.e. more starved -- the regime that favours the method. Growth\n"
          "of the saving with resolution is confounded with that and must not be\n"
          "attributed to resolution alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
