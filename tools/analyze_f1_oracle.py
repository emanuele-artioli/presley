#!/usr/bin/env python3
"""F1: how much of the EVCA proxy's apparent skill survives inter coding?

The paper's claim (a) -- top-25% of superblocks by EVCA frees 93.0-99.4% of the
bits a perfect oracle would free -- was measured all-intra, where a block's cost
is essentially its own spatial detail and the correlation is near-tautological.
`NEXT(sec:implementation)` blocks that figure from any reviewer-visible sentence
until it is re-measured under inter coding, through the runner, with a hash.

Bounds and the decision rule were pre-registered in docs/F1_ORACLE_BITS.md before
any of these runs existed. Every headline prints next to its band; a breach is
labelled a BREACH.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from presley import db  # noqa: E402

# From docs/F1_ORACLE_BITS.md, quoted rather than re-derived.
BOUNDS = {
    "rho_each":   (0.30, 0.85, "per-video Spearman rho (SC vs marginal bits)"),
    "rho_mean":   (0.45, 0.75, "mean Spearman rho"),
    "cap_each":   (0.70, 0.95, "per-video capture ratio at top-25%"),
    "cap_mean":   (0.70, 0.95, "mean capture ratio (expected ~0.85)"),
    "oracle_frac": (0.10, 0.45, "oracle top-25% share of total bits"),
}
ALARMS = {
    "rho_tautology": (0.95, "rho above +0.95 reproduces the all-intra tautology -- "
                            "verify the encode is really inter before reporting"),
    "rho_useless":   (0.15, "mean rho below +0.15 means the proxy is near-useless under "
                            "realistic coding -- a paper-changing negative; check for an "
                            "implementation bug (score alignment, SB grid) first"),
    "cap_tautology": (0.99, "capture above 99% is the tautology again"),
    "cap_broken":    (0.50, "capture below 50% inverts the claim -- paper-changing; "
                            "clear an implementation review first"),
}


def band(key, value):
    lo, hi, label = BOUNDS[key]
    ok = lo <= value <= hi
    return f"[{label}: {lo:g}..{hi:g} -> {'in bounds' if ok else '*** BREACH ***'}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    conn = db.connect(args.results_dir)
    runs = []
    for row in conn.execute(
            "SELECT hash, video FROM runs WHERE component = 'probe_oracle_bits' ORDER BY video"):
        doc = db.get_run(conn, row["hash"])
        if doc.get("invariant_failures"):
            print(f"UNCITABLE {row['hash']} ({row['video']}): {doc['invariant_failures']}")
            continue
        p = doc.get("oracle_bits_probe")
        if p:
            runs.append((row["video"], row["hash"], p))

    if not runs:
        print("no citable probe_oracle_bits runs found")
        return

    print("=" * 84)
    print("F1: EVCA cost proxy vs a true leave-one-superblock-out bit oracle, INTER coding")
    print(f"n = {len(runs)} probe videos. Bounds pre-registered in docs/F1_ORACLE_BITS.md.")
    print("=" * 84)
    print(f"\n{'video':16}{'rho SC':>9}{'rho blend':>11}{'cap SC':>9}{'cap blend':>11}"
          f"{'random':>9}{'vs rand':>9}{'oracle%':>9}{'neg':>6}")

    rho_sc, rho_bl, cap_sc, cap_bl, orac, nulls = [], [], [], [], [], []
    for v, h, p in runs:
        m = np.asarray(p["marginal_bits"], dtype=float)
        frac = p["oracle_bits_top_k"] / p["reference_bits_total"]
        # The null this proxy has to beat. A capture ratio has a floor well above
        # zero -- picking k superblocks at random already captures ~40% of the
        # oracle's bits -- so a bare "83% of oracle" overstates the proxy's skill.
        # The original 93-99% claim was never quoted against a null either.
        rng = np.random.default_rng(0)
        k = p["top_k"]
        rand = float(np.mean([m[rng.choice(m.size, k, replace=False)].sum()
                              for _ in range(2000)]))
        null = rand / p["oracle_bits_top_k"] if p["oracle_bits_top_k"] > 0 else float("nan")
        rho_sc.append(p["spearman_sc"]); rho_bl.append(p["spearman_blended"])
        cap_sc.append(p["capture_ratio_sc"]); cap_bl.append(p["capture_ratio_blended"])
        orac.append(frac); nulls.append(null)
        print(f"{v:16}{p['spearman_sc']:9.3f}{p['spearman_blended']:11.3f}"
              f"{p['capture_ratio_sc']:9.3f}{p['capture_ratio_blended']:11.3f}"
              f"{null:9.3f}{p['capture_ratio_sc']-null:+9.3f}"
              f"{100*frac:9.1f}{p['n_negative_marginal']:6d}")

    rho_sc = np.array(rho_sc); rho_bl = np.array(rho_bl)
    cap_sc = np.array(cap_sc); cap_bl = np.array(cap_bl)
    orac = np.array(orac); nulls = np.array(nulls)

    print("\n--- against the pre-registered bounds ---")
    print(f"  mean rho (SC)          {rho_sc.mean():+.3f}   {band('rho_mean', rho_sc.mean())}")
    print(f"  per-video rho range    {rho_sc.min():+.3f}..{rho_sc.max():+.3f}   "
          f"{band('rho_each', rho_sc.min())} / {band('rho_each', rho_sc.max())}")
    print(f"  mean capture (SC)      {cap_sc.mean():.3f}   {band('cap_mean', cap_sc.mean())}")
    print(f"  per-video capture      {cap_sc.min():.3f}..{cap_sc.max():.3f}   "
          f"{band('cap_each', cap_sc.min())} / {band('cap_each', cap_sc.max())}")
    print(f"  mean oracle share      {orac.mean():.3f}   {band('oracle_frac', orac.mean())}")

    print("\n--- capture against its null (NOT pre-registered; added after the fact) ---")
    print("  This was missing from the original claim and from these bounds. A capture")
    print("  ratio cannot be read as 'fraction of the way to the oracle': random")
    print("  selection already scores well above zero, so the null is the baseline.")
    print(f"  mean random null       {nulls.mean():.3f}")
    print(f"  mean capture - null    {(cap_sc - nulls).mean():+.3f}   "
          f"(per video {(cap_sc - nulls).min():+.3f}..{(cap_sc - nulls).max():+.3f})")
    near_chance = [runs[i][0] for i in range(len(runs)) if cap_sc[i] - nulls[i] < 0.15]
    if near_chance:
        print(f"  *** near-chance on: {', '.join(near_chance)} -- on these the proxy is "
              "barely better than picking superblocks at random, and that must be stated "
              "wherever the mean is quoted")

    print("\n--- bound 3: does the alpha/beta blend beat SC alone? ---")
    dpp = 100 * (cap_bl - cap_sc)
    print(f"  capture(blend) - capture(SC): mean {dpp.mean():+.2f} pp, "
          f"range {dpp.min():+.2f}..{dpp.max():+.2f} pp")
    print("  [expected 0..+10 pp; ALARM if the blend is worse by more than 5 pp]")
    if dpp.min() < -5:
        print("  *** BREACH: the blend loses to its own spatial half on some videos -- "
              "that is a finding about Eq.(importance), not a footnote")
    elif abs(dpp).max() < 0.5:
        print("  the blend is indistinguishable from SC alone here (see note below)")

    print("\n--- explicit alarm checks ---")
    fired = False
    if rho_sc.max() > ALARMS["rho_tautology"][0]:
        fired = True; print(f"  *** ALARM: {ALARMS['rho_tautology'][1]}")
    if rho_sc.mean() < ALARMS["rho_useless"][0]:
        fired = True; print(f"  *** ALARM: {ALARMS['rho_useless'][1]}")
    if cap_sc.max() > ALARMS["cap_tautology"][0]:
        fired = True; print(f"  *** ALARM: {ALARMS['cap_tautology'][1]}")
    if cap_sc.mean() < ALARMS["cap_broken"][0]:
        fired = True; print(f"  *** ALARM: {ALARMS['cap_broken'][1]}")
    if not fired:
        print("  none fired")

    print("\n--- decision rule (docs/F1_ORACLE_BITS.md), applied ---")
    c, r = cap_sc.mean(), rho_sc.mean()
    if c >= 0.70 and r >= 0.45:
        verdict = ("the numerator claim SURVIVES inter coding. The paper may say the cost "
                   "proxy is close to a bit oracle, quoting the INTER number and naming "
                   "93-99% as the all-intra upper bound it is -- never 93-99% alone.")
    elif c >= 0.50:
        verdict = ("'mostly solved' is sayable; 'solved' is NOT. Report the drop from "
                   "all-intra explicitly.")
    else:
        verdict = ("the numerator is NOT solved. NOTE(sec:implementation)'s reading needs "
                   "revising in the other direction too. Report it; do not retry with a "
                   "different score until it passes.")
    print(f"  mean capture {c:.3f}, mean rho {r:+.3f}  ->  {verdict}")


if __name__ == "__main__":
    main()
