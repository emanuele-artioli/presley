#!/usr/bin/env python
"""Is the W2-B extension ready to fold into the paper? Exit 0 = yes.

Answers the question with a predicate rather than a judgement. Every condition
here is one that, if skipped, has already produced a wrong statement in this
project at least once.
"""
import json, os, sys, yaml, glob
sys.path.insert(0, 'src')
from presley.runner import compute_experiment_hash

def main():
    exps = yaml.safe_load(open('experiments_w2b_ext.yaml'))['experiments']
    n = len(exps)
    missing, unevaluated, no_lpips, clip_breach, other_inv = [], [], [], [], []
    for e in exps:
        h = compute_experiment_hash(e); p = f'results/{h}/result.json'
        if not os.path.exists(p):
            missing.append(h); continue
        d = json.load(open(p)); m = d.get('metrics') or {}
        inv = d.get('invariant_failures') or []
        if not m:
            unevaluated.append(h); continue
        if (m.get('foreground') or {}).get('lpips_mean') is None or \
           (m.get('background') or {}).get('lpips_mean') is None:
            no_lpips.append(h)
        for x in inv:
            (clip_breach if 'clipped' in x else other_inv).append(h)

    checks = [
        ("all runs present",            not missing,      f"{len(missing)}/{n} have no result.json"),
        ("all runs evaluated",          not unevaluated,  f"{len(unevaluated)} have an empty metrics block"),
        ("LPIPS present on all",        not no_lpips,     f"{len(no_lpips)} lack FG or BG LPIPS"),
        ("no non-clipping invariants",  not other_inv,    f"{len(set(other_inv))} carry another invariant failure"),
    ]
    ok = True
    print(f"W2-B extension: {n} planned entries\n")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed else f"  -- {detail}"))
        ok &= passed
    if clip_breach:
        print(f"\n  NOTE: {len(set(clip_breach))} run(s) breach the CLIPPING invariant. "
              "That is evidence about the graded transport, not a blocker -- it is "
              "counted and reported, per the pre-registration.")
    print(f"\n{'READY to fold' if ok else 'NOT READY'}")
    if not ok:
        print("Do not fold partial results: a sign test on a subset of a "
              "pre-declared n is exactly the optional stopping the design avoids.")
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
