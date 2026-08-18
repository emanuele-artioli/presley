#!/usr/bin/env python
"""W1-C: does score-based placement buy anything at fixed budget and strength?

Evaluates the pre-registered bounds in docs/PREREG_WAVE_OVERNIGHT.md before any
headline is read. Arms differ only in `selection` (score|random) and
`fg_protect`, so budget and per-block strength are matched by construction.
"""
import json, os, sys, glob, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import yaml
from bd_rate import bd_rate, overlap_fraction
from presley.runner import compute_experiment_hash

ARMS = {(False, True): 'A score+prot', (True, True): 'B rand+prot',
        (False, False): 'C score-prot', (True, False): 'D rand-prot'}

def load():
    exps = yaml.safe_load(open('experiments_w1c_placement.yaml'))['experiments']
    base, arms = collections.defaultdict(dict), collections.defaultdict(lambda: collections.defaultdict(dict))
    for e in exps:
        h = compute_experiment_hash(e); p = f'results/{h}/result.json'
        if not os.path.exists(p): continue
        d = json.load(open(p))
        if d.get('invariant_failures'): continue
        v, qp = e['video'], e['codec_params']['qp']
        m = d['metrics']
        rec = dict(rate=d['actual_bitrate_bps'],
                   bg=m['background']['lpips_mean'], fg=m['foreground']['lpips_mean'],
                   sel=(m.get('block_level') or {}).get('selected_fraction'))
        if e['component'] == 'baselines':
            base[v][qp] = rec
        else:
            key = ARMS[(e.get('selection') == 'random', e.get('fg_protect', False))]
            arms[v][key][qp] = rec
    return base, arms

def curve(d, field):
    qps = sorted(d)
    return [d[q]['rate'] for q in qps], [d[q][field] for q in qps]

def main():
    base, arms = load()
    print("=== W1-C placement control ===\n")
    names = list(ARMS.values())
    bd = collections.defaultdict(dict)
    for v in sorted(arms):
        for a in names:
            if len(arms[v].get(a, {})) < 4:
                continue
            r, q = curve(arms[v][a], 'bg')
            rb, qb = curve(base[v], 'bg')
            bd[v][a] = dict(bg=bd_rate(rb, qb, r, q, lower_is_better=True),
                            ov=overlap_fraction(rb, r),
                            fg=sum(x['fg'] for x in arms[v][a].values())/4)
    hdr = f"{'video':14}" + "".join(f"{a:>16}" for a in names)
    print(hdr); print('-'*len(hdr))
    for v in sorted(bd):
        print(f"{v:14}" + "".join(f"{bd[v][a]['bg']:>15.1f}%" if a in bd[v] else f"{'--':>16}" for a in names))
    print("\n--- contrasts (BD-rate vs baseline on BG-LPIPS; negative = cheaper) ---")
    for label, x, y in [("ranking  A vs B", 'A score+prot', 'B rand+prot'),
                        ("exclusion A vs C", 'A score+prot', 'C score-prot')]:
        vs = [v for v in bd if x in bd[v] and y in bd[v]]
        d = [bd[v][x]['bg'] - bd[v][y]['bg'] for v in vs]
        better = sum(1 for z in d if z < 0)
        print(f"{label}: n={len(vs)}  A better on {better}/{len(vs)}  median delta {sorted(d)[len(d)//2]:+.1f} pp")
    print("\n--- FG-LPIPS mean per arm (exclusion should show here) ---")
    for a in names:
        vals = [bd[v][a]['fg'] for v in bd if a in bd[v]]
        if vals: print(f"  {a:16} mean FG-LPIPS {sum(vals)/len(vals):.4f}  (n={len(vals)})")
    print("\n--- overlap gate (>=0.50 required) ---")
    bad = [(v, a, bd[v][a]['ov']) for v in bd for a in bd[v] if bd[v][a]['ov'] < 0.50]
    print("  all pass" if not bad else f"  BREACH: {bad}")

if __name__ == '__main__':
    main()
