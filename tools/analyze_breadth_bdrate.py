#!/usr/bin/env python
"""W1-A: does 'the bridge saves bits on most clips' survive at matched quality?

fig:breadth currently plots a matched-QP bitrate delta. A saving is only a
saving at equal quality, which needs a BD-rate over a real ladder. Both arms are
now on QP {32,37,42,47}. Bounds in docs/PREREG_WAVE_OVERNIGHT.md (W1-A).
"""
import json, os, sys, glob, collections, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import yaml
from bd_rate import bd_rate, overlap_fraction
from presley.runner import compute_experiment_hash

RUNGS = [32, 37, 42, 47]

def load():
    """Scan results/ directly: the wave YAML holds only the cells that were
    missing, so loading from it would miss every pre-existing rung."""
    base = collections.defaultdict(dict); arms = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in glob.glob('results/*/result.json'):
        try: d = json.load(open(p))
        except Exception: continue
        if d.get('invariant_failures'): continue
        c = d['config']
        if c.get('codec') != 'x265' or c.get('width') != 640 or c.get('height') != 360: continue
        qp = (c.get('codec_params') or {}).get('qp')
        if qp not in RUNGS: continue
        m = d.get('metrics') or {}
        if not (m.get('foreground') or {}).get('lpips_mean'): continue
        v = c['video']
        rec = dict(rate=d['actual_bitrate_bps'], fg=m['foreground']['lpips_mean'],
                   bg=(m.get('background') or {}).get('lpips_mean'))
        comp = c.get('component')
        if comp == 'baselines':
            base[v][qp] = rec
        elif comp == 'elvis' and c.get('inpainter') == 'none' and c.get('removal_mode') == 'blackout' \
                and c.get('block_size') == 8:
            arms[v]['ELVIS'][qp] = rec
        elif comp == 'presley_ai' and c.get('degradation') == 'downsample' \
                and c.get('restorer') == 'realesrgan' and c.get('block_size') == 8 \
                and c.get('fg_protect') and c.get('shrink_amount') == 0.25 \
                and c.get('selection', 'score') == 'score':
            arms[v]['PRESLEY'][qp] = rec
    return base, arms

def main():
    base, arms = load()
    out = collections.defaultdict(dict)
    for v in sorted(arms):
        for a in ('ELVIS', 'PRESLEY'):
            d = arms[v].get(a, {})
            common = [q for q in RUNGS if q in d and q in base[v]]
            if len(common) < 4: continue
            r = [d[q]['rate'] for q in common];  y = [d[q]['fg'] for q in common]
            rb = [base[v][q]['rate'] for q in common]; yb = [base[v][q]['fg'] for q in common]
            ov = overlap_fraction(rb, r)
            out[v][a] = dict(bd=bd_rate(rb, yb, r, y, lower_is_better=True), ov=ov,
                             qpdelta=100*(d[RUNGS[0]]['rate']/base[v][RUNGS[0]]['rate']-1) if RUNGS[0] in d and RUNGS[0] in base[v] else None)
    for a in ('ELVIS', 'PRESLEY'):
        rows = [(v, out[v][a]) for v in out if a in out[v]]
        gated = [(v, o) for v, o in rows if o['ov'] >= 0.50]
        print(f"\n=== {a} arm: n={len(rows)} ladders, {len(gated)} pass the overlap gate ===")
        print(f"{'video':22}{'BD-rate FG-LPIPS':>18}{'overlap':>9}{'matched-QP rate d%':>20}")
        for v, o in sorted(gated, key=lambda x: x[1]['bd']):
            q = f"{o['qpdelta']:+.1f}%" if o['qpdelta'] is not None else "--"
            print(f"{v:22}{o['bd']:>17.1f}%{o['ov']:>9.2f}{q:>20}")
        if not gated: continue
        save = sum(1 for _, o in gated if o['bd'] < 0)
        n = len(gated)
        p = min(1.0, 2*sum(math.comb(n, k) for k in range(max(save, n-save), n+1))/2**n)
        med = sorted(o['bd'] for _, o in gated)[n//2]
        print(f"  saves bits at MATCHED QUALITY on {save}/{n} = {save/n:.2f}   median BD-rate {med:+.1f}%   sign p={p:.4f}")
        qd = [o['qpdelta'] for _, o in gated if o['qpdelta'] is not None]
        if qd:
            qs = sum(1 for x in qd if x < 0)
            print(f"  for contrast, at MATCHED QP the same ladders 'save' on {qs}/{len(qd)} = {qs/len(qd):.2f}")

if __name__ == '__main__':
    main()
