#!/usr/bin/env python
"""W2-B: graded vs binary at n>=9, block_size 64 / 1080p.

W1-B gave 6/7 (p=0.125). Three more clips were pre-registered with a stopping
rule fixed in advance. Clipping-invariant breaches are evidence about the
graded transport, not dropped datapoints -- they are counted and reported.
"""
import json, glob, collections, math, sys
import numpy as np
sys.path.insert(0, 'scripts')
from bd_rate import bd_rate, overlap_fraction

def main():
    arm = collections.defaultdict(lambda: collections.defaultdict(dict))
    bad = collections.defaultdict(lambda: collections.defaultdict(int))
    pending = collections.defaultdict(lambda: collections.defaultdict(int))
    for p in glob.glob('results/*/result.json'):
        try: d = json.load(open(p))
        except Exception: continue
        c = d['config']; m = d.get('metrics') or {}
        if c.get('component') != 'presley_ai' or c.get('width') != 1920: continue
        if c.get('block_size') != 64 or c.get('degradation') != 'downsample': continue
        a = 'graded' if c.get('downsample_levels') else 'binary'
        qp = (c.get('codec_params') or {}).get('qp')
        inv = d.get('invariant_failures') or []
        if inv:
            # Distinguish a REAL breach from a run that simply has not been
            # evaluated yet: a pre-metrics result carries "metrics block is
            # missing or empty", which is transient and says nothing about the
            # transport. Counting those as breaches invents an alarm.
            if any('clipped' in x for x in inv):
                bad[c['video']][a] += 1
            else:
                pending[c['video']][a] += 1
            continue
        bg = (m.get('background') or {}).get('lpips_mean'); fg = (m.get('foreground') or {}).get('lpips_mean')
        if bg is None: continue
        arm[c['video']][a][qp] = dict(rate=d['actual_bitrate_bps'], bg=bg, fg=fg)
    print(f"{'clip':16}{'BD-rate graded vs binary':>26}{'ovl':>7}{'FG delta':>10}{'status':>12}")
    res = []
    for v in sorted(arm):
        b, g = arm[v]['binary'], arm[v]['graded']
        qs = sorted(set(b) & set(g))
        if len(qs) < 4:
            print(f"{v:16}{'incomplete':>26}{'':>7}{'':>10}"
                  f"{('graded breached x%d' % bad[v]['graded']) if bad[v]['graded'] else 'missing':>12}")
            continue
        bd = bd_rate([b[q]['rate'] for q in qs], [b[q]['bg'] for q in qs],
                     [g[q]['rate'] for q in qs], [g[q]['bg'] for q in qs], lower_is_better=True)
        ov = overlap_fraction([b[q]['rate'] for q in qs], [g[q]['rate'] for q in qs])
        fgd = abs(np.mean([g[q]['fg'] for q in qs]) - np.mean([b[q]['fg'] for q in qs]))
        res.append((v, bd, ov, fgd)); print(f"{v:16}{bd:>25.1f}%{ov:>7.2f}{fgd:>10.4f}{'ok':>12}")
    if not res: return
    n = len(res); w = sum(1 for r in res if r[1] > 0)
    p = min(1.0, 2*sum(math.comb(n, i) for i in range(max(w, n-w), n+1))/2**n)
    print(f"\nn={n} clean clips: graded WORSE on {w}/{n}, exact two-tailed p = {p:.4f}")
    print(f"median BD-rate {np.median([r[1] for r in res]):+.1f}%   "
          f"max |FG delta| {max(r[3] for r in res):.4f} (validity bound 0.02)   "
          f"min overlap {min(r[2] for r in res):.2f}")
    tot = sum(bad[v]['graded'] for v in bad); totb = sum(bad[v]['binary'] for v in bad)
    print(f"\nCLIPPING breaches: graded {tot}, binary {totb}"
          f"  -- clips: {sorted(v for v in bad if bad[v]['graded'])}")
    pg = sum(pending[v]['graded'] for v in pending); pb = sum(pending[v]['binary'] for v in pending)
    if pg or pb:
        print(f"not yet evaluated (no metrics, NOT a breach): graded {pg}, binary {pb}"
              f"  -- clips: {sorted(set(pending))}")

if __name__ == '__main__':
    main()
