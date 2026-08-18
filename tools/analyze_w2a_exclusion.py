#!/usr/bin/env python
"""W2-A: does hard foreground exclusion buy measurable quality?

W1-C's version was void: on 5 of its 6 clips the score-based top-k contained
almost no foreground even unprotected, so the two arms ran the same selection.
These six clips were chosen on a PRE-OUTCOME criterion -- the share of
foreground a purely score-based top-k degrades -- so the mechanism is exercised.
Bounds in docs/PREREG_WAVE_OVERNIGHT.md (W2-A).
"""
import json, glob, collections, math, sys
import numpy as np
sys.path.insert(0, 'scripts')
from bd_rate import bd_rate, overlap_fraction

CLIPS = ['drift-chicane', 'lindy-hop', 'dogs-jump', 'tennis', 'dogs-scale', 'bmx-trees']
RUNGS = [43, 50, 55, 60]

def main():
    base = collections.defaultdict(dict); arm = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in glob.glob('results/*/result.json'):
        try: d = json.load(open(p))
        except Exception: continue
        if d.get('invariant_failures'): continue
        c = d['config']; m = d.get('metrics') or {}
        if c.get('codec') != 'svtav1' or c.get('width') != 640: continue
        # NB: baselines carry block_size=None, so the block-size filter belongs
        # on the arm branch only -- applying it here drops every baseline and
        # makes every ladder look incomplete.
        if c.get('video') not in CLIPS: continue
        qp = (c.get('codec_params') or {}).get('qp')
        if qp not in RUNGS: continue
        fg = (m.get('foreground') or {}).get('lpips_mean'); bg = (m.get('background') or {}).get('lpips_mean')
        if fg is None: continue
        rec = dict(rate=d['actual_bitrate_bps'], fg=fg, bg=bg)
        if c['component'] == 'baselines':
            base[c['video']][qp] = rec
        elif (c.get('component') == 'presley_ai' and c.get('degradation') == 'downsample'
              and c.get('restorer') == 'realesrgan' and c.get('shrink_amount') == 0.25
              and c.get('block_size') == 8
              and c.get('selection', 'score') == 'score'):
            arm[c['video']]['protect' if c.get('fg_protect') else 'noprotect'][qp] = rec
    print(f"{'clip':16}{'FG gap (C-A)':>14}{'A fg':>9}{'C fg':>9}{'BD-rate C vs A':>17}{'ovl':>6}")
    fgd, bdr = [], []
    for v in CLIPS:
        a, c_ = arm[v].get('protect', {}), arm[v].get('noprotect', {})
        qs = [q for q in RUNGS if q in a and q in c_ and q in base[v]]
        if len(qs) < 4:
            print(f"{v:16}{'incomplete':>14}"); continue
        afg = np.mean([a[q]['fg'] for q in qs]); cfg = np.mean([c_[q]['fg'] for q in qs])
        ra = [a[q]['rate'] for q in qs]; ya = [a[q]['bg'] for q in qs]
        rc = [c_[q]['rate'] for q in qs]; yc = [c_[q]['bg'] for q in qs]
        bd = bd_rate(ra, ya, rc, yc, lower_is_better=True); ov = overlap_fraction(ra, rc)
        fgd.append(cfg - afg); bdr.append(bd)
        print(f"{v:16}{cfg-afg:>+14.4f}{afg:>9.4f}{cfg:>9.4f}{bd:>16.1f}%{ov:>6.2f}")
    if not fgd: return
    n = len(fgd); k = sum(1 for x in fgd if x > 0.03)
    print(f"\nFG gap (dropping protection makes the foreground WORSE when positive):")
    print(f"  median {np.median(fgd):+.4f}   exceeding the pre-registered 0.03 on {k}/{n}")
    print(f"  exceeding the 0.05 perceptual margin on {sum(1 for x in fgd if x>0.05)}/{n}")
    w = sum(1 for x in fgd if x > 0)
    p = min(1.0, 2*sum(math.comb(n, i) for i in range(max(w, n-w), n+1))/2**n)
    print(f"  worse without protection on {w}/{n}, exact two-tailed p = {p:.4f}")
    print(f"\nBackground BD-rate, no-protect vs protect: median {np.median(bdr):+.1f}% "
          f"(negative = dropping protection is cheaper, which is expected)")

if __name__ == '__main__':
    main()
