#!/usr/bin/env python
"""F14: does anything predict which clips bit relocation works on?

Sixth attempt. Family and bounds pre-registered in docs/PREREG_WAVE_OVERNIGHT.md
before this ran. Every candidate is computable from the SOURCE clip alone --
nothing downstream of the degradation or the encode -- so no candidate can be
circular with the outcome.
"""
import json, glob, os, sys, collections
import numpy as np, cv2
sys.path.insert(0, 'src')
from scipy.stats import spearmanr
from presley.preprocessing import get_evca_scores, get_reference_frames, resolve_masks
from presley.suite import holm_adjust

RUNGS = [32, 37, 42, 47]

def outcome():
    base, arm = collections.defaultdict(dict), collections.defaultdict(dict)
    for p in glob.glob('results/*/result.json'):
        try: d = json.load(open(p))
        except Exception: continue
        if d.get('invariant_failures'): continue
        c = d['config']
        if c.get('codec') != 'x265' or c.get('width') != 640 or c.get('height') != 360: continue
        qp = (c.get('codec_params') or {}).get('qp')
        if qp not in RUNGS: continue
        if c['component'] == 'baselines':
            base[c['video']][qp] = d['actual_bitrate_bps']
        elif (c['component'] == 'presley_ai' and c.get('degradation') == 'downsample'
              and c.get('restorer') == 'realesrgan' and c.get('block_size') == 8
              and c.get('fg_protect') and c.get('shrink_amount') == 0.25
              and c.get('selection', 'score') == 'score'):
            arm[c['video']][qp] = d['actual_bitrate_bps']
    out = {}
    for v in arm:
        qs = [q for q in RUNGS if q in arm[v] and q in base[v]]
        if len(qs) < 4: continue
        out[v] = float(np.mean([100*(arm[v][q]/base[v][q]-1) for q in qs]))
    return out

def features(v):
    raw, _, _ = get_reference_frames(v, 640, 360, 'dataset', 'cache')
    ref = os.path.join('cache', f'{v}_640x360', 'reference_frames')
    T, S = get_evca_scores(v, 640, 360, 8, raw, ref, 'cache')
    masks = resolve_masks('ufo', v, 640, 360, 8, ref, 'cache', 'dataset')
    nbx, nby = 80, 45
    bg_share, ratio = [], []
    for i in range(min(len(S), len(masks))):
        m = cv2.resize(masks[i], (nbx, nby), interpolation=cv2.INTER_NEAREST) > 0
        s = S[i]
        if s.shape != m.shape: m = cv2.resize(m.astype(np.uint8), (s.shape[1], s.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        tot = s.sum()
        if tot <= 0: continue
        bg_share.append(s[~m].sum()/tot)
        fg_mean = s[m].mean() if m.any() else 0.0
        bg_mean = s[~m].mean() if (~m).any() else 1e-9
        ratio.append(bg_mean/max(fg_mean, 1e-9))
    return dict(bg_bit_share=float(np.mean(bg_share)),
                bg_fg_ratio=float(np.mean(ratio)),
                temporal=float(np.mean(T)),
                spatial_cv=float(np.std(S)/max(np.mean(S), 1e-9)))

def main():
    out = outcome()
    print(f"clips with a complete PRESLEY ladder: {len(out)}")
    rows = {}
    for v in sorted(out):
        try: f = features(v)
        except Exception as e:
            print(f"  skip {v}: {type(e).__name__}"); continue
        rows[v] = f
    # candidate 3 needs the baseline bpp, from the outcome pass
    base = collections.defaultdict(dict)
    for p in glob.glob('results/*/result.json'):
        try: d = json.load(open(p))
        except Exception: continue
        c = d['config']
        if c.get('component') == 'baselines' and c.get('codec') == 'x265' and c.get('width') == 640:
            qp = (c.get('codec_params') or {}).get('qp')
            if qp in RUNGS: base[c['video']][qp] = d['actual_bitrate_bps']
    for v in rows:
        if RUNGS[0] in base[v]:
            rows[v]['bpp'] = base[v][RUNGS[0]] / (640*360*float(json.load(open(glob.glob('results/*/result.json')[0])).get('video_framerate') or 24))
    vs = [v for v in rows if 'bpp' in rows[v]]
    names = ['bg_bit_share', 'bg_fg_ratio', 'bpp', 'temporal', 'spatial_cv']
    y = [out[v] for v in vs]
    print(f"n = {len(vs)} clips\noutcome: mean matched-QP bitrate delta, PRESLEY vs baseline")
    print(f"  saves bits on {sum(1 for z in y if z<0)}/{len(y)}   median {np.median(y):+.1f}%\n")
    res = []
    for nm in names:
        x = [rows[v][nm] for v in vs]
        r = spearmanr(x, y)
        res.append((nm, r.statistic, r.pvalue))
    adj = holm_adjust([r[2] for r in res])
    print(f"{'candidate':18}{'rho':>8}{'p raw':>10}{'p Holm':>10}  verdict")
    for (nm, rho, p), pa in zip(res, adj):
        v = 'SIGNIFICANT' if pa < 0.05 else ('loser' if p >= 0.05 else 'loser (dies under Holm)')
        print(f"{nm:18}{rho:>8.3f}{p:>10.3f}{pa:>10.3f}  {v}")
    print(f"\nsignificant after Holm: {sum(1 for pa in adj if pa<0.05)}/5")

if __name__ == '__main__':
    main()
