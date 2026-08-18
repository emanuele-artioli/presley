#!/usr/bin/env python
"""F3: is there a seam? Quality at degraded/undegraded borders vs interior.

The introduction argues ROI coding stalled because coding neighbouring regions
at different quality leaves a visible boundary, and that a generative layer
removes the need for one. That argument is currently asserted. This measures it
on our own output: among DEGRADED blocks, compare those touching an undegraded
neighbour against those whose neighbours are all degraded.

Both groups received the identical operator at the identical strength, so any
difference is positional -- it is the seam, not the degradation.
"""
import glob, json, sys
import numpy as np
sys.path.insert(0, 'src')
from presley.sidechannel import load_level_masks
from scipy.stats import wilcoxon

def load(h):
    d = json.load(open(f'results/{h}/result.json'))
    psnr = np.load(f'results/{h}/block_psnr.npz')['arr_0']
    sm = load_level_masks(f"results/{h}/strength_maps.npz")
    return d['config'], psnr, sm

def baseline_index():
    """clip+qp -> baseline hash, so damage can be measured against the same content."""
    idx = {}
    for q in glob.glob('results/*/result.json'):
        try: d = json.load(open(q))
        except Exception: continue
        c = d['config']
        if c.get('component') != 'baselines': continue
        idx[(c['video'], c.get('codec'), c.get('width'), (c.get('codec_params') or {}).get('qp'))] = q.split('/')[1]
    return idx


def main():
    """Seam vs interior must be compared on DAMAGE, not on absolute quality.

    Interior blocks sit in the middle of large degraded regions, which the
    selector chose because they are the most removable -- i.e. the most complex
    content, which has lower PSNR before anything is done to it. Comparing raw
    PSNR therefore compares content difficulty, not position. Subtracting each
    block's own baseline PSNR removes that.
    """
    base = baseline_index()
    rows = []
    for p in sorted(glob.glob('results/*/strength_maps.npz')):
        h = p.split('/')[1]
        try:
            c, psnr, sm = load(h)
        except Exception:
            continue
        if c.get('component') != 'presley_ai' or c.get('degradation') != 'downsample':
            continue
        if c.get('restorer') != 'realesrgan' or not c.get('fg_protect'):
            continue
        if sm.shape != psnr.shape:
            continue
        deg = sm > 0
        if deg.sum() == 0:
            continue
        # a degraded block is on the seam if any 4-neighbour is undegraded
        pad = np.pad(deg, ((0, 0), (1, 1), (1, 1)), constant_values=True)
        allnb = (pad[:, :-2, 1:-1] & pad[:, 2:, 1:-1] &
                 pad[:, 1:-1, :-2] & pad[:, 1:-1, 2:])
        seam = deg & ~allnb
        interior = deg & allnb
        if seam.sum() < 50 or interior.sum() < 50:
            continue
        bh = base.get((c['video'], c.get('codec'), c.get('width'),
                       (c.get('codec_params') or {}).get('qp')))
        if not bh: continue
        try: bp = np.load(f'results/{bh}/block_psnr.npz')['arr_0']
        except Exception: continue
        if bp.shape != psnr.shape: continue
        dmg = bp - psnr                      # positive = degraded arm is worse
        f = np.isfinite(dmg)
        s = float(np.median(dmg[seam & f])); i = float(np.median(dmg[interior & f]))
        rows.append((c['video'], (c.get('codec_params') or {}).get('qp'), s, i, s - i,
                     int(seam.sum()), int(interior.sum())))
    if not rows:
        print("no runs with both seam and interior degraded blocks"); return
    d = np.array([r[4] for r in rows])
    print(f"n = {len(rows)} runs (presley_ai downsample+realesrgan, fg_protect)\n")
    print(f"{'video':22}{'qp':>5}{'seam dmg':>11}{'interior':>10}{'delta':>9}")
    for r in sorted(rows, key=lambda x: x[4])[:12]:
        print(f"{r[0]:22}{str(r[1]):>5}{r[2]:>11.2f}{r[3]:>10.2f}{r[4]:>+9.2f}")
    if len(rows) > 12: print(f"  ... {len(rows)-12} more")
    print(f"\nmedian seam-minus-interior DAMAGE: {np.median(d):+.3f} dB  (positive = seam damaged MORE)")
    print(f"seam damaged MORE on {int((d>0).sum())}/{len(d)} runs")
    try:
        w = wilcoxon(d)
        print(f"Wilcoxon signed-rank p = {w.pvalue:.4g}")
    except Exception as e:
        print("Wilcoxon failed:", e)
    print(f"\n0.5 dB is the margin this project treats as perceptible.")
    print(f"runs whose |delta| exceeds it: {int((np.abs(d)>0.5).sum())}/{len(d)}")

if __name__ == '__main__':
    main()
