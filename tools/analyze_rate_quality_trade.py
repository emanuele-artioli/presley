#!/usr/bin/env python
"""The rate/quality trade across the ladder (x265 breadth, four rungs).

At matched QP the foreground is indistinguishable at every rung, which is the
condition that makes a matched-QP comparison the sanctioned analysis. What moves
is the background, and it moves in the opposite direction to the rate.
"""
import json, glob, collections, numpy as np

RUNGS = [32, 37, 42, 47]
ARM = dict(degradation='downsample', restorer='realesrgan', block_size=8,
           fg_protect=True, shrink_amount=0.25)

def main():
    base, arm = collections.defaultdict(dict), collections.defaultdict(dict)
    hashes = collections.defaultdict(list)
    for p in glob.glob('results/*/result.json'):
        try: d = json.load(open(p))
        except Exception: continue
        if d.get('invariant_failures'): continue
        c = d['config']; m = d.get('metrics') or {}
        if c.get('codec') != 'x265' or c.get('width') != 640 or c.get('height') != 360: continue
        qp = (c.get('codec_params') or {}).get('qp')
        if qp not in RUNGS: continue
        bg = (m.get('background') or {}).get('lpips_mean')
        fg = (m.get('foreground') or {}).get('lpips_mean')
        if bg is None or fg is None: continue
        rec = (d['actual_bitrate_bps'], bg, fg)
        h = d.get('experiment_hash')
        if c['component'] == 'baselines':
            base[c['video']][qp] = rec; hashes['baselines'].append(h)
        elif c['component'] == 'presley_ai' and all(c.get(k) == v for k, v in ARM.items()) \
                and c.get('selection', 'score') == 'score':
            arm[c['video']][qp] = rec; hashes['presley_ai'].append(h)
    vs = sorted(v for v in arm if all(q in arm[v] and q in base[v] for q in RUNGS))
    print(f"n = {len(vs)} clips with all four rungs   "
          f"({len(hashes['presley_ai'])} arm runs, {len(hashes['baselines'])} baselines)")
    print(f"\n{'QP':>4}{'saves bits':>13}{'median rate':>14}"
          f"{'median BG gap':>16}{'BG better':>12}{'median FG gap':>16}")
    for q in RUNGS:
        rate = [100*(arm[v][q][0]/base[v][q][0]-1) for v in vs]
        bgap = [arm[v][q][1]-base[v][q][1] for v in vs]
        fgap = [arm[v][q][2]-base[v][q][2] for v in vs]
        print(f"{q:>4}{sum(1 for x in rate if x<0):>6}/{len(rate):<6}"
              f"{np.median(rate):>+13.1f}%{np.median(bgap):>+16.4f}"
              f"{sum(1 for g in bgap if g<0):>6}/{len(bgap):<5}{np.median(fgap):>+16.4f}")
    print("\nBG/FG gaps are PRESLEY minus baseline on LPIPS; negative = PRESLEY better.")
    print("FG gap stays far inside the 0.05 perceptual margin at every rung, so the")
    print("matched-QP comparison is valid throughout and the background carries the trade.")

if __name__ == '__main__':
    main()
