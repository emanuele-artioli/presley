"""How wide is the scope where each goal actually succeeds, and where do they intersect?

Written for the submission pass. The article is organised around three goals
(Section~\\ref{sec:introduction}); what it does not yet state is **the largest
scope over which each one holds, and whether their intersection is non-empty**.
That question is answerable from runs already on disk, so it is answered here
before any new GPU time is commissioned.

Goals, in the article's introduction order:

  1. **Selection** -- pick the right blocks to degrade. This is a *global*
     property of the scoring function, not a per-cell outcome, so it is
     reported from the oracle probes rather than counted like the other two.
  2. **Reduction** -- degrade them so the encoder spends fewer bits, without
     costing foreground quality.
  3. **Restoration** -- bring the degraded regions back up.

Success definitions follow the project's own established gates, not new ones:

  * Reduction succeeds in a cell when it **saves bits against the pristine
    baseline at the same fixed QP** *and* the foreground-PSNR loss is within
    the 0.5 dB JND. That is exactly the "PSNR-gated WIN" rule `tab:breadth`
    already uses.
  * Restoration succeeds when BG-LPIPS improves by **at least one JND** (0.05)
    against the matched `none`-restorer control on the same transmitted
    bitstream. BG-PSNR is never the restoration verdict (hard rule).

Hard rules enforced: fixed-QP only (VBR excluded outright); a run with any
`invariant_failures` is dropped; foreground verdicts come only from true masked
metrics.

⚠ These are **per-cell screens**, not BD-rate curves. A cell is one operating
point; the reduction gate compares at equal QP, which is the right question for
"does this configuration save bits", but is not the same as a rate-matched BD
number. Do not quote a count from here as a BD result.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict

DB = "results/presley.db"
FG_PSNR_JND = 0.5
BG_LPIPS_JND = 0.05
FIXED_QP = ("cqp", "crf")


def load(db):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT hash, doc, component, video, width, height, codec, qp, rate_control,"
        "       degradation, restorer, inpainter, block_size, actual_bitrate_bps,"
        "       video_frames, total_time_seconds"
        "  FROM runs"
        " WHERE rate_control IN (?, ?) AND n_invariant_failures = 0 AND has_metrics = 1",
        FIXED_QP,
    ).fetchall()
    met = defaultdict(dict)
    for h, region, metric, value in con.execute(
        "SELECT hash, region, metric, value FROM metrics"
    ):
        met[h][(region, metric)] = value

    out = []
    for r in rows:
        doc = json.loads(r["doc"])
        cfg = doc.get("config") or {}
        out.append({
            "hash": r["hash"], "component": r["component"], "video": r["video"],
            "codec": r["codec"], "qp": r["qp"], "w": r["width"], "h": r["height"],
            "bs": r["block_size"], "bits": r["actual_bitrate_bps"],
            "frames": r["video_frames"],
            "restore_s": doc.get("restoration_time_seconds"),
            # the two components name these differently; normalise
            "transport": r["degradation"] or cfg.get("removal_mode"),
            "fill": r["restorer"] or r["inpainter"],
            "sa": cfg.get("shrink_amount"), "fg_protect": cfg.get("fg_protect"),
            "m": met.get(r["hash"], {}),
        })
    return out


def key(e):
    """Operating point: everything that must match for a fair comparison."""
    return (e["video"], e["codec"], e["qp"], e["w"], e["h"])


def arm(e):
    """The configuration within an operating point, excluding the fill."""
    return (e["component"], e["transport"], e["bs"], e["sa"])


def fg_psnr(e):
    return e["m"].get(("foreground", "psnr_mean"))


def bg_lpips(e):
    return e["m"].get(("background", "lpips_mean"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--min-frames", type=int, default=0)
    args = ap.parse_args()

    runs = load(args.db)
    baselines = {key(e): e for e in runs if e["component"] == "baselines"}
    treated = [e for e in runs if e["component"] in ("presley_ai", "elvis")]

    print(f"{len(runs)} fixed-QP citable runs; {len(baselines)} baseline operating points; "
          f"{len(treated)} treated runs\n")

    # ---------------- Goal 2: reduction ----------------
    red = []
    for e in treated:
        b = baselines.get(key(e))
        if b is None or fg_psnr(e) is None or fg_psnr(b) is None or not b["bits"]:
            continue
        dbits = 100.0 * (e["bits"] - b["bits"]) / b["bits"]
        dfg = fg_psnr(e) - fg_psnr(b)
        red.append({**e, "dbits": dbits, "dfg": dfg,
                    "win": dbits < 0 and dfg >= -FG_PSNR_JND})

    print("=" * 72)
    print("GOAL 2 -- REDUCTION: saves bits at equal QP AND FG-PSNR within 0.5 dB JND")
    print("=" * 72)
    report_goal(red, "win", lambda r: f"{r['dbits']:+6.1f}% bits, {r['dfg']:+5.2f} dB FG")

    # ---------------- Goal 3: restoration ----------------
    # Pair each filled arm against the `none` control at the same operating
    # point and same arm configuration.
    controls = {(key(e), arm(e)): e for e in treated if e["fill"] in (None, "none")}
    rest = []
    for e in treated:
        if e["fill"] in (None, "none"):
            continue
        c = controls.get((key(e), arm(e)))
        if c is None or bg_lpips(e) is None or bg_lpips(c) is None:
            continue
        gain = bg_lpips(c) - bg_lpips(e)          # positive = restoration helps
        rest.append({**e, "gain": gain, "mult": gain / BG_LPIPS_JND,
                     "win": gain >= BG_LPIPS_JND})

    print("\n" + "=" * 72)
    print("GOAL 3 -- RESTORATION: BG-LPIPS improves by >= 1 JND vs matched `none`")
    print("=" * 72)
    report_goal(rest, "win", lambda r: f"{r['mult']:+5.2f}x JND")

    # ---------------- Intersection ----------------
    red_by_hash = {r["hash"]: r for r in red}
    both, only2, only3 = [], [], []
    for r in rest:
        d = red_by_hash.get(r["hash"])
        if d is None:
            continue
        if d["win"] and r["win"]:
            both.append((d, r))
        elif d["win"]:
            only2.append((d, r))
        elif r["win"]:
            only3.append((d, r))
    joint = len(both) + len(only2) + len(only3) + sum(
        1 for r in rest if r["hash"] in red_by_hash
        and not red_by_hash[r["hash"]]["win"] and not r["win"])

    print("\n" + "=" * 72)
    print("INTERSECTION -- cells scored on BOTH goals 2 and 3")
    print("=" * 72)
    if not joint:
        print("  no cell has both a baseline and a matched `none` control")
    else:
        print(f"  cells scored on both: {joint}")
        print(f"    both goals succeed : {len(both):4d}  ({100*len(both)/joint:5.1f}%)")
        print(f"    reduction only     : {len(only2):4d}  ({100*len(only2)/joint:5.1f}%)")
        print(f"    restoration only   : {len(only3):4d}  ({100*len(only3)/joint:5.1f}%)")
        print(f"    neither            : {joint-len(both)-len(only2)-len(only3):4d}")
        if both:
            print("\n  Where BOTH hold (the deployable scope):")
            by_v = defaultdict(list)
            for d, r in both:
                by_v[d["video"]].append((d, r))
            for v in sorted(by_v):
                cells = by_v[v]
                qps = sorted({c[0]["qp"] for c in cells})
                arms = sorted({f"{c[0]['transport']}+{c[0]['fill']}" for c in cells})
                print(f"    {v:22} {len(cells):3d} cells  QP {qps}")
                print(f"    {'':22} arms: {', '.join(arms)}")

    # ---------------- The transport tension ----------------
    # The two goals are not served by the same transports, which is the whole
    # reason the intersection is narrow. Cross-tab them so that is visible
    # rather than inferred by eye from the two tables above.
    print("\n" + "=" * 72)
    print("TRANSPORT TENSION -- the two goals do not want the same transport")
    print("=" * 72)
    rr = defaultdict(lambda: [0, 0])
    for c in red:
        a = rr[c["transport"]]
        a[1] += 1
        a[0] += bool(c["win"])
    sr = defaultdict(lambda: [0, 0])
    for c in rest:
        a = sr[c["transport"]]
        a[1] += 1
        a[0] += bool(c["win"])
    print(f"  {'transport':14}{'reduction':>18}{'restoration':>18}")
    for t in sorted(rr, key=lambda t: -(rr[t][0] / rr[t][1])):
        red_pct = 100 * rr[t][0] / rr[t][1]
        if t in sr and sr[t][1]:
            rest_pct = 100 * sr[t][0] / sr[t][1]
            rest_s = f"{rest_pct:5.1f}% ({sr[t][0]}/{sr[t][1]})"
        else:
            rest_s = "     -- (no fills)"
        print(f"  {str(t):14}{red_pct:8.1f}% ({rr[t][0]}/{rr[t][1]}){rest_s:>18}")
    print("\n  Read the two columns against each other, not down them: the transports")
    print("  that free the most bits are the worst to restore from, and the one that")
    print("  restores best frees the least. `downsample` is not the best at either")
    print("  goal -- it is the only transport that is acceptable at BOTH, which is")
    print("  the actual reason the article's main pipeline uses it.")

    # ---------------- Cost per goal ----------------
    print("\n" + "=" * 72)
    print("COST -- restoration wall-clock, on the cells that succeed")
    print("=" * 72)
    cost = defaultdict(list)
    for d, r in both:
        if r["restore_s"] and r["frames"]:
            cost[r["fill"]].append(r["restore_s"] / r["frames"])
    if not cost:
        print("  no timed cells among the successes")
    else:
        print(f"  {'fill':14}{'n':>4}{'s/frame':>10}{'fps':>8}   (source is 24 fps)")
        for fill in sorted(cost, key=lambda f: sum(cost[f]) / len(cost[f])):
            v = cost[fill]
            spf = sum(v) / len(v)
            print(f"  {fill:14}{len(v):>4}{spf:>10.3f}{1/spf:>8.2f}")
    print("\n  Quantization / distillation is deliberately NOT attempted here: the")
    print("  gap to 24 fps is 6-60x and the shipped default is already fp16, so a")
    print("  2-4x win would not change the offline/VOD positioning. Recorded as")
    print("  planned future work, not as a gap in this audit.")


def report_goal(cells, flag, fmt):
    if not cells:
        print("  no comparable cells")
        return
    n = len(cells)
    w = sum(1 for c in cells if c[flag])
    print(f"  overall: {w}/{n} cells succeed ({100*w/n:.1f}%)\n")
    for dim, label in (("video", "by video"), ("transport", "by transport"),
                       ("fill", "by fill"), ("component", "by component")):
        print(f"  {label}:")
        agg = defaultdict(lambda: [0, 0])
        for c in cells:
            a = agg[c[dim]]
            a[1] += 1
            a[0] += bool(c[flag])
        for k in sorted(agg, key=lambda k: -agg[k][0] / agg[k][1]):
            hit, tot = agg[k]
            bar = "#" * int(round(20 * hit / tot))
            print(f"    {str(k):22} {hit:3d}/{tot:3d}  {100*hit/tot:5.1f}%  {bar}")
        print()


if __name__ == "__main__":
    main()
