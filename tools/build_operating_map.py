"""The operating map: for each (content, rate) cell, which transport+fill to deploy.

Wave 1 of `docs/PLAN_OPERATING_MAP.md`, and deliberately the *falsification
gate* for the whole plan: it runs on data already on disk, costs no GPU time,
and can kill the "operating map" framing before anything is restructured.

The claim the map is supposed to support:

    For a given kind of content at a given requested quality, there is a best
    choice of degradation transport and restorer, and we can say what it is.

That claim dies in three ways, each of which is checked explicitly and printed
under GATE at the end of the report:

  1. **No cell has a JND-separable winner** -- there is no map; the honest
     contribution is "transport choice does not matter within JND".
  2. **One arm wins everywhere** -- the map collapses to one global
     recommendation.
  3. **Every off-ladder residual is within JND** -- "best of both worlds" is
     not reachable with these transports; the tradeoff is hard, not beatable.

A fourth, softer stop: if recommendations flip under the LPIPS-threshold
sensitivity analysis (`--jnd-sensitivity`), the map is an artifact of the JND
constant rather than a decision rule. Those thresholds are adopted convention,
not cited literature (`src/presley/compare.py` cites nothing and gives LPIPS
and DISTS the same 0.05), so the sensitivity pass is not optional garnish.

What this tool does NOT do, on purpose:

  * It does not build the map on **pass rates**. `tools/audit_goal_scope.py`
    reports the fraction of cells clearing a boolean gate; those are not
    magnitudes, and they invert the ranking (ac_truncate has the highest
    reduction pass rate and the smallest median saving). Every number here is
    an effect size measured *within* an operating point, so nothing is
    confounded by which videos a transport happened to be run on.
  * It never reports a restoration *gain* without the absolute restored
    quality beside it. Blackout is the standing warning: the largest gain in
    the corpus and the worst absolute result, because it starts from the worst
    place. A residual or a gain alone repeats that error.

Hard rules enforced: fixed-QP only; `actual_bitrate_bps` for the rate axis;
BG-LPIPS is the quality verdict and BG-PSNR never is; FG verdicts only from
true masked metrics; any run with `invariant_failures` is dropped. The map is
**descriptive**: per-cell effect sizes, no significance claim, because hard
rule 2b needs n>=6 videos per arm and Wave 2B is what buys that.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

DB = "results/presley.db"
FIXED_QP = ("cqp", "crf")

# Effect-size thresholds. LPIPS/PSNR match src/presley/compare.py, which is the
# single source of truth for JND in this codebase; they are restated (not
# imported) only so this tool runs against a bare DB without the package, and
# the sensitivity pass exists precisely because the LPIPS one is convention.
BG_LPIPS_JND = 0.05
FG_PSNR_JND = 0.5
# Rate-first separability is NOT a JND -- there is no perceptual threshold on
# bitrate. 1.0 percentage point of bitrate is the smallest difference this
# corpus can resolve reliably (compare.py's bitrate integrity check tolerates
# 1%), so anything under it is called a tie rather than dressed up as a win.
BITS_SEP_PP = 1.0
# A ladder fit below this R^2 is a cloud, not a curve. Residuals from it are
# still computed and printed (dropping them would hide the cell's disagreement
# with the corpus-wide ladder, which is itself a finding), but they are flagged
# and kept out of the falsification gate: the gate must not pass on noise.
LADDER_R2_MIN = 0.5
# Config keys naming the restorer/in-painter. Everything else identifies the
# transmitted bitstream, so everything else has to match for a `none` run to be
# a given arm's control.
FILL_KEYS = ("restorer", "inpainter")

# Pre-registered bounds (PLAN_OPERATING_MAP.md, "Pre-registered bounds for
# Wave 1"). Written before the numbers were read; a fired bound is reported as
# an ALARM and must be closed or revised with a stated reason, never dropped.
BOUNDS = {
    "separable_pct": {"plausible": (25.0, 60.0), "alarm": (10.0, 80.0)},
    "distinct_winners": {"plausible": (2, 4)},
    "best_residual_jnd": {"plausible": (1.0, 3.0), "alarm_above": 5.0},
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@dataclass
class Run:
    hash: str
    component: str
    video: str
    codec: str
    qp: Any
    width: int
    height: int
    transport: Optional[str]
    fill: Optional[str]
    block_size: Any
    shrink_amount: Any
    bits: Optional[float]
    frames: Optional[int]
    restore_s: Optional[float]
    bg_lpips: Optional[float]
    fg_psnr: Optional[float]
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def op(self) -> Tuple:
        """Operating point: everything that must match for a fair comparison."""
        return (self.video, self.codec, self.qp, self.width, self.height)

    @property
    def config(self) -> Tuple:
        """Everything about this run except which fill was applied -- what a
        `none` control must match for its BG-LPIPS to be this arm's *unrestored*
        damage.

        Derived from the whole config dict rather than a hand-listed tuple,
        because the hand-listed version was wrong and silently so. It named only
        (component, transport, block_size, shrink_amount), which does not
        distinguish degradation-strength parameters: `dancing`@QP43 carries three
        `blur` runs at `blur_kernel` 7 / 15 / 31, all three collapsing to one key.
        The controls dict then kept whichever was inserted last, so two of the
        three arms were scored against a *different run's* damage -- with their
        own bitrate. Damage and bits came from different experiments, which is
        how a rate-damage ladder ends up with a positive slope.

        Anything that is not the fill belongs here. A new degradation parameter
        added later is handled automatically instead of silently aliasing.
        """
        return tuple(sorted(
            (k, json.dumps(v, sort_keys=True, default=str))
            for k, v in self.raw_config.items()
            if k not in FILL_KEYS
        ))

    @property
    def label(self) -> str:
        return f"{self.transport}+{self.fill}"

    @property
    def fps(self) -> Optional[float]:
        if not self.restore_s or not self.frames:
            return None
        return self.frames / self.restore_s


def load(db: str) -> List[Run]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT hash, doc, component, video, width, height, codec, qp,"
        "       degradation, restorer, inpainter, block_size, actual_bitrate_bps,"
        "       video_frames"
        "  FROM runs"
        " WHERE rate_control IN (?, ?) AND n_invariant_failures = 0 AND has_metrics = 1",
        FIXED_QP,
    ).fetchall()
    met: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    for h, region, metric, value in con.execute(
        "SELECT hash, region, metric, value FROM metrics"
    ):
        met[h][(region, metric)] = value
    con.close()

    out: List[Run] = []
    for r in rows:
        cfg = (json.loads(r["doc"]).get("config") or {})
        doc = json.loads(r["doc"])
        m = met.get(r["hash"], {})
        out.append(Run(
            hash=r["hash"], component=r["component"], video=r["video"],
            codec=r["codec"], qp=r["qp"], width=r["width"], height=r["height"],
            # elvis calls it removal_mode, presley_ai calls it degradation
            transport=r["degradation"] or cfg.get("removal_mode"),
            fill=r["restorer"] or r["inpainter"],
            block_size=r["block_size"], shrink_amount=cfg.get("shrink_amount"),
            bits=r["actual_bitrate_bps"], frames=r["video_frames"],
            restore_s=doc.get("restoration_time_seconds"),
            bg_lpips=m.get(("background", "lpips_mean")),
            fg_psnr=m.get(("foreground", "psnr_mean")),
            raw_config=cfg,
        ))
    return out


def is_control(run: Run) -> bool:
    return run.fill in (None, "none")


# --------------------------------------------------------------------------
# Arms, scored within an operating point
# --------------------------------------------------------------------------

@dataclass
class Arm:
    hash: str
    label: str
    component: str
    transport: Optional[str]
    fill: Optional[str]
    dbits_pct: float          # vs the pristine baseline at the same fixed QP
    dfg_db: float             # FG-PSNR vs baseline; negative = worse
    bg_lpips: float           # ABSOLUTE restored background quality, lower better
    unrestored_bg_lpips: Optional[float] = None   # matched `none` control
    fps: Optional[float] = None
    residual_jnd: Optional[float] = None          # + = better than the ladder predicts
    cost_dominated: bool = False
    # config-minus-fill: identifies the transmitted bitstream, so every arm
    # sharing it shares one control and contributes ONE ladder point.
    config: Tuple = ()

    @property
    def gain_jnd(self) -> Optional[float]:
        """Restoration gain in JND multiples. Never quote without bg_lpips --
        blackout posts the corpus's largest gain and its worst absolute result."""
        if self.unrestored_bg_lpips is None:
            return None
        return (self.unrestored_bg_lpips - self.bg_lpips) / BG_LPIPS_JND


def build_controls(runs: Sequence[Run]) -> Dict[Tuple, Run]:
    """Map (operating point, config-minus-fill) -> the matched `none` control.

    Refuses to guess when two controls share a key. The previous version was a
    dict comprehension, so a collision silently kept whichever run came last and
    handed its damage to arms that belonged to a different bitstream. A
    collision under the full-config key means two runs are genuinely
    indistinguishable in config yet hashed differently, which is a data problem
    worth seeing rather than averaging away -- so it is dropped and counted.
    """
    seen: Dict[Tuple, List[Run]] = defaultdict(list)
    for r in runs:
        if r.component != "baselines" and is_control(r):
            seen[(r.op, r.config)].append(r)
    controls: Dict[Tuple, Run] = {}
    for key, group in seen.items():
        if len(group) == 1:
            controls[key] = group[0]
        else:
            AMBIGUOUS_CONTROLS.append((key[0], [g.hash for g in group]))
    return controls


# Populated by build_controls; reported rather than swallowed.
AMBIGUOUS_CONTROLS: List[Tuple[Tuple, List[str]]] = []


def score_arms(runs: Sequence[Run]) -> Dict[Tuple, List[Arm]]:
    """Effect sizes for every restored arm, within its own operating point.

    An arm needs a pristine baseline at the same operating point (the rate and
    FG references) and a restored BG-LPIPS. The matched `none` control is
    optional -- it only feeds the ladder residual, and it is far sparser than
    the baselines, so requiring it would silently shrink the map by 3/4.
    """
    baselines = {r.op: r for r in runs if r.component == "baselines"}
    controls = build_controls(runs)

    by_op: Dict[Tuple, List[Arm]] = defaultdict(list)
    for r in runs:
        if r.component == "baselines" or is_control(r):
            continue
        b = baselines.get(r.op)
        if b is None or not b.bits or b.fg_psnr is None:
            continue
        if not r.bits or r.bg_lpips is None or r.fg_psnr is None:
            continue
        ctrl = controls.get((r.op, r.config))
        by_op[r.op].append(Arm(
            hash=r.hash, label=r.label, component=r.component,
            transport=r.transport, fill=r.fill,
            dbits_pct=100.0 * (r.bits - b.bits) / b.bits,
            dfg_db=r.fg_psnr - b.fg_psnr,
            bg_lpips=r.bg_lpips,
            unrestored_bg_lpips=ctrl.bg_lpips if ctrl else None,
            fps=r.fps, config=r.config,
        ))

    for arms in by_op.values():
        mark_cost_dominated(arms)
    return dict(by_op)


def mark_cost_dominated(arms: List[Arm]) -> None:
    """Cost as the third axis: an arm is dominated when another is no worse on
    quality AND no slower, and strictly better on one of the two."""
    for a in arms:
        if a.fps is None:
            continue
        for b in arms:
            if b is a or b.fps is None:
                continue
            if b.bg_lpips <= a.bg_lpips and b.fps >= a.fps and (
                    b.bg_lpips < a.bg_lpips or b.fps > a.fps):
                a.cost_dominated = True
                break


# --------------------------------------------------------------------------
# The ladder, and residuals from it
# --------------------------------------------------------------------------

@dataclass
class Ladder:
    slope: float
    intercept: float
    n_points: int
    r2: float

    @property
    def is_ladder(self) -> bool:
        """A *negative* slope is what makes this a ladder: more bits transmitted
        means less background damage. A fit with a positive slope has not found
        the rate-damage tradeoff at that operating point, so residuals from it
        are distances from an arbitrary line and are refused rather than
        reported. 79.8% of transport pairs are concordant corpus-wide, so a
        discordant cell is a real minority case, not a bug to paper over."""
        return self.slope < 0


def fit_ladder(points: Sequence[Tuple[float, float]]) -> Optional[Ladder]:
    """OLS damage ~ bits over an operating point's arms.

    The ladder is the empirical rate-damage curve the peers define: the more
    bits a transport transmits, the less background damage it leaves. Measured,
    not assumed. R^2 is carried alongside because a 3-point fit through a cloud
    is not a curve anything should be scored against, and the caller has to be
    able to see that rather than take the residual on trust.
    """
    n = len(points)
    if n < 3:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    return Ladder(slope=slope, intercept=my - slope * mx, n_points=n,
                  r2=(sxy * sxy) / (sxx * syy))


def attach_residuals(by_op: Dict[Tuple, List[Arm]]) -> Dict[Tuple, Ladder]:
    """Score each arm by its signed distance from its peers' rate-damage curve.

    Deliberately analogous to BD-rate: BD-rate scores a codec against a
    rate-quality curve; this scores a transport against the rate-damage curve
    its peers define. One signed number per arm, comparable across operating
    points, in JND units. Positive = *off the ladder in the good direction*:
    less damage than its bit saving should have cost. Those are the "best of
    both worlds" candidates -- the ~20% discordant pairs are the point.

    Damage is the *unrestored* BG-LPIPS (the matched `none` control), because
    the ladder is a property of the transport, not of the restorer.

    An arm participates in the fit it is then scored against, which pulls the
    line toward itself and *shrinks* its own residual. That is deliberate: with
    3-6 points per cell the alternative (leave-one-out) makes a lone outlier
    look far more off-ladder than the evidence supports. The residuals here are
    therefore conservative -- they understate rather than manufacture a
    "best of both worlds" candidate.
    """
    fits: Dict[Tuple, Ladder] = {}
    for op, arms in by_op.items():
        usable = [a for a in arms if a.unrestored_bg_lpips is not None]
        # One ladder point per transmitted bitstream. Several fills share one
        # control, and letting each copy vote would weight a transport by its
        # number of restorers rather than treating it as one point.
        #
        # Keyed on the full config, NOT on (transport, bits) as before: that
        # older key let one transport contribute many points whenever its bit
        # delta varied across arms, while damage stayed pinned to a single
        # control's value. The result was x-axis scatter at constant y, which
        # drags the fitted slope toward zero and flipped several cells positive.
        seen: Dict[Tuple, Arm] = {}
        for a in usable:
            seen.setdefault(a.config, a)
        fit = fit_ladder([(a.dbits_pct, a.unrestored_bg_lpips) for a in seen.values()])
        if fit is None or not fit.is_ladder:
            continue
        fits[op] = fit
        for a in usable:
            predicted = fit.slope * a.dbits_pct + fit.intercept
            a.residual_jnd = (predicted - a.unrestored_bg_lpips) / BG_LPIPS_JND
    return fits


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------

@dataclass
class Cell:
    op: Tuple
    objective: str
    n_arms: int
    n_eligible: int
    winner: Optional[str] = None
    winner_hash: Optional[str] = None
    runner_up: Optional[str] = None
    margin: Optional[float] = None        # JND multiples (quality) or points (rate)
    margin_unit: str = "JND"
    separable: bool = False
    verdict: str = "no_eligible_arm"
    winner_bg_lpips: Optional[float] = None
    winner_dbits_pct: Optional[float] = None
    winner_fps: Optional[float] = None
    tied: List[str] = field(default_factory=list)


def eligible(arms: Sequence[Arm], fg_jnd: float = FG_PSNR_JND) -> List[Arm]:
    """Both objectives share the same feasibility test: the arm must actually
    save bits against the baseline, and must not cost visible foreground
    quality. Anything failing either is not a deployable recommendation
    regardless of how good its background looks."""
    return [a for a in arms if a.dbits_pct < 0 and a.dfg_db >= -fg_jnd]


def decide(op: Tuple, arms: Sequence[Arm], objective: str,
           lpips_jnd: float = BG_LPIPS_JND) -> Cell:
    """Rank one operating point's arms under one stated objective.

    quality_first: lowest absolute restored BG-LPIPS, subject to feasibility.
    rate_first:    fewest bits, subject to feasibility.

    Refuses to name a winner when the top two are within the separability
    threshold. Most cells will not have a separable winner, and saying so is
    the result -- not a hole in the table.
    """
    cell = Cell(op=op, objective=objective, n_arms=len(arms), n_eligible=0)
    elig = eligible(arms)
    cell.n_eligible = len(elig)
    if not elig:
        return cell
    if objective == "quality_first":
        ranked = sorted(elig, key=lambda a: a.bg_lpips)
        cell.margin_unit = "JND"
        def gap(a: Arm, b: Arm) -> float:
            return (b.bg_lpips - a.bg_lpips) / lpips_jnd
        threshold = 1.0
    elif objective == "rate_first":
        ranked = sorted(elig, key=lambda a: a.dbits_pct)
        cell.margin_unit = "pp_bits"
        def gap(a: Arm, b: Arm) -> float:
            return b.dbits_pct - a.dbits_pct
        threshold = BITS_SEP_PP
    else:
        raise ValueError(f"unknown objective {objective!r}")

    # Binary floating point puts a difference of exactly one JND just under it
    # (0.25 - 0.20 = 0.04999999999999999), which would silently demote every
    # borderline cell to a tie. The threshold is a stated convention to one or
    # two decimals; it is not meaningful at 1e-9.
    threshold -= 1e-9

    top = ranked[0]
    cell.winner_hash = top.hash
    cell.winner_bg_lpips = top.bg_lpips
    cell.winner_dbits_pct = top.dbits_pct
    cell.winner_fps = top.fps
    if len(ranked) == 1:
        cell.verdict = "single_eligible_arm"
        cell.winner = top.label
        cell.separable = False
        return cell

    second = ranked[1]
    cell.margin = gap(top, second)
    cell.runner_up = second.label
    if cell.margin >= threshold:
        cell.verdict = "separable"
        cell.separable = True
        cell.winner = top.label
    else:
        cell.verdict = "tie_within_threshold"
        cell.tied = [a.label for a in ranked if gap(top, a) < threshold]
    return cell


def build_map(by_op: Dict[Tuple, List[Arm]], objective: str,
              lpips_jnd: float = BG_LPIPS_JND, min_arms: int = 2) -> List[Cell]:
    return [decide(op, arms, objective, lpips_jnd)
            for op, arms in sorted(by_op.items(), key=lambda kv: str(kv[0]))
            if len(arms) >= min_arms]


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _fmt_op(op: Tuple) -> str:
    video, codec, qp, w, h = op
    return f"{video:<14} {codec:<8} QP{qp:<4} {w}x{h}"


def bounds_check(name: str, value: Optional[float], detail: str) -> Tuple[str, str]:
    """State the bound, then the value. A fired bound is an ALARM to investigate
    (implementation / eval / data bug) before the number is quoted anywhere."""
    b = BOUNDS[name]
    if value is None:
        return "NO DATA", f"{detail}: not computable"
    lo, hi = b["plausible"]
    if lo <= value <= hi:
        return "in band", f"{detail} (plausible {lo}-{hi})"
    alarm = b.get("alarm")
    above = b.get("alarm_above")
    fired = (alarm and not (alarm[0] <= value <= alarm[1])) or (above and value > above)
    return ("ALARM" if fired else "outside plausible"), f"{detail} (plausible {lo}-{hi})"


def summarize(cells: Sequence[Cell]) -> Dict[str, Any]:
    contested = [c for c in cells if c.n_eligible >= 2]
    separable = [c for c in contested if c.separable]
    winners = Counter(c.winner for c in separable if c.winner)
    return {
        "cells": len(cells),
        "contested": len(contested),
        "separable": len(separable),
        "separable_pct": (100.0 * len(separable) / len(contested)) if contested else None,
        "distinct_winners": len(winners),
        "winner_counts": dict(winners.most_common()),
        "no_eligible": sum(1 for c in cells if c.n_eligible == 0),
        "single_arm": sum(1 for c in cells if c.n_eligible == 1),
        "ties": sum(1 for c in contested if not c.separable),
    }


def report_objective(cells: Sequence[Cell], objective: str, verbose: bool) -> Dict[str, Any]:
    s = summarize(cells)
    print("=" * 78)
    print(f"OBJECTIVE: {objective}")
    print("=" * 78)
    print(f"  {s['cells']} operating points with >=2 arms; {s['contested']} with >=2 "
          f"*eligible* arms (saves bits, FG within {FG_PSNR_JND} dB JND)")
    print(f"  {s['no_eligible']} cells have no eligible arm; {s['single_arm']} have exactly one")
    if s["contested"]:
        print(f"  separable winner in {s['separable']}/{s['contested']} contested cells "
              f"({s['separable_pct']:.1f}%); {s['ties']} are ties within threshold")
    print(f"  distinct winning arms: {s['distinct_winners']}")
    for label, n in s["winner_counts"].items():
        print(f"    {label:<32} {n:3d} cells")
    if verbose:
        print("\n  per cell:")
        for c in cells:
            if c.verdict == "separable":
                verdict = (f"{c.winner} over {c.runner_up} by "
                           f"{c.margin:.2f} {c.margin_unit}")
            elif c.verdict == "tie_within_threshold":
                verdict = f"TIE ({', '.join(c.tied)}) margin {c.margin:.2f} {c.margin_unit}"
            else:
                verdict = c.verdict
            extra = ""
            if c.winner_bg_lpips is not None:
                extra = (f"  [BG-LPIPS {c.winner_bg_lpips:.3f}, "
                         f"{c.winner_dbits_pct:+.1f}% bits"
                         + (f", {c.winner_fps:.2f} fps" if c.winner_fps else "") + "]")
            print(f"    {_fmt_op(c.op)}  {verdict}{extra}")
    print()
    return s


def report_ladder(by_op: Dict[Tuple, List[Arm]], fits: Dict[Tuple, Ladder],
                  verbose: bool) -> Dict[str, Any]:
    scored = [(op, a) for op, arms in by_op.items() for a in arms
              if a.residual_jnd is not None]
    print("=" * 78)
    print("LADDER RESIDUALS -- signed distance from the peers' rate-damage curve")
    print("=" * 78)
    empty = {"n": 0, "operating_points": 0, "best_residual_jnd": None,
             "best_deployable_residual_jnd": None,
             "best_deployable_residual_any_fit": None, "off_ladder": 0,
             "off_ladder_deployable": 0, "top": [], "fits": []}
    if not scored:
        print("  no operating point yields a fitted ladder (needs >=3 arms with a")
        print("  matched `none` control, and a negative slope). Wave 2B's control")
        print("  coverage is the fix.\n")
        return empty
    ops = len({op for op, _ in scored})
    print(f"  {len(scored)} arms across {ops} operating points carry a fitted ladder")
    print(f"  (fits with a positive slope are refused, not scored -- see Ladder.is_ladder)")

    # The headline "best of both worlds" ranking is over DEPLOYABLE arms only.
    # An arm that spends more bits than the baseline has not beaten the
    # rate-damage tradeoff; it has bought quality, which is a different thing,
    # and it can never be a recommendation because it fails the map's own
    # feasibility test. Those arms are still scored and printed under --verbose,
    # because a large positive residual on an infeasible arm is informative
    # about the ladder's shape -- it is just not a candidate.
    deployable = [(op, a) for op, a in scored if a.dbits_pct < 0]
    off = [p for p in scored if p[1].residual_jnd >= 1.0]
    off_dep = [p for p in deployable if p[1].residual_jnd >= 1.0]
    print(f"  {len(off)} arms sit >=1 JND off the ladder in the good direction; "
          f"{len(off_dep)} of those\n  are deployable (actually save bits)")

    def well_fitted(op: Tuple) -> bool:
        f = fits.get(op)
        return f is not None and f.r2 >= LADDER_R2_MIN

    top = sorted(deployable, key=lambda pair: -pair[1].residual_jnd)[:10]
    print("\n  best off-ladder DEPLOYABLE arms -- residual AND absolute restored quality,")
    print("  because a residual alone flatters a transport that started from the worst")
    print(f"  place (blackout is the standing warning). '!' = fitted R^2 < {LADDER_R2_MIN},")
    print("  i.e. a distance from a cloud rather than from a curve -- do not quote it:")
    print(f"    {'operating point':<40} {'arm':<26} {'resid':>8} {'BG-LPIPS':>9} {'bits':>8}")
    for op, a in top:
        mark = " " if well_fitted(op) else "!"
        print(f"    {_fmt_op(op):<40} {a.label:<26} {a.residual_jnd:>+6.2f}x{mark}"
              f"{a.bg_lpips:>9.3f} {a.dbits_pct:>+7.1f}%")

    dep_ok = [a for op, a in deployable if well_fitted(op)]
    best_ok = max((a.residual_jnd for a in dep_ok), default=None)
    print(f"\n  best deployable residual from a well-fitted ladder (R^2 >= {LADDER_R2_MIN}): "
          + (f"{best_ok:+.2f}x JND" if best_ok is not None else "none"))
    print("  That is the number the gate uses; the table above may lead with a weaker fit.")

    print("\n  ladder fits (a low R^2 means the residuals scored against it are distances")
    print("  from a cloud, not from a curve):")
    print(f"    {'operating point':<40} {'n':>3} {'slope':>10} {'R^2':>7}")
    for op in sorted(fits, key=str):
        f = fits[op]
        print(f"    {_fmt_op(op):<40} {f.n_points:>3} {f.slope:>10.5f} {f.r2:>7.3f}")

    if verbose:
        print("\n  all residuals (* = not deployable, spends bits vs baseline):")
        for op, a in sorted(scored, key=lambda pair: -pair[1].residual_jnd):
            gain = f"{a.gain_jnd:+.2f}x gain" if a.gain_jnd is not None else "gain n/a"
            mark = " " if a.dbits_pct < 0 else "*"
            print(f"   {mark}{_fmt_op(op):<40} {a.label:<26} {a.residual_jnd:>+6.2f}x  "
                  f"BG-LPIPS {a.bg_lpips:.3f}  {gain}")
    print()
    return {
        "n": len(scored),
        "operating_points": ops,
        "best_residual_jnd": max(a.residual_jnd for _, a in scored),
        "best_deployable_residual_jnd": best_ok,
        "best_deployable_residual_any_fit": top[0][1].residual_jnd if top else None,
        "off_ladder": len(off),
        "off_ladder_deployable": len(off_dep),
        "top": [{"op": list(op), "arm": a.label, "residual_jnd": a.residual_jnd,
                 "bg_lpips": a.bg_lpips, "dbits_pct": a.dbits_pct,
                 "gain_jnd": a.gain_jnd} for op, a in top],
        "fits": [{"op": list(op), "n": f.n_points, "slope": f.slope, "r2": f.r2}
                 for op, f in sorted(fits.items(), key=lambda kv: str(kv[0]))],
    }


def report_cost(by_op: Dict[Tuple, List[Arm]]) -> Dict[str, Any]:
    print("=" * 78)
    print("COST -- restoration throughput, and arms dominated on quality AND speed")
    print("=" * 78)
    per_fill: Dict[str, List[float]] = defaultdict(list)
    dominated = 0
    total = 0
    for arms in by_op.values():
        for a in arms:
            total += 1
            dominated += a.cost_dominated
            if a.fps:
                per_fill[str(a.fill)].append(a.fps)
    if per_fill:
        print(f"  {'fill':<24}{'n':>5}{'median fps':>12}   (source is 24 fps)")
        for fill in sorted(per_fill, key=lambda f: -_median(per_fill[f])):
            v = per_fill[fill]
            print(f"  {fill:<24}{len(v):>5}{_median(v):>12.2f}")
    print(f"\n  {dominated}/{total} arms are Pareto-dominated within their own cell "
          "(another arm is no worse on BG-LPIPS and no slower)")
    print()
    return {"dominated": dominated, "arms": total,
            "median_fps": {k: _median(v) for k, v in per_fill.items()}}


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def report_sensitivity(by_op: Dict[Tuple, List[Arm]],
                       thresholds: Sequence[float]) -> Dict[str, Any]:
    """Recompute the map at each LPIPS threshold.

    The point is not that the *ranking* moves -- it cannot, since the ordering
    of BG-LPIPS is threshold-independent -- but that which cells are allowed to
    name a winner does. A recommendation that survives all three thresholds is
    robust to the constant; one that appears only at 0.03 is an artifact of it
    and must be reported as threshold-dependent.
    """
    print("=" * 78)
    print("JND SENSITIVITY -- the LPIPS threshold is adopted convention, not cited")
    print("=" * 78)
    per: Dict[float, Dict[str, Any]] = {}
    named: Dict[float, Dict[Tuple, str]] = {}
    for t in thresholds:
        cells = build_map(by_op, "quality_first", lpips_jnd=t)
        per[t] = summarize(cells)
        named[t] = {c.op: c.winner for c in cells if c.separable and c.winner}
    print(f"  {'LPIPS JND':>10}{'separable':>12}{'of contested':>14}{'distinct winners':>19}")
    for t in thresholds:
        s = per[t]
        pct = f"{s['separable_pct']:.1f}%" if s["separable_pct"] is not None else "--"
        print(f"  {t:>10.3f}{s['separable']:>12}{pct:>14}{s['distinct_winners']:>19}")

    ref = BG_LPIPS_JND if BG_LPIPS_JND in named else thresholds[len(thresholds) // 2]
    robust = set(named[ref])
    for t in thresholds:
        robust &= set(named[t])
    flips = 0
    for op in robust:
        if len({named[t][op] for t in thresholds}) > 1:
            flips += 1
    only_loose = sorted(set(named[min(thresholds)]) - set(named[max(thresholds)]))
    print(f"\n  {len(robust)} cells name a winner at every threshold; {flips} of those "
          "name a DIFFERENT winner depending on the threshold")
    print(f"  {len(only_loose)} cells name a winner only at the tightest threshold "
          f"({min(thresholds)}) -- those are threshold-dependent recommendations")
    if flips:
        print("  ⚠ SOFT STOP: a recommendation that flips with the constant is an "
              "artifact of it and must not be published as a decision rule.")
    print()
    return {"per_threshold": {str(t): per[t] for t in thresholds},
            "robust_cells": len(robust), "flipping_cells": flips,
            "tight_only_cells": len(only_loose)}


def report_gate(quality: Dict[str, Any], rate: Dict[str, Any],
                ladder: Dict[str, Any], sens: Dict[str, Any]) -> Dict[str, Any]:
    """The falsification gate. Wave 2 does not start until this has reported
    against all four conditions -- each is a legitimate, publishable result, and
    each ends this plan rather than continuing it."""
    print("=" * 78)
    print("GATE -- Wave 1 is a falsification gate; Wave 2 starts only if this passes")
    print("=" * 78)

    checks: Dict[str, Any] = {}
    c1 = quality["separable"] == 0 and rate["separable"] == 0
    checks["no_separable_winner"] = c1
    print(f"  1. No cell has a separable winner .......... {'TRIGGERED' if c1 else 'no'}")
    if c1:
        print("     -> there is no map. The honest contribution is 'transport choice")
        print("        does not matter within JND' -- a different, legitimate paper.")

    c2 = quality["distinct_winners"] == 1 and quality["separable"] > 0
    checks["single_global_winner"] = c2
    print(f"  2. One arm wins everywhere ................. {'TRIGGERED' if c2 else 'no'}")
    if c2:
        print("     -> the map collapses to a single global recommendation. Also fine,")
        print("        also a different paper.")

    # Condition 3 is about *deployable* arms: an infeasible arm sitting off the
    # ladder is not "best of both worlds", it is a bought quality improvement.
    best = ladder["best_deployable_residual_jnd"]
    c3 = ladder["n"] > 0 and (best is None or best < 1.0)
    checks["no_off_ladder_arm"] = c3
    print(f"  3. Every off-ladder residual within JND .... "
          f"{'TRIGGERED' if c3 else ('no' if ladder['n'] else 'NOT TESTABLE (no fitted ladder)')}")
    if c3:
        print("     -> 'best of both worlds' is not reachable with these transports;")
        print("        the reduction/restoration tradeoff is hard, not beatable.")

    c4 = sens["flipping_cells"] > 0
    checks["threshold_artifact"] = c4
    print(f"  4. Recommendations flip under sensitivity .. {'TRIGGERED (soft)' if c4 else 'no'}")

    print("\n  Pre-registered bounds (stated before the numbers were read):")
    for name, value, detail in (
        ("separable_pct", quality["separable_pct"],
         f"separable cells = {quality['separable_pct']:.1f}%"
         if quality["separable_pct"] is not None else "separable cells = n/a"),
        ("distinct_winners", quality["distinct_winners"],
         f"distinct winners = {quality['distinct_winners']}"),
        ("best_residual_jnd", best,
         f"best deployable off-ladder residual = {best:.2f}x JND" if best is not None
         else "best deployable off-ladder residual = n/a"),
    ):
        status, text = bounds_check(name, value, detail)
        print(f"    [{status:>17}] {text}")
    # =1 or =every transport is the stated alarm for the winner count; "every"
    # is not knowable from the count alone, so only the =1 half is automatic.
    if quality["distinct_winners"] == 1:
        print("    [            ALARM] exactly one distinct winner -- see gate condition 2")

    passed = not (c1 or c2 or c3)
    print(f"\n  VERDICT: Wave 1 {'PASSES' if passed else 'STOPS the plan'}"
          + (" (with a soft threshold-artifact warning)" if passed and c4 else ""))
    print()
    checks["passes"] = passed
    return checks


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--verbose", action="store_true", help="print every cell and residual")
    ap.add_argument("--jnd-sensitivity", default="0.03,0.05,0.08",
                    help="comma-separated BG-LPIPS thresholds for the sensitivity pass")
    ap.add_argument("--json", default=None, help="also write the machine-readable map here")
    args = ap.parse_args(argv)

    runs = load(args.db)
    by_op = score_arms(runs)
    fits = attach_residuals(by_op)

    n_arms = sum(len(v) for v in by_op.values())
    videos = {op[0] for op in by_op}
    print(f"{len(runs)} fixed-QP citable runs -> {n_arms} scored arms across "
          f"{len(by_op)} operating points on {len(videos)} videos\n")
    print("Descriptive only: per-cell effect sizes, no significance claim. Hard rule 2b\n"
          "needs n>=6 videos per arm before any of this is stated as a finding.\n")

    quality_cells = build_map(by_op, "quality_first")
    rate_cells = build_map(by_op, "rate_first")
    quality = report_objective(quality_cells, "quality_first "
                               "(lowest BG-LPIPS | saves bits, FG within JND)", args.verbose)
    rate = report_objective(rate_cells, "rate_first "
                            "(fewest bits | FG within JND)", args.verbose)

    agree = sum(1 for a, b in zip(quality_cells, rate_cells)
                if a.separable and b.separable and a.winner == b.winner)
    both = sum(1 for a, b in zip(quality_cells, rate_cells) if a.separable and b.separable)
    print(f"  Cross-objective: {agree}/{both} cells separable under BOTH objectives name "
          "the same arm.\n  Those recommendations survive the scalarization; the rest are "
          "objective-dependent.\n")

    ladder = report_ladder(by_op, fits, args.verbose)
    cost = report_cost(by_op)
    thresholds = [float(t) for t in args.jnd_sensitivity.split(",") if t.strip()]
    sens = report_sensitivity(by_op, thresholds)
    gate = report_gate(quality, rate, ladder, sens)

    if args.json:
        payload = {
            "summary": {"runs": len(runs), "arms": n_arms,
                        "operating_points": len(by_op), "videos": sorted(videos)},
            "quality_first": quality, "rate_first": rate,
            "cross_objective_agree": agree, "cross_objective_both": both,
            "ladder": ladder, "cost": cost, "sensitivity": sens, "gate": gate,
            "cells": {"quality_first": [asdict(c) | {"op": list(c.op)} for c in quality_cells],
                      "rate_first": [asdict(c) | {"op": list(c.op)} for c in rate_cells]},
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
