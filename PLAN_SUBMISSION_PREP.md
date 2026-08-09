# PRESLEY → TOMM: state, and what is left

Rewritten 2026-08-09, replacing the plan of 2026-08-07. That plan was written
before any of it ran; this one records what happened, what changed as a result,
and what a new session should do next.

**Read §1 and §6 before touching anything.** §6 is the register of things that
already went wrong here.

---

## 1. Status at a glance

| | 2026-08-07 | prev | now | target |
|---|---|---|---|---|
| main body pages | 46 | 28 | **23** | 23 |
| appendix pages | 0 | 4 | **5** | ≤5 |
| references pages | — | 2 | **2** | outside |
| total PDF pages | — | 32 | **29** | — |
| total words | 21,192 | 12,134 | **10,155** | — |
| tables | 32 | 9 | **4** | 6–8 |
| figures | 8 | 15 | **16** | ~13 |
| abstract words | 503 | 313 | **301** | ~250 |
| unresolved `\ref` | ? | **3** | **0** | 0 |
| CLAIM anchors | 44 | 44 | **44** | unchanged, always |

- Paper repo `main` @ `4386f0b`, clean, pushed to Overleaf.
- Code branch `claude/presley-submission-prep-eef0a0` @ `c8ee22a`, clean,
  pushed. It fast-forwarded `...-8b3b2d`, so that branch is an ancestor.
- Nothing is running. Full suite green: 535 passed, 17 deselected.

Render and measure with, from the code repo root:

```bash
tools/render_paper.sh /tmp/presley_render && python tools/paper_metrics.py --pages 29
```

`paper_metrics.py --pages` counts the **whole PDF**, so it still reports
"over budget"; that is the tool comparing 29 against 23, not a real overflow.
The 23-page limit applies to the main body, which is measured by where the
appendix starts:

```bash
pdftotext /tmp/presley_render/main.pdf - | awk '/Methodological Pitfalls/{print "body ends p"p+1; exit} /\f/{p++}'
```

**A caveat worth resolving before submitting.** That measure puts references
*outside* the 23 pages, because the bibliography currently prints after the
appendix. If TOMM counts references inside the limit, the real figure is 25,
not 23, and two more pages have to come out. Nobody has checked the venue's
wording; the 23/appendix-excluded convention was inherited from the previous
plan and this session kept it rather than re-litigating it.

## 2. What is done

**Wave 0 — revision tracking retired.** 119 `\rev{}` unwrapped, 16 `\del{}`
deleted, `ulem` dropped. Deleted text logged to
`archive/retired_del_spans_2026-08-07.txt`. Tool: `tools/retire_revision_marks.py`,
17 tests. Pre-retirement state is local tag `archive/tracked-changes-2026-08-07`
and commit `1bfd869` (Overleaf rejects tag pushes; the commit is in its history).

**W1a — equations now describe the code.** Five errors fixed, one serious: the
printed restoration algorithm's loop *never executed* for a binary strength map,
which is the configuration every reported result uses. Also: Eq. 3 said the
background sign is inverted where the code multiplies by 10; Eq. 4 described a
graded scheme that was built, measured and retired. Added an algorithm for the
real selection rule and an equation for the side channel. Pinned by
`tests/test_paper_equations.py`, which re-implements the printed equations in
plain loops so it cannot pass by comparing the code to itself. The extracted
`combine_removability` is byte-identical on three cached cells.

**W1b — JND retired.** Zero JND references in reviewer-visible text. The paper
had *already admitted* it knew of no source calibrating 0.05 as a JND for LPIPS
or DISTS, then gated ~60 claims on it, applying one constant to two metrics on
different scales. Claims now gate on paired significance: exact two-tailed sign
test, Holm-corrected over every candidate tried including losers, pair count
stated, and "underpowered" where the floor cannot be reached. Some claims got
weaker; that is the point.

**W1c — figure programme.** `tools/figkit.py` establishes the three-artefact
convention: every `plot_*.py` emits `<name>.pdf`, `<name>.json` and
`<name>.desc.tex`, the last an ACM `\Description{}` carrying a plain sentence
then the same data as compact JSON. That block never renders, is read by screen
readers, and is machine-parseable — so the numbers travel with the figure at
zero page cost. Eight new figures:

| figure | replaces |
|---|---|
| `restorer_catalogue` | 6 tables |
| `breadth` | 4 tables |
| `ablation` | 3 tables |
| `ratematched` | `tab:ratematched-n13` |
| `restorability`, `selection_cost`, `stage_timing`, `resolution_ladder` | new |

**W1d — structural cut, partly done.** 46 → 28 main-body pages. Evaluation
reorganised around the three goals; 23 tables retired; appendix built.

**W1e — per-stage timing.** All four components record `stage_times_seconds`
against one vocabulary (`src/presley/stagetiming.py`), output-only so no hash
moves. Campaign in `tools/stage_timing_campaign.py`: 12 cells × 3 trials, one
device, refuses to average across device populations. Results in
`docs/w1e_stage_timings.json`. Restoration is 70–89% of wall clock; degradation
costs more than encoding; **selection costs 0.006 s, about a thousandth of the
pipeline.**

**W1f — resolution ladder.** 144 runs, 6 videos × {360p, 720p, 1080p} × 4 QP ×
{baseline, presley}. Block size scales (8/12/16/24) so every rung keeps an 80×45
grid. FG-LPIPS BD-rate: 360p 3/6 (p=1.0), 720p 5/5 (p=0.0625), 1080p 5/5
(p=0.0625). **Underpowered, not a win**, and confounded — at fixed QP the higher
rungs run at half the bits per pixel, i.e. more starved, the regime that favours
the method.

**W1g — the corrected selection objective does not pay.** Screened first
(`tools/analyze_corrected_objective.py`), then raced. Damage predictor:
`src/presley/damagemodel.py`, leave-one-video-out, held-out ρ +0.400, and the
loader *refuses* a model that trained on the clip being scored. Race: 48 runs,
BG-LPIPS median +5.1%, better on 2/6, p=0.69; FG flat at +0.3% as the hard
exclusion requires. A null that is a net loss, since the rule starts +3.6% down
on rate. Full write-up: `docs/W1G_SELECTION_RACE.md`.

---

## 3. What is left, in order

**1. DONE — the cut, 28 → 23 main-body pages.** What actually moved the
number, in rough order of yield:

- *Figures were the budget, not words.* At 430×560pt the plots and their
  captions were ~3.4 of the 28 pages. Two were mostly whitespace by
  construction: `breadth` forced every panel to the tallest panel's row count,
  and `regime_reversal` drew four ladders as a 2×2 square. Redrawn wide in
  `tools/`, they went from h/w 0.87→0.64 and 0.76→0.33 at unchanged label size.
- *Relocation to the appendix*, which sits outside the limit: three detail
  tables, the metrics/significance methodology, and the two leaf restoration
  algorithms. The appendix is now full at 5 pages, so this lever is spent —
  three of the eight pitfalls were dropped to make room.
- *Deleting things said twice.* `sec:downsample-vs-uniform` carried ~330 words
  of verbatim duplicate paragraphs; the dog/pigs reversal was stated three
  times; the restorer catalogue appeared in both the method and the evaluation.
- Prose compression everywhere else, and caption trims.

**A defect was found and fixed on the way:** `fig:restorers` was cited three
times and the figure environment did not exist. The catalogue plot was
generated last session and never inserted, so the PDF rendered "Figure ??"
three times — through a render, a metrics pass and a commit. **Grep the
rendered text for `⁇` after every structural edit;** it is one command and it
would have caught this a session earlier:

```bash
pdftotext /tmp/presley_render/main.pdf - | grep -c '⁇'
```

**2. DONE-ish — abstract 313 → 301.** Short of the ~250 target. It is three
paragraphs and the third carries the article's main contribution; getting to
250 means dropping a claim, which is an editorial call rather than a trim.

**3. Remaining figures.** Qualitative crops (degraded / restored / pristine)
is still the one real gap — a generative restoration paper with no visual
example, and referee 2 asked. Frames are under `results/<hash>/restored_frames/`.
Note there is no page headroom left: a new figure has to displace something.

**4. Wave 3 — submission mechanics.**
- Rebuild `response_letter.tex`. It is **frozen and stale**: it cites tables by
  label and 23 of them no longer exist. Rebuild in one pass against final
  labels. A letter contradicting the manuscript is the worst available failure.
- `python tools/make_submission_copy.py` writes a referee-safe tree with the
  ~117 non-CLAIM markers stripped. The `.tex` source goes to TOMM, and the
  markers contain retraction histories and notes addressed to agents.
- Confirm the rendered bibliography contains FVMD (a past entry had no key).

**5. Optional, if a performance claim is wanted later.** The user has said this
must eventually become a performance paper. On current evidence it is not one:
the only unambiguous win is against ELVIS, our own predecessor, and there is no
demonstrated win over uniform downscale + client-side SR, which splits 2–4.
Closing that gap is new science, not a rewrite. A 6th usable video at 720p/1080p
would at least turn the resolution result from 5/5 p=0.0625 into a claim.

---

## 4. Open decisions — the user's, not yours

**A. The camel saturation invariant.** Five `camel` runs at 720p/1080p carry a
non-empty `invariant_failures` and are therefore uncitable. Consequence: the
resolution rungs are n=5 and **cannot** reach p=0.05; with camel they would be
6/6 at p=0.031. Appendix B now names this as the reason for the gap, so it is
load-bearing.

The evidence says it is not a numerical divergence: camel's *output* saturation
is flat across the whole ladder (0.89–0.93%) while pass/fail flips on ~0.1 pp of
drift in the *input*; the excess is 0.52–0.62 pp against the 1.1–8.7 pp of the
NAFNet failure the check was calibrated on; and quality degrades smoothly rather
than collapsing. Options: (a) leave them uncitable; (b) amend the invariant to
require a quality collapse as well as saturation, which is the property that
actually distinguishes the NAFNet case; (c) drop camel from the ladder.
Recommendation (b), but **do not raise a threshold to unblock your own campaign.**

**B. Four deleted files in the shared main checkout.** `HANDOFF.md`,
`HANDOFF-goals-wave.md`, `HANDOFF_PAPER_REWRITE.md` and
`NOISE_MODE_DECISION_REPORT.md` are deleted in the working tree of
`/home/itec/emanuele/presley` and were **not** deleted by this session. They are
still in `HEAD`, so `git checkout -- <file>` restores them. Another session may
be mid-edit. Left alone deliberately.

---

## 5. Landmarks

| path | what |
|---|---|
| `docs/W1G_SELECTION_RACE.md` | the selection race, its bounds and its scope |
| `docs/w1e_stage_timings.json` | per-stage timings, 12 cells × 3 trials |
| `config/damage_predictor.json` | 13 leave-one-video-out folds |
| `config/w1f_ladder_*.yaml`, `config/w1g_selection_race.yaml` | the run files |
| `tools/figkit.py` | the three-artefact figure convention |
| `sections/appendix.tex` | the pitfalls, provenance, pre-registrations |
| `archive/retired_del_spans_2026-08-07.txt` | every `\del{}` span removed |

---

## 6. Things that already went wrong here

**Your own new runs can break a published analysis.** The W1f ladder used the
same recipe as the published corpus but passed `restorer_params: {}` where those
runs pass `{denoise_strength: 1.0, tile: 400}`. That made
`analyze_ratematched_n13.py`'s arm selector ambiguous and it refused to run —
which is the only reason it was caught. Its ARMS spec now whitelists the two
values the published corpus uses. **After adding runs, re-run the analyses that
touch the same rungs and check they still reproduce.**

**Tables are not the page budget; words are.** Removing six tables bought one
page. The appendix and whole-result deletion are what move the count.

**A figure can invert its own finding.** The restorer catalogue drawn as
gain-over-control ranks blackout in-painting best — but blackout destroys more,
so a bigger recovery is a deeper hole, not a better picture. Absolute quality
inverts the ordering. Both panels are now mandatory.

**Check which arm a table actually reports before claiming to replace it.** The
first breadth figure drew 11 `presley_ai` clips and claimed all four breadth
tables; `tab:breadth` is the ELVIS bridge on 33 clips.

**Read the tool's own summary rather than re-averaging its rows.** Re-averaging
the printed per-video capture ratios gives 0.834 where the article cites 0.833.

**Environment, every time:**
- Figures: `PRESLEY_PAPER_DIR=/home/itec/emanuele/presley/68e8b6bb11d0dd9e62a67aef`.
  Without it a script run from a worktree writes a stray `Figures/` the
  manuscript never sees.
- Analysis tools run from a worktree need `--data-root /home/itec/emanuele/presley`
  — `results/` and `cache/` exist only in the main checkout.
- Runs needing worktree code need `PYTHONPATH=<worktree>/src`. **Do not copy
  source into the main checkout**; it is shared with other agents.
- `presley-run` exits 0 when every experiment fails. Verify result count against
  entry count; `grep -c 'Error running experiment' <log>`.
- Fresh runs have no region LPIPS. `presley-evaluate results/ --backfill-lpips`,
  and never two backfills at once.
- Overleaf's git bridge rejects tags and new branches. Push `main` only.

**Bounds get stated before the number and can still be wrong.** Two fired in
this session and both were the bound's fault, not the measurement's: selection
timing came in below a floor sized for pixel-level work when the operation is
block-level, and the resolution-ladder band was stated for the Goal-1 rate
saving and then applied to BG-LPIPS, where a positive value is the *accepted
cost* of relocation. Revise the bound with a stated reason; do not quietly widen
it.

**A figure can be cited and not exist, and everything still looks fine.**
`fig:restorers` was referenced three times with no figure environment
anywhere. `tectonic` printed no warning this session's tooling surfaced, the
page count looked plausible, and the render "succeeded". The check is one
command against the rendered text, and it belongs after every structural edit:
`pdftotext main.pdf - | grep -c '⁇'`.

**Redrawing a figure's layout breaks code that hardcoded the old shape.**
Turning `regime_reversal` from 2×2 to 1×4 raised `'Axes' object is not
subscriptable` from three separate places — axis labels, the annotated panels
and the legend handles — each indexing `axes[i][j]`. And the first 1×4 attempt
rendered with the four panel titles overlapping into each other, which the
page count does not notice. **Look at the rendered page after changing a
figure's aspect**, do not just read the new page total.

**Never delete a CLAIM.** 44 anchors, verified against `HEAD` after every
structural edit in this session. Once a script reported "0 comments kept" and
that had to be checked before proceeding — it was fine, the blocks sat outside
the replaced span, but the check is not optional.
