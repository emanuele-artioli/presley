# Handoff — the HOLE wave is done bar H2's blackout half (2026-08-02)

Supersedes the earlier 2026-08-02 handoffs. **Three of the four HOLEs are
closed and pushed. Only `HOLE(tab:goal2)` is open**, and its data is already on
disk — what remains is reading the blackout half and writing it up.

The wave's headline: **Goal 1 (bit relocation) does not generalize; Goal 2
(restoration behaviour) does.** That reversal is the main thing to carry
forward, and it forced corrections to the abstract and conclusion.

## What is running

Nothing, or a short LPIPS backfill (`logs/h2b_driver.log`, last line
`backfill exited with 0` when done). Check that file before launching anything.

## Your task: finish `HOLE(tab:goal2)`

1. `python tools/analyze_holes_h2.py` — bounds encoded, JND from
   `presley.compare`. The **freeze half already reads and replicates**; the
   blackout half needs the backfill to have finished.
2. Fold into `HOLE(tab:goal2)`. Follow `tab:conditioned-breadth` or
   `tab:av1-breadth` as the worked example: GOAL + CLAIM with hashes, one NOTE
   per caveat, and the HOLE cleared **by the edit that lands the data**.
3. n=4 videos is under hard rule 2b's n≥6 — descriptive only, never
   "significant". `NOTE(tab:goal2)` records this finding has been **mis-stated
   twice in opposite directions**, so report per-cell direction, not a pooled
   verdict.

### What the freeze half already says (read, not guessed)

Restoration on a freeze fill is neutral-to-harmful in all four cells — best
restorer +0.12× (dog) to −1.07× JND (bear), all inside the pre-registered
−1.2…+1.2× wash band. Telea degrades it unambiguously at 2.29–3.15× JND,
extending the standing 2.36–2.76× result. **This replicates `CLAIM(tab:goal2)`**
rather than contradicting it.

## What landed this session

| | outcome |
|---|---|
| **H4 → `tab:budget-knee`** | budget lever **inert below sa 0.25**; gain there is sub-JND (0.57×/0.83×). `tab:priced-trade`'s fixed 0.25 sits exactly at the knee |
| **H3 → `tab:conditioned-breadth`** | **Goal 1 does not replicate.** dog/pigs need +44.1%/+27.6% *more* bits for equal FG quality vs −16.1%/−13.8% on the incumbents. Goal 2 survives on 3 of 4 videos |
| **H1b → `tab:av1-breadth`** | **the starved win does not generalize.** +18.3…+34.4% bits *and* −0.57…−0.86 dB FG at matched rate. The flip is **inverted** — frees bits at the mildest rung, costs at the most starved |
| **Abstract + conclusion** | corrected: they asserted the 26–29% saving unconditionally, which the new tables contradict |
| **Dispersion sweep** | clean; recorded in `operational.md` with the grep that redoes it |

**Two mechanisms, same reversal, same content** (H3 is `presley_ai` bs8 +
Real-ESRGAN; H1b is `elvis` bs16 + ProPainter). That is why it is treated as a
content result, not a component bug. **The content property is unidentified and
FG area is refuted** — dog 0.114 ≈ bear 0.111, and dog reverses hardest. This
is now an open problem of the same standing as the selection objective.

## Hard-won this session — read before running anything

- **`presley-run` exits 0 when every experiment in a wave fails.** It catches
  the per-entry error, prints `Error running experiment <hash>`, and continues.
  16 of H2's 32 cells failed this way and the chain moved on silently.
  **Verify result count against entry count; never trust exit 0.**
  `grep -c 'Error running experiment' <log>` gives it directly. In `bugs.md`.
- **`tab:goal2` spans two components** — freeze is `presley_ai`
  (`degradation`/`restorer`), blackout is `elvis`
  (`removal_mode`/`inpainter`). A generator treating it as one silently
  produces a half-invalid set. **Read a cited src hash's config before copying
  a table's recipe.**
- **Region LPIPS is missing from fresh runs, and not only for `elvis`.**
  Standard evaluation writes *overall* LPIPS but leaves FG/BG absent on
  `presley_ai` too, so a fresh set reads as "not citable" everywhere until
  `presley-evaluate results/ --backfill-lpips`. `--only <hash>` scopes it.
- **Do not run two backfills at once.** A scoped `--only` job and a global one
  will write the same `result.json`. I hit this and killed one; files were
  clean, but it is a corruption risk on a gitignored, hours-expensive artifact.
- **Check a bound at the operating point its own text names.** H1b's FG bound
  is specified *at matched rate*; checked at fixed QP it reads 0.88–1.20 dB and
  looks like a >1 dB alarm, versus −0.57…−0.86 dB where it belongs.
- **Do not pre-register a one-sided band on the quantity under test.** H3's FG
  band was −25…+5% with an alarm only for "too good"; the result came off the
  other end and the alarm text pointed at the wrong check.
- **Closing an alarm: reproduce a landed number with the new tool first.** Both
  fired alarms were closed in one command by re-running the new BD path on
  bear/camel and matching the published figures exactly.
- **The wave's cost estimates were ~40× pessimistic.** ProPainter ran at
  39–77 s, not ~49 min. Wall clock here is dominated by evaluation and
  backfill, not the restorer. Do not schedule this class of work overnight.

## State

- Code `main` clean and pushed; CI green as of the last code commit.
- Paper pushed to Overleaf, all edits `\rev{}`-tracked. **`pdflatex` is not
  installed here** — brace balance and column counts were checked by hand.
- `experiments.yaml` 896 entries; 36 `_retired` (20 pre-existing + 16 H2
  blackout mis-specifications, retired in place with their reason).
- Results ~1030.

## Still open, unchanged

- **Q6 DiffBIR — ASK before any wire.** No mechanism argument → stay deferred.
- **Q8 DC-VSR — blocked upstream**; `_retired`.
- **0c** — `drift-straight` unexplained, ρ's CI includes zero. No GPU needed.
- **New:** the content axis that decides whether bit relocation works. Two
  mechanisms agree it is content, nothing predicts which way.
- `open-questions.md` / `dead-ends.md` over the 300-line ceiling.
- 16 pending fixed-QP `none`-restorer controls: legitimate, never run, undecided.

## Gotchas

- **The editable install points at the main checkout.** Tests from a worktree
  import the *other* tree's `presley` unless `PYTHONPATH=<wt>/src`.
- **Never `import torch` before `presley`/`sqlite3`** (CXXABI_1.3.15).
- `nohup … &` in a harness `Bash` call returns when the wrapper exits; that is
  not the job finishing. `pgrep -f <pattern>` matches your own check command.
- Carried: filter fixed-QP with `codec=svtav1`; NAFNet / Real-HAT-GAN reject
  `fp32=False`; `fg_protect=True`; BG-PSNR is never the Goal-2 verdict; never
  `rm` `results/`/`dataset/`/`cache/`.

## Landmarks

| Path | Why |
|---|---|
| `docs/HOLE_CLOSURE_WAVE.md` | bounds + **every result and alarm resolution** |
| `tools/analyze_holes_{h3,h1b,h2}.py` | the analyses, bounds encoded in output |
| `config/holes_wave_{h3,h1b,h2,h2b}.yaml` | scoped run-files; resume-safe |
| `sections/evaluation.tex` → `tab:av1-breadth` | worked example of clearing a HOLE |
| `research-log/bugs.md` | the exit-0 silent failure, and the two-component trap |
| `research-log/operational.md` | dispersion sweep, bound-setting lessons |
