# Handoff — the HOLE wave is complete; all four closed (2026-08-02)

Supersedes the earlier 2026-08-02 handoffs. **Nothing is running. All four
HOLEs of the wave are closed, landed in the paper, and pushed to Overleaf.**
Code `main` is clean, CI green (run 30745316953).

The wave's finding, and the thing to carry into any next session:

> **Goal 1 (bit relocation) does not generalize. Goal 2 (restoration) does.**
> The same two added videos give the paper's *largest* restoration wins and bit
> relocation's *outright failures*. The two are not one axis.

This forced corrections to the abstract and the conclusion, which had asserted
the 26–29% starved saving unconditionally.

## What each set produced

| set | HOLE closed | outcome |
|---|---|---|
| **H4** | `tab:priced-trade` → `tab:budget-knee` | budget lever **inert below sa 0.25**; gain there is sub-JND (0.57×/0.83×). The table's fixed 0.25 sits exactly at the knee |
| **H3** | `tab:conditioned` → `tab:conditioned-breadth` | **Goal 1 fails.** dog/pigs need **+44.1%/+27.6%** more bits for equal FG quality vs −16.1%/−13.8% on the incumbents. Goal 2 survives on 3 of 4 |
| **H1b** | `tab:av1` → `tab:av1-breadth` | **the starved win does not generalize.** +18.3…+34.4% bits **and** −0.57…−0.86 dB FG at matched rate. The flip is **inverted** |
| **H2** | `tab:goal2` → `tab:goal2-breadth` | **replicates — the only set where no bound fired.** blackout clears JND in 4/4; freeze in none, as claimed |

Two independent mechanisms (H3 = `presley_ai` bs8 + Real-ESRGAN; H1b = `elvis`
bs16 + ProPainter) show the same reversal on the same content, which is why it
is a content result and not a component bug. **The responsible content property
is unidentified, and FG area is refuted** — dog 0.114 ≈ bear 0.111, and dog
reverses hardest. That is now an open problem of the same standing as the
selection objective.

Both fired alarms were closed the same way, in one command each: **re-run the
new analysis path on the old data and check it reproduces the published
number.** It did, exactly, both times.

## Suggested next steps (none started, none urgent)

- **The content axis.** Two mechanisms agree it is content; nothing predicts
  which way. n≥6 would be needed for any significance claim. This is the
  highest-value open question the wave created.
- **A 5th and 6th video** would move `tab:goal2-breadth` and the breadth
  results from descriptive to significant under hard rule 2b. Cheap now that
  the cost model is known (below).
- **16 pending fixed-QP `none`-restorer controls** — legitimate, never run,
  nobody has decided whether they are wanted. Unchanged for several sessions.
- `open-questions.md` / `dead-ends.md` are over the log's 300-line ceiling.

## Hard-won this session — read before running anything

- **`presley-run` exits 0 when every experiment in a wave fails.** It catches
  the per-entry error, prints `Error running experiment <hash>`, and continues.
  16 of H2's 32 cells failed this way and the chain moved on silently.
  **Verify result count against entry count; never trust exit 0.**
  `grep -c 'Error running experiment' <log>` gives it directly. In `bugs.md`.
- **`tab:goal2` spans two components** — freeze is `presley_ai`
  (`degradation`/`restorer`), blackout is `elvis` (`removal_mode`/`inpainter`).
  A generator treating it as one silently produces a half-invalid set.
  **Read a cited src hash's config before copying a table's recipe.**
- **Region LPIPS is missing from fresh runs, and not only for `elvis`.**
  Standard evaluation writes *overall* LPIPS but leaves FG/BG absent on
  `presley_ai` too, so a fresh set reads as "not citable" everywhere until
  `presley-evaluate results/ --backfill-lpips`. `--only <hash>` scopes it, and
  a second GPU is usually free.
- **Never run two backfills at once** — a scoped `--only` job and a global one
  will write the same `result.json`. Hit and corrected here; files were clean,
  but it is a corruption risk on a gitignored, hours-expensive artifact.
- **Check a bound at the operating point its own text names.** H1b's FG bound
  is specified *at matched rate*; checked at fixed QP it reads 0.88–1.20 dB and
  looks like a >1 dB alarm, versus −0.57…−0.86 dB where it belongs.
- **Do not pre-register a one-sided band on the quantity under test.** H3's FG
  band was −25…+5% with an alarm only for "too good"; the result came off the
  other end and the alarm text pointed at the wrong check.
- **Cost estimates were ~40× pessimistic.** ProPainter ran at 39–77 s, not
  ~49 min, at 640×360. Wall clock is dominated by *evaluation and backfill*,
  not the restorer. Do not schedule this class of work overnight — H3
  contradicted a landed claim and an overnight schedule would have hidden that
  for a day.
- **Editing YAML by hand-quoting breaks it.** A `_retired` reason containing
  inner double quotes terminated the YAML string and corrupted
  `experiments.yaml`; restored from git and redone letting `yaml.safe_dump` do
  the escaping. Always let the YAML writer escape.

## State

- `experiments.yaml` **896 entries**, 36 `_retired` (20 pre-existing + 16 H2
  blackout mis-specifications, retired in place with their reason).
- Results ~1030. Every run cited in the new tables has empty
  `invariant_failures`.
- Paper pushed to Overleaf, all edits `\rev{}`-tracked. **`pdflatex` is not
  installed on this host** — brace balance and tabular column counts were
  checked by hand; Overleaf is the first real compile. **If it fails, that is
  where to look, not in the numbers.**
- Only two `HOLE`s remain in the manuscript, both unrelated to this wave:
  `HOLE(tab:instantir-kill)` (DiffBIR — **ASK before any wire**) and
  `HOLE(sec:downsample-vs-uniform)`.

## Still open, unchanged

- **Q6 DiffBIR — ASK before any wire.** No mechanism argument → stay deferred.
- **Q8 DC-VSR — blocked upstream**; `_retired` in the queue.
- **0c** — `drift-straight` unexplained, ρ's CI includes zero.
  `tools/analyze_drift_straight_0c.py`, no GPU.

## Gotchas

- **The editable install points at the main checkout.** Tests run from a
  worktree import the *other* tree's `presley` unless `PYTHONPATH=<wt>/src`.
- **Never `import torch` before `presley`/`sqlite3`** (CXXABI_1.3.15) — it
  fails at *runtime*, in-function.
- `nohup … &` in a harness `Bash` call returns when the wrapper exits; that is
  not the job finishing. `pgrep -f <pattern>` matches your own check command —
  use log mtimes and result dirs.
- Carried: filter fixed-QP with `codec=svtav1`; NAFNet / Real-HAT-GAN reject
  `fp32=False`; `fg_protect=True`; BG-PSNR is never the Goal-2 verdict; never
  `rm` `results/`/`dataset/`/`cache/`.

## Landmarks

| Path | Why |
|---|---|
| `docs/HOLE_CLOSURE_WAVE.md` | bounds + **every result and alarm resolution**, start here |
| `tools/analyze_holes_{h3,h1b,h2}.py` | the analyses; bounds encoded and checked in output |
| `config/holes_wave_{h3,h1b,h2,h2b}.yaml` | scoped run-files; resume-safe |
| `sections/evaluation.tex` → `tab:av1-breadth` | worked example of clearing a HOLE |
| `research-log/bugs.md` | the exit-0 silent failure, and the two-component trap |
| `research-log/operational.md` | dispersion sweep, bound-setting lessons |
