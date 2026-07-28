# Handoff — InstantIR kill documented; NAFNet / Real-HAT next (2026-07-28)

## What triggered this handoff

Session boundary after documenting Q1–Q3 findings in the paper
(`tab:instantir-kill`, paper commit `c67b3f6`). Next work is **code
integration** (NAFNet Q5, then Real-HAT-GAN Q4), not more InstantIR
tuning. **No Overleaf push** unless the human asks — paper is committed
locally in `68e8b6bb11d0dd9e62a67aef/` only.

## One-paragraph summary

PRESLEY degrades background blocks (downsample / blur / holes) under
**fixed QP**, restores them client-side, and must show Goal 1 (bits freed
for FG) and Goal 2 (BG restored toward original on LPIPS/DISTS). The
blur+InstantIR arm was retired in `tab:conditioned`; Q2 re-ran InstantIR
with InstantX-style settings (`steps=20`, `creative_start=0.7`,
`preview_start=0.2`) and the kill **stands**. Next: wire a CNN deblur
gauge (**NAFNet**) into the same blur recipe, then a recent SR GAN
(**Real-HAT-GAN**) as Real-ESRGAN's twin. Queue: [`docs/EXPERIMENTS_QUEUED.md`](docs/EXPERIMENTS_QUEUED.md).

## Current state (verified 2026-07-28)

### Paper repo `68e8b6bb11d0dd9e62a67aef/`

- Branch: `main`, commit **`c67b3f6`** — `tab:instantir-kill` + RESEARCH_LOG
  InstantIR reconfirmation. **Not pushed to Overleaf.**
- Table sits after `tab:conditioned` prose; `HOLE(tab:instantir-kill)` waits
  for NAFNet / DiffBIR rows and optional BG-DISTS.

### Code repo `presley/`

- Branch: `main` tracking `origin/main`.
- **Committed:** `4063b34` BSRGAN KAIR weight remap **and** InstantIR
  `num_inference_steps` plumbing in `restoration.py`.
- **Uncommitted (carry forward):**
  - `src/presley/components/presley_ai.py` — pass `num_inference_steps` from
    `restorer_params` (needed for Q2 yaml; without it defaults stay at 1)
  - `tests/test_instantir_inference_steps.py` (new)
  - `experiments.yaml` — Q2/Q3 entries + hashes
  - `docs/EXPERIMENTS_QUEUED.md`, `HANDOFF.md`
- **Weights on disk:** `weights/BSRGAN.pth`, `weights/BSRGAN._wrapped.pth`.
  **Missing:** NAFNet GoPro `.pth`, Real-HAT-GAN `.pth` (download first).

### Experiment results (citable: `invariant_failures=[]`, CRF)

| ID | Hash | Notes |
|---|---|---|
| Q2 InstantIR corrected bear | `36fb007975de0593` | BG-LPIPS 0.387; ~774 s |
| Q2 InstantIR corrected camel | `afd6dc1e5fa2ebe1` | BG-LPIPS 0.328; ~836 s |
| InstantIR as-run bear/camel | `81d840b4ef6831f2` / `732c285f818a1453` | CLAIM(tab:conditioned) |
| unsharp bear/camel | `e26598d90f189991` / `6857ccfe20a8c701` | still best blur restorer |
| Q3 BSRGAN bear/camel | `432569abedc41fca` / `e437776f348cadbe` | within-JND of Real-ESRGAN; RE slightly better LPIPS |
| Q1 LPIPS | 9 D6 hashes | backfilled; still `fast_only` for other deferred metrics |

### Running now

GPUs idle (device 0 ~6 GB other-user; device 1 empty). No PRESLEY job should
be running — if `presley-run … camel … instantir` zombies remain, they are
safe to kill (results already written).

```bash
nvidia-smi
ps -ef | rg 'presley-run' | rg -v rg
```

## Implementation plan (next session)

### Wave 1 — NAFNet (Q5)  *[priority: InstantIR kill needs a CNN gauge]*

Plan source: explore agent notes in prior session + matrix in
`docs/EXPERIMENTS_QUEUED.md`.

1. **Download weights** (no pip into `presley` env):
   ```bash
   # ~272 MB GoPro deblur
   hf download mikestealth/nafnet-models NAFNet-GoPro-width64.pth --local-dir weights/
   # optional smoke: NAFNet-GoPro-width32.pth (~69 MB)
   ```
2. **Vendor arch** — new `src/presley/nafnet_arch.py`: `LayerNorm2d` +
   `NAFBlock` + `NAFNet` (self-contained; **do not** install megvii NAFNet /
   forked basicsr; **avoid** OpenCV ONNX path).
3. **Wire restore** in `restoration.py`:
   - `get_nafnet` / load `ckpt["params"]`
   - `restore_with_nafnet_adaptive` — InstantIR-shaped dir I/O; **single
     full-frame forward** (blur transport is already one Gaussian); FG via
     existing `composite_passthrough`
4. **Register** in `presley_ai.py`:
   ```python
   'nafnet': ('blur',) + INPAINT_DEGRADATIONS
   _STRENGTH_CLAMP['nafnet'] = 8
   ```
5. **Tests:** extend `tests/test_presley_ai_restorer_dispatch.py`; stub
   contract test (no real weights in CI).
6. **Q5 experiments** (fixed-QP only; filter `codec=svtav1`):
   - Same recipe as InstantIR CLAIM: bear qp51 / camel qp50, blur,
     `sa=0.25`, `fg_protect`, compare to unsharp + InstantIR-corrected hashes
   - Dry-run → run → `presley-compare --region background` → fold rows into
     `HOLE(tab:instantir-kill)`

### Wave 2 — Real-HAT-GAN (Q4)

1. Download `Real_HAT_GAN_sharper.pth` from `Acly/hat` → `weights/`.
2. Integrate `restorer: real_hat_gan` on **downsample** (mirror BSRGAN /
   Real-ESRGAN path; HAT arch — check whether a small vendor or existing
   package fits torch 2.1.2 without upgrading diffusers).
3. Few fixed-QP cells vs Real-ESRGAN (`e2cb6bed165d69b1` / `c29c94b5c290f208`).

### Wave 3 — conditional / later

- **Q6 DiffBIR** only if NAFNet still loses to unsharp *and* InstantIR.
- **Q7/Q8** Stream-DiffVSR / DC-VSR when downsample-diffusion story is needed.
- **Q9** noise rematch; **Q10** mask dilate/erode (`HOLE(sec:evaluation)`).
- Paper write-ups already queued (D6 tables) — data largely exists.

### Skipped wave split rationale

Wave 1 then 2 is sequential (shared `restoration.py` / `presley_ai.py`). Do
not parallelize two restorers in one checkout without worktrees.

## Open questions

1. **Overleaf push** of `c67b3f6` (`tab:instantir-kill`) — human call.
2. **Commit code-side InstantIR steps + tests + yaml** — human call (left
   uncommitted this session).
3. NAFNet: single-pass only for v1, or InstantIR-parity multi-round ablation?
   Default: **single-pass**.

## Ops gotchas (do not rediscover)

- Filter fixed-QP InstantIR/BSRGAN/NAFNet with **`codec=svtav1`**. Bare
  `restorer=instantir` rematches incomplete **VBR** dirs (e.g.
  `results/287d46b17dd07d54/` — leave alone).
- Bool filters: `fg_protect=True` (capital T); `true` matches nothing.
- Duplicate InstantIR launches can overwrite `result.json` mid-eval.
- BG-PSNR is never the Goal-2 verdict; InstantIR's ~2 dB crater is
  corroboration only.
- Blur **frees bits** under fixed QP — InstantIR Goal-2 failure ≠ blur
  method failure.

## Immediate next steps (ordered)

1. Commit or stash the uncommitted InstantIR-steps / yaml / docs changes in
   the code repo (ask human if unsure).
2. Download `NAFNet-GoPro-width64.pth` into `weights/`.
3. Implement Wave 1 (arch → restore → register → tests → dry-run Q5).
4. Run Q5 bear first; compare to unsharp / InstantIR-corrected; extend
   `tab:instantir-kill` via `/update-paper` (no Overleaf push unless asked).
5. Then Wave 2 Real-HAT-GAN.

## Landmarks

| Path | Why |
|---|---|
| `docs/EXPERIMENTS_QUEUED.md` | Queue + restorer matrix + InstantIR audit |
| `68e8b6bb11d0dd9e62a67aef/sections/evaluation.tex` | `tab:instantir-kill`, `HOLE` for NAFNet rows |
| `68e8b6bb11d0dd9e62a67aef/RESEARCH_LOG.md` | Dead-ends / (a-KILL) |
| `src/presley/components/presley_ai.py` | `RESTORER_DEGRADATIONS` dispatch |
| `src/presley/restoration.py` | InstantIR / BSRGAN / (future NAFNet) |
| `tests/test_presley_ai_restorer_dispatch.py` | Fast registration tests |
| `.claude/skills/run-experiment/SKILL.md` | How to add/run yaml cells |
| `.claude/skills/update-paper/SKILL.md` | Fold Q5 into paper markers |
