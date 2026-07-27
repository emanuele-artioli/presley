# Handoff — sole-env YOLO / invariants / restorer queue (2026-07-27)

## Session closable?

**Yes for this plan's scope.** YOLO works in `presley`, invariants refresh+test
green, BSRGAN.pth on disk, InstantIR audit + restorer matrix + Q1–Q10 written,
code pushed (or ready to push with this commit). **Do not run Q1–Q10 here** —
next session owns GPU integrations and experiments.

## Where to start next

1. Read [`docs/EXPERIMENTS_QUEUED.md`](docs/EXPERIMENTS_QUEUED.md) (authoritative
   queue + InstantIR audit + restorer matrix).
2. Grep paper `HOLE()` before commissioning runs.
3. Suggested order: **Q1** (LPIPS backfill, cheap) → **Q2** InstantIR corrected
   smoke → **Q3** BSRGAN few cells → integrate **NAFNet** (Q5) / **Real-HAT-GAN**
   (Q4) → conditional DiffBIR (Q6) → Stream-DiffVSR/DC-VSR (Q7/Q8).

## Done this session

| Item | Evidence |
|---|---|
| YOLO in sole `presley` env | `ultralytics==8.4.104` via `--no-deps` + thop/polars/nvidia-ml-py; torch still `2.1.2+cu121`. Optional `[yolo]` in `pyproject.toml`. Smoke: `from ultralytics import YOLOE`; `get_yolo_masks('bear',…)` cache-hit 82 frames. |
| Docs | `AGENTS.md` Environment exception; `tools/generate_yolo_masks.py`; `get_yolo_masks` docstring. |
| Sticky `invariant_failures` | `evaluation/run.py` re-runs `check_result` after metrics (and on early-return refresh); `runner.main` `backfill(..., force=True)` after `evaluate_all`. Regression: `test_sticky_pre_metrics_failure_clears_once_metrics_exist`. |
| Remaining uncitable results | 6 legitimate (4 VBR + 2 InstantIR Goal-2 LPIPS worse than transmitted) — **not** sticky metrics-missing. |
| BSRGAN weights | `/home/itec/emanuele/presley/weights/BSRGAN.pth` (~64 MB, sha256 `5d505a07…`). Search path includes `cwd/weights/` when run from repo root. Gitignored `*.pth`. |
| Noise threshold fix | `filter_frame_noise` uses `round(score)>0` like blur; `tests/test_degradation_noise_threshold.py`. Rematch is **Q9**. |
| Queue / matrix / InstantIR audit | All in `docs/EXPERIMENTS_QUEUED.md`. |

## Locked scientific facts (do not reverse casually)

- **Blur frees bits** under fixed QP after S1 budget match (like downsample). **Noise costs bits.** Do not retire blur because InstantIR failed Goal 2.
- Keep **Real-ESRGAN**; **BSRGAN** = few fixed-QP cells only.
- Prefer **Real-HAT-GAN** as the recent SR GAN (not a full BSRGAN campaign).
- Blur gauge: **NAFNet** (CNN — do not call it a GAN). Second diffusion only if corrected InstantIR still loses: **DiffBIR**.
- InstantIR defaults are likely **misconfigured** (`steps=1`, `creative_start=1.0`, `preview_start=0.0`). Existing CLAIM hashes = as-run until Q2.

## Host / ops

- GPUs: use `presley.gpu_utils.pick_gpu` / `preflight_gpu`. Servers are gpu5/gpu6 — not device indices. Check `nvidia-smi` before long runs.
- Long jobs: experiment-runner or background shell; never hand-rolled `pgrep` wait loops.
- **No Overleaf push** unless human asks. Paper repo is separate under `68e8b6bb11d0dd9e62a67aef/` (gitignored here).
- Do not wholesale `rm` `results/` / `cache/` / `dataset/`.

## Out of scope / next session only

- Implementing Real-HAT-GAN, NAFNet, DiffBIR, Stream-DiffVSR, DC-VSR.
- Running any Q* GPU batch.
- Paper/Overleaf write-up of D6 tables (data mostly exists; write-up queued in EXPERIMENTS_QUEUED).

## YOLO install (if another machine needs it)

```bash
conda run -n presley pip install 'ultralytics==8.4.104' --no-deps
conda run -n presley pip install ultralytics-thop polars 'nvidia-ml-py>=12'
```

Checkpoint: `/home/itec/emanuele/Models/YOLO/yoloe-11l-seg.pt`
