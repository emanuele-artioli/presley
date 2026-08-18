# PRESLEY architecture

A map of what lives where, and how a run flows through it. CI checks that every
module under `src/presley/` appears here, so this cannot quietly go stale.

## Dataflow

```
experiments.yaml
   │  each entry hashed (runner.compute_experiment_hash) -> results/<hash>/
   ▼
runner.dispatch_component  ──►  components/{baselines,roi,elvis,presley_ai,
                                              probe_oracle_bits,probe_block_damage}
   │                                    │
   │                          preprocessing: reference frames, EVCA scores, UFO masks
   │                                    │
   │                          degradation (QP map / downsample / blur / freeze)
   │                                    │
   │                          encode_utils: x265 / x264 / SVT-AV1 / kvazaar
   │                                    │
   │                          restoration (Real-ESRGAN, InstantIR, ProPainter, E2FGVI)
   │                                    ▼
   │                          result.json  +  sidechannel strength maps
   ▼
evaluation/  ──►  metrics per region  ──►  invariants.check_result
                                                │
                                    invariant_failures written into result.json
                                                │
                             compare (JND gate)  ──►  the paper's CLAIM lines
```

## Modules

### Orchestration
| Module | Responsibility |
|---|---|
| `runner.py` | Hashes each `experiments.yaml` entry, skips completed ones, dispatches via `COMPONENT_RUNNERS`, records the invariant verdict. |
| `concurrency.py` | Parallel execution helpers for multi-experiment runs. |
| `gpu_utils.py` | Picks the least-loaded GPU and sizes VRAM-sensitive batches before the component's heavy imports. |

### Components — one per method under test
| Module | Responsibility |
|---|---|
| `components/baselines.py` | Straight encodes, the comparison target for everything else. |
| `components/roi.py` | Codec-native ROI (kvazaar, x265 AQ, SVT-AV1) and the presley_* pixel degradations. |
| `components/elvis.py` | Block removal plus client-side in-painting (the NOSSDAV method). |
| `components/presley_ai.py` | Mask-driven degradation with generative restoration. |
| `components/probe_oracle_bits.py` | F1 measurement probe, not a transport: leave-one-superblock-out marginal bit cost vs the EVCA proxy, under inter coding. Runs through the runner only so its numbers carry a `results/<hash>`. |
| `components/probe_block_damage.py` | Claim-(b) measurement probe: within-run dispersion of per-superblock damage after restoration, against a pristine encode made inside the run. Wraps `run_presley_ai` rather than reimplementing it, so unlike the other probe it **does** emit a restored video — never pool it as a transport arm. |

### Pipeline stages
| Module | Responsibility |
|---|---|
| `preprocessing.py` | Reference frames, EVCA complexity scores, UFO foreground masks. |
| `degradation.py` | Server-side degradations: QP mapping, downsampling, blur, noise, freeze, blackout. |
| `restoration.py` | Client-side restoration model wrappers. |
| `encode_utils.py` | Encoder invocations, rate-control derivation, QP-offset mapping, decode. |
| `sidechannel.py` | Packs the strength maps transmitted alongside the video. |
| `blockdamage.py` | 64x64 superblock geometry for the damage denominator: area-exact pooling, MSE-space (never PSNR) averaging. Shared by `components/probe_block_damage.py` and `tools/mine_block_damage.py` so the two cannot drift apart on where boundaries fall. |
| `io.py` | Frame and mask loading, directory hygiene. |
| `utils.py` | Small shared helpers. |
| `hnerv_arch.py`, `hnerv_utils.py` | HNeRV learned-codec baseline. |
| `hat_arch.py` | Vendored HAT architecture (XPixelGroup/HAT), inference only, float32 — backs the Real-HAT-GAN restorer. |
| `nafnet_arch.py` | Vendored NAFNet architecture (megvii-research/NAFNet), inference only. |
| `stream_diffvsr.py` | Stream-DiffVSR glue: subprocess inference against an isolated vendor checkout and env. |
| `dc_vsr.py` | DC-VSR weight layout plus the gate that refuses inference (HF repo ships UNet EMA weights only). |

### Evaluation
| Module | Responsibility |
|---|---|
| `evaluation/masked.py` | Region-restricted PSNR/MSE/SSIM; foreground bounding boxes. |
| `evaluation/perceptual.py` | LPIPS, DISTS, FID — the metrics carrying the quality claim. |
| `evaluation/vmaf.py` | VMAF and the yuv420 writing it needs. |
| `evaluation/fvmd.py` | Frechet Video Motion Distance; reported, never gating. |
| `evaluation/cache.py` | Memoized reference frames and masks — the real cost of a pass. |
| `evaluation/run.py` | The evaluation pass over one experiment or the whole tree. |
| `evaluation/backfill.py` | In-place metric backfills; no re-encoding, re-entrant. |
| `evaluation/reports.py` | Validity diagnostics for FVMD and FID. |
| `evaluation/cli.py` | The `presley-evaluate` entry point. |
| `components/evaluation.py` | Shim re-exporting the package; keeps the installed console script working. |

### Claims
| Module | Responsibility |
|---|---|
| `compare.py` | The JND gate. Decides whether a quality difference is real, and enforces which keys may back a foreground claim. |
| `suite.py` | The suite significance layer on top of `compare.py`: exact sign/Wilcoxon tests, bootstrap CI, effect size and Holm correction over N>1 paired runs. Adds the `sub_jnd_significant` verdict; never overrides a JND call, never promotes a sub-JND effect to a perceptual win. |
| `db.py` | The results store, and the source of truth for run metadata. Holds each run as a canonical document plus derived query rows (narrow `metrics`, `artifacts`), and encodes the citability rules as SQL views — `v_citable` (empty `invariant_failures` and evaluated), `v_fg_metrics` (the union-bbox metrics are absent, not merely discouraged), `v_rate`. `result.json` is a derived mirror; either side rebuilds the other. |
| `invariants.py` | The methodology rules as code: fixed-QP mandate, bitrate accounting, restoration not regressing. Writes `invariant_failures` into each result. |

## Contracts worth knowing before changing anything

- **The experiment hash is the identity of every result on disk.** It is a
  sha256 of the config dict alone, excluding `hash` and `_`-prefixed keys — so
  refactoring code cannot orphan a `results/<hash>/` directory, but changing
  what goes *into* an entry will. Never reformat `experiments.yaml` wholesale;
  the `# hash:` provenance comments are position-sensitive.
- **A result with a non-empty `invariant_failures` is not citable.** Both
  `/results-report` and `/update-paper` refuse it.
- **Foreground metrics come only from true masked keys.** `compare.BANNED_FG_KEYS`
  lists the union-bbox artifacts that must never back an FG claim.
- **An all-false mask makes the masked metrics score the whole frame.** So an
  absent mask yields a plausible "foreground" number rather than an error.
- **Degradation experiments must be fixed-QP/CRF.** Under a bitrate target the
  encoder spends it regardless of source complexity, which inverts Goal 1.
  `invariants._check_fixed_qp_mandate` enforces this.
