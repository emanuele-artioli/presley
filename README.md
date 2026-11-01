# PRESLEY

Research implementation of **Perceptual Refinement of an End-to-End Video
Streaming Pipeline via Generative AI Layers**, by Emanuele Artioli, Farzad
Tashtarian, and Christian Timmerer. PRESLEY extends ELVIS with in-place
background degradation, a bit-packed strength side channel, and restoration
conditioned on transmitted pixels.

The evaluated deployment is offline delivery: the server prepares a standard
bitstream plus side channel, caches serve those files, and the client decodes
and restores before playback. Measured rates include both payloads. Improved
quality at matched QP is not itself evidence of bitrate savings at equal quality.

## Manuscript figure inputs (CPU only)

[paper-artifact/](paper-artifact/) contains archived numerical figure inputs,
reference plots, checksums, and provenance. It supports redrawing eight
numerical figures without videos, model checkpoints, or a GPU:

```bash
python -m venv .venv-figures
. .venv-figures/bin/activate
pip install -r paper-artifact/requirements-figures.txt
python tools/reproduce_paper_figures.py --verify-only
python tools/reproduce_paper_figures.py --out /tmp/presley-figures-new
```

The output directory must be new. The command leaves the manuscript, archived
inputs, and original results untouched. This redraws archived measurements;
it does not rerun the experiments or replace the submitted figure PDFs.

## Experiment environment

The full pipeline uses Linux, Python 3.10, CUDA 12.1, and PyTorch 2.1.2.
Use a separate environment from the lightweight figure workflow:

```bash
conda env create -f environment.yaml
conda activate presley
./install_openmmlab.sh
```

`environment.yaml` installs this project in editable mode. The helper installs
OpenMMLab and the forked model packages. It requires network access and native
build tools. Keep the pinned torch, torchvision, transformers, diffusers and
einops versions in `environment.yaml`/`pyproject.toml`; do not upgrade them to
resolve a model installation problem. The helper's upstream Git dependencies
are development installs; it is not a complete historical environment lock.

Install FFmpeg/ffprobe with libx265 support. The Kvazaar and SVT-AV1 comparisons
also require the corresponding encoder binaries on PATH. Restoration requires
the selected model's upstream checkpoint. Real-ESRGAN and several other
backbones resolve/download their own weights; HAT, NAFNet and InstantIR also
support local weights as documented in `src/presley/restoration.py`. Weights
and third-party datasets are not bundled here.

## Frames and foreground masks

`dataset/<video>/` contains sequentially named original JPEG/PNG frames. Ground
truth annotations use `dataset/annotations/<video>/*.png`, with matching frame
counts and filenames. Any nonzero instance ID is foreground. For example:

```text
dataset/bear/00000.jpg
dataset/annotations/bear/00000.png
dataset/mosev2/<clip>/00001.jpg
dataset/annotations/mosev2/<clip>/00001.png
dataset/youtube_vos/<clip>/00000.jpg
dataset/annotations/youtube_vos/<clip>/00000.png
```

The experiments include DAVIS, MOSEv2 and YouTube-VOS. Preserve the original
frame sampling and the dataset namespace used by each configuration; duration
handling is defined in `src/presley/preprocessing.py`. For detector masks,
install the optional YOLOE dependencies using the controlled commands in
`pyproject.toml`, and set `DEFAULT_YOLO_MODEL_PATH` in
`src/presley/preprocessing.py` to your local YOLOE-11l-seg checkpoint. The
current runner uses that default; it does not expose a YAML checkpoint option.

## Run and evaluate

Start with a filtered configuration and a dry run:

```bash
presley-run experiments.yaml --filter component=baselines --filter video=bear --dry-run
presley-run experiments.yaml --filter component=baselines --filter video=bear
presley-evaluate results/
```

The runner hashes configurations and skips completed results. Keep original
results and caches; place new campaigns separately. Inspect each result's
`invariant_failures`, not just the runner's exit status. Degradation comparisons
use fixed QP/CRF; compare actual transmitted bitrate, including side channels.
The `config/` files provide bounded campaign recipes; `experiments.yaml` retains
the broader research history, including retired configurations.

Supported families include codec baselines/ROI, ELVIS removal with inpainting,
and PRESLEY downsampling with conditioned super-resolution. Noise injection and
pixel-plane QP mapping are retained research candidates, not recommended modes.
Blur and InstantIR are distinct from noise injection.

## Reporting and checks

`presley-compare` reports foreground/background effect sizes, declared reporting
margins, and paired significance. Those margins are not calibrated human
just-noticeable differences. Use `reporting_verdict` and `margin_notice` when
interpreting JSON; historical `JND` names and verdict codes remain for analysis
compatibility. A nonsignificant difference does not establish equivalence.

```bash
pytest                      # fast CPU tier; excludes GPU/data/invariant tiers
python tools/sync_agent_rules.py --check
python tools/check_architecture.py
```

The full test tier needs the dependencies listed in `.github/workflows/ci.yml`.
Use `CITATION.cff` for citation metadata. This checkout does not assign a new
project-wide license; upstream code, checkpoints and datasets retain their own
licensing terms.
