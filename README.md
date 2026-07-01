# PRESLEY - Perceptual Refinement of an End-to-End Video Streaming Pipeline via Generative AI Layers

This repository implements PRESLEY, a unified adaptive video compression framework that includes legacy methods (ELVIS) and new adaptive degradation and restorative generative AI methods.

## Setup Instructions

We use `conda` and `pyproject.toml` to manage the environment.

1. **Create the Conda environment**:
   This installs heavy system-level ML binaries (PyTorch with CUDA support).
   ```bash
   conda env create -f environment.yaml
   conda activate presley
   ```

2. **Install OpenMMLab dependencies**:
   ```bash
   chmod +x install_openmmlab.sh
   ./install_openmmlab.sh
   ```

## Dataset

The `dataset/` directory contains symlinks to DAVIS video frame directories.
Each subdirectory contains sequentially-numbered JPEG frames:

    dataset/
    ├── bear/
    │   ├── 00000.jpg
    │   ├── 00001.jpg
    │   └── ...
    ├── camel/
    └── ...

To set up the dataset, symlink your DAVIS frame directories:

    mkdir -p dataset
    ln -sfn /path/to/DAVIS/bear dataset/bear

The pipeline reads frames directly from these directories as ground truth (no compression artifacts). Videos are resized to the experiment's target resolution at preprocessing time.

## Usage

PRESLEY provides a modular component architecture driven by YAML configuration.

### 1. Run experiments
Define experiments in `experiments.yaml` and run them:
```bash
presley-run experiments.yaml
```
The runner will skip any experiments that have already been computed.
You can filter which experiments to run:
```bash
presley-run experiments.yaml --filter component=baselines --filter video=bear
```

### 2. Evaluate results
The runner automatically calls the evaluation component, but you can also re-evaluate manually:
```bash
presley-evaluate results/
```

## Methods Implemented

- **Baseline**: Standard H.265 encoding
- **Adaptive ROI**: ROI-based quantization using removability scores
- **ELVIS v1**: Block removal with inpainting restoration (OpenCV Telea, ProPainter, E2FGVI)
- **PRESLEY Downsample**: Adaptive downsampling with super-resolution (Real-ESRGAN, Upscale-A-Video)
- **PRESLEY Blur**: Controlled noise injection with deblurring (InstantIR)
