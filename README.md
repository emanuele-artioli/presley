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

## Usage

PRESLEY provides two main console scripts installed automatically via `pip` (defined in `pyproject.toml`):

### 1. Run a single pipeline test
Run the main script to process a test video:
```bash
presley
```
You can optionally provide a configuration file or command line arguments:
```bash
presley --reference-video /path/to/video.mp4 --block-size 16 --target-bitrate 1500000
```

### 2. Search Modes (Grid and Random Search)
You can run automated parameter searches:
```bash
# Run a grid search across defined parameters
presley-search --mode grid

# Run a random search over the grid space (defaults to 10 samples)
presley-search --mode random --samples 20
```

## Methods Implemented

- **Baseline**: Standard H.265 encoding
- **Adaptive ROI**: ROI-based quantization using removability scores
- **ELVIS v1**: Block removal with inpainting restoration (OpenCV Telea, ProPainter, E2FGVI)
- **PRESLEY Downsample**: Adaptive downsampling with super-resolution (Real-ESRGAN, Upscale-A-Video)
- **PRESLEY Blur**: Controlled noise injection with deblurring (InstantIR)
