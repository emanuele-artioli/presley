#!/bin/bash
set -e

echo "Installing OpenMMLab dependencies via MIM..."

# Fix for openmim complaining about pkg_resources
pip install "setuptools<70"

# mmcv-full has native extensions that must match your CUDA / torch installation.
# Run these commands AFTER creating the conda environment and installing the pyproject.toml dependencies.
python3 -m mim install mmcv-full==1.7.2 mmdet==2.28.2

# mmpose is installed with --no-deps to avoid mmtrack/mmpycocotools build issues
pip install mmpose==0.29.0 --no-deps

# install mmpretrain
python3 -m mim install mmpretrain

echo "OpenMMLab dependencies installed successfully."

echo "Installing GitHub dependencies..."
pip install "evca @ git+https://github.com/emanuele-artioli/EVCA.git"
pip install "ufo @ git+https://github.com/emanuele-artioli/UFO.git"
pip install "propainter @ git+https://github.com/emanuele-artioli/ProPainter.git"
pip install --no-deps "e2fgvi @ git+https://github.com/emanuele-artioli/E2FGVI.git"
pip install "realesrgan @ git+https://github.com/emanuele-artioli/Real-ESRGAN.git"
pip install "instantir @ git+https://github.com/emanuele-artioli/InstantIR.git"
pip install "upscale-a-video @ git+https://github.com/emanuele-artioli/Upscale-A-Video.git"

# Re-enforce the correct einops version required by fvmd and presley
pip install einops==0.6.1

echo "GitHub dependencies installed successfully."
