"""
Pre-generate YOLO masks for DAVIS mask-sensitivity videos.

Run inside the sole `presley` conda env (ultralytics installed via the
optional `[yolo]` extra — see pyproject.toml / AGENTS.md Environment):

    conda run -n presley --no-capture-output python tools/generate_yolo_masks.py

Once cache/<video>_640x360/yolo_masks/ is populated, `presley-run` with
`mask_source: yolo` only reads the PNGs.
"""
import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

CACHE_DIR = "/home/itec/emanuele/presley/cache"
MODEL_PATH = "/home/itec/emanuele/Models/YOLO/yoloe-11l-seg.pt"
VIDEOS = ["bear", "bmx-trees", "camel", "dog", "india", "pigs", "tennis"]
WIDTH, HEIGHT = 640, 360
PROMPTS = ["person", "animal", "vehicle", "object"]
CONF = 0.25


def generate_masks_for_video(video_name: str, model):
    key_dir = os.path.join(CACHE_DIR, f"{video_name}_{WIDTH}x{HEIGHT}")
    ref_dir = os.path.join(key_dir, "reference_frames")
    yolo_dir = os.path.join(key_dir, "yolo_masks")

    frame_files = sorted(Path(ref_dir).glob("*.png"))
    if not frame_files:
        print(f"  [SKIP] No reference frames at {ref_dir}", flush=True)
        return 0

    existing = list(Path(yolo_dir).glob("*.png")) if os.path.exists(yolo_dir) else []
    if len(existing) == len(frame_files):
        print(f"  [SKIP] Already cached ({len(existing)} frames)", flush=True)
        return len(existing)

    os.makedirs(yolo_dir, exist_ok=True)
    print(f"  Generating {len(frame_files)} frames → {yolo_dir}", flush=True)

    t0 = time.time()
    for i, fpath in enumerate(frame_files):
        img = cv2.imread(str(fpath))
        result = model.predict(img, conf=CONF, verbose=False)[0]
        fg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if result.masks is not None:
            for m in result.masks.data.cpu().numpy():
                m_resized = cv2.resize(m, (img.shape[1], img.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                fg = np.maximum(fg, (m_resized > 0.5).astype(np.uint8))
        cv2.imwrite(os.path.join(yolo_dir, fpath.name), fg * 255)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    frame {i+1}/{len(frame_files)}  ({elapsed:.1f}s elapsed)", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s", flush=True)
    return len(frame_files)


def main():
    import torch
    from ultralytics import YOLOE

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found: {MODEL_PATH}")

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLOE from {MODEL_PATH} on {device_str}", flush=True)
    model = YOLOE(MODEL_PATH)
    model.to(device_str)
    model.set_classes(PROMPTS, model.get_text_pe(PROMPTS))
    print(f"Model loaded. Prompts: {PROMPTS}\n", flush=True)

    for video in VIDEOS:
        print(f"[{video}]", flush=True)
        generate_masks_for_video(video, model)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
