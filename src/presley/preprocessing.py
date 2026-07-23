import os
import cv2
import sys
import shutil
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalizes a NumPy array to the range [0, 1]."""
    min_val, max_val = arr.min(), arr.max()
    return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr


# Per-dataset native frame rates for pre-extracted frame-sequence sources
# (DAVIS/MOSEv2/YouTube-VOS all ship as directories of still images with no
# embedded timing metadata, so there is no way to *measure* their rate the
# way ffprobe measures an actual video file -- it has to come from each
# dataset's own documentation). Do not add a blanket default here: a wrong
# per-dataset assumption silently corrupts every downstream duration/bitrate
# computation (`duration = len(frames) / framerate` in the components).
#
#   davis        -- 24 fps. Perazzi et al., "A Benchmark Dataset and
#                    Evaluation Methodology for Video Object Segmentation",
#                    CVPR 2016 (confirmed, not just carried over from the old
#                    hardcoded default -- see RESEARCH_LOG / D4 for the check).
#   youtube_vos  -- 6 fps. YouTube-VOS releases frames pre-sampled every 5th
#                    frame of a 30fps source (Xu et al., "YouTube-VOS:
#                    Sequence-to-Sequence Video Object Segmentation", ECCV
#                    2018) -- confirmed against the actual downloaded frame
#                    filenames (00000, 00005, 00010, ...). This is the
#                    released rate, not the original video's rate.
#   mosev2       -- 30.0 is a JUDGMENT CALL, not a documented value: neither
#                    the MOSEv2 paper (arXiv:2508.05630) nor mose.video state
#                    an fps, and the released frame indices are NOT evenly
#                    spaced (e.g. clip fii86rku starts 00001, 00003, 00004,
#                    00005, 00006, 00007, 00010, 00015, ...) -- consistent
#                    with filtering near-duplicate/low-quality frames out of
#                    a ~30fps web-video source rather than fixed-interval
#                    resampling. Because gaps are irregular, `duration =
#                    len(frames) / framerate` (used throughout the
#                    components) is only approximate for MOSEv2 regardless of
#                    which constant is used here -- do not cite MOSEv2
#                    bitrate/duration numbers as precise without re-deriving
#                    real inter-frame spacing (not available from this
#                    release) first. See dataset/mosev2/PROVENANCE.md.
KNOWN_DATASET_FRAMERATES = {
    "davis": 24.0,
    "youtube_vos": 6.0,
    "mosev2": 30.0,
}

_VIDEO_FILE_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")


def probe_framerate(video_name: str, dataset_dir: str, dataset: str = None) -> float:
    """
    Determine the true frame rate for `video_name`'s source instead of
    blanket-assuming 24fps for every dataset (the previous behavior here,
    which happens to be correct for DAVIS but was never actually verified
    against DAVIS's own documentation, and is wrong for other datasets).

    Resolution order:
      1. If `dataset_dir/video_name` contains an actual video file
         (.mp4/.mkv/.webm/.avi/.mov/.m4v), `ffprobe` its real stream
         framerate -- the only source of truth when one exists.
      2. Otherwise (a pre-extracted frame directory, which is how DAVIS,
         MOSEv2 and YouTube-VOS are all ingested here), fall back to
         `KNOWN_DATASET_FRAMERATES[dataset]` -- a per-dataset documented
         value, not a single constant reused across datasets.
      3. If `dataset` isn't given explicitly, infer it from `video_name`'s
         first path segment (e.g. "mosev2/xyz" -> "mosev2"), defaulting to
         "davis" for a bare name -- this project's existing flat DAVIS
         videos (`dataset/bear`, `dataset/tennis`, ...) have no namespace
         prefix, so this keeps every existing experiment resolving to
         exactly the same 24.0 it always used.

    Raises ValueError rather than silently guessing if none of the above
    resolves.
    """
    source_dir = os.path.join(dataset_dir, video_name)
    video_files = []
    if os.path.isdir(source_dir):
        for ext in _VIDEO_FILE_EXTS:
            video_files.extend(sorted(Path(source_dir).glob(f"*{ext}")))

    if video_files:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video_files[0])],
            capture_output=True, text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            num_str, _, den_str = probe.stdout.strip().partition('/')
            try:
                num, den = float(num_str), float(den_str) if den_str else 1.0
                if den > 0:
                    return num / den
            except ValueError:
                pass
        raise RuntimeError(
            f"ffprobe failed to determine framerate for {video_files[0]}: "
            f"{probe.stderr.strip() or probe.stdout.strip()}"
        )

    if dataset is None:
        dataset = video_name.split('/', 1)[0] if '/' in video_name else 'davis'
    fps = KNOWN_DATASET_FRAMERATES.get(dataset)
    if fps is None:
        raise ValueError(
            f"No known framerate for dataset '{dataset}' (video '{video_name}') and no "
            f"video file found in {source_dir} to ffprobe. Add an entry to "
            f"KNOWN_DATASET_FRAMERATES in preprocessing.py or pass `dataset` explicitly."
        )
    return fps


def get_reference_frames(video_name: str, width: int, height: int, dataset_dir: str, cache_dir: str,
                         dataset: str = None):
    """
    Returns (raw_yuv_path, frames_list, framerate).
    Caches extracted frames and YUV at target resolution.

    `dataset` selects which entry of KNOWN_DATASET_FRAMERATES applies (see
    `probe_framerate`); leave it None to infer from `video_name`'s namespace
    prefix (existing bare DAVIS names -> 'davis', unchanged behavior).
    """
    framerate = probe_framerate(video_name, dataset_dir, dataset)

    key_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}")
    os.makedirs(key_dir, exist_ok=True)
    
    raw_yuv_path = os.path.join(key_dir, "reference_raw.yuv")
    ref_frames_dir = os.path.join(key_dir, "reference_frames")
    
    if os.path.exists(raw_yuv_path) and os.path.exists(ref_frames_dir):
        # Already cached
        frames = []
        for p in sorted(Path(ref_frames_dir).glob("*.png")):
            frames.append(cv2.imread(str(p), cv2.IMREAD_COLOR))
        if frames:
            return raw_yuv_path, frames, framerate

    # Not cached, build it from dataset
    source_dir = os.path.join(dataset_dir, video_name)
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source video frames not found at {source_dir}")
        
    os.makedirs(ref_frames_dir, exist_ok=True)
    frames = []
    
    src_paths = sorted(Path(source_dir).glob("*.jpg"))
    if not src_paths:
        src_paths = sorted(Path(source_dir).glob("*.png"))
        
    for i, p in enumerate(src_paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None: continue
        
        # Resize to target
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
        dst_path = os.path.join(ref_frames_dir, f"{i+1:05d}.png")
        cv2.imwrite(dst_path, img)
        frames.append(img)
        
    # Generate YUV (lossless equivalent of the resized frames for EVCA to use)
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(framerate),
        "-i", os.path.join(ref_frames_dir, "%05d.png"),
        "-pix_fmt", "yuv420p", raw_yuv_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
    return raw_yuv_path, frames, framerate


def get_evca_scores(video_name: str, width: int, height: int, block_size: int,
                    raw_yuv_path: str, reference_frames_dir: str, cache_dir: str):
    """
    Computes EVCA temporal and spatial complexity scores.
    Returns (temporal_3d, spatial_3d).
    """
    key_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}_bs{block_size}")
    os.makedirs(key_dir, exist_ok=True)
    
    evca_tc_dest = os.path.join(key_dir, "evca_TC_blocks.csv")
    evca_sc_dest = os.path.join(key_dir, "evca_SC_blocks.csv")
    
    frame_count = len(list(Path(reference_frames_dir).glob("*.png")))
    
    if not (os.path.exists(evca_tc_dest) and os.path.exists(evca_sc_dest)):
        try:
            import importlib
            evca_pkg = importlib.import_module('evca')
        except ImportError as exc:
            raise RuntimeError("The 'evca' package is not installed.") from exc
            
        evca_root = Path(evca_pkg.__file__).resolve().parent
        package_tc = evca_root / 'evca_TC_blocks.csv'
        package_sc = evca_root / 'evca_SC_blocks.csv'
        
        for p in (package_tc, package_sc, evca_root / 'evca.csv'):
            if p.exists():
                try: p.unlink()
                except: pass
                
        evca_cmd = [
            sys.executable, '-m', 'evca.main', 
            '-i', os.path.abspath(raw_yuv_path), 
            '-r', f'{width}x{height}', 
            '-b', str(block_size), 
            '-f', str(frame_count), 
            '-c', os.path.join(os.path.abspath(key_dir), 'evca.csv'), 
            '-bi', '1'
        ]
        result = subprocess.run(evca_cmd, capture_output=True, text=True, cwd=str(key_dir))
        if result.returncode != 0:
            raise RuntimeError(f"EVCA execution failed: {result.stderr}\n{result.stdout}")
            
        # EVCA writes directly to cwd, which is key_dir, so the files are already there
        
    temporal_array = np.loadtxt(evca_tc_dest, delimiter=',', skiprows=1)
    spatial_array = np.loadtxt(evca_sc_dest, delimiter=',', skiprows=1)
    
    num_blocks_x = width // block_size
    num_blocks_y = height // block_size
    num_frames = min(temporal_array.shape[1], spatial_array.shape[1])
    
    temporal_3d = temporal_array[:, :num_frames].T.reshape(num_frames, num_blocks_y, num_blocks_x)
    spatial_3d = spatial_array[:, :num_frames].T.reshape(num_frames, num_blocks_y, num_blocks_x)
    
    temporal_3d = normalize_array(temporal_3d)
    spatial_3d = normalize_array(spatial_3d)
    
    return temporal_3d, spatial_3d


def get_ufo_masks(video_name: str, width: int, height: int, block_size: int,
                  reference_frames_dir: str, cache_dir: str, temporal_pool: bool = False):
    """
    Returns UFO masks as (F, H, W) array.
    """
    key_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}")
    ufo_masks_dir = os.path.join(key_dir, "ufo_masks")
    
    frame_files = sorted(Path(reference_frames_dir).glob("*.png"))
    
    if not os.path.exists(ufo_masks_dir) or len(list(Path(ufo_masks_dir).glob("*.png"))) != len(frame_files):
        os.makedirs(ufo_masks_dir, exist_ok=True)
        import torch
        from ufo.test import segment_frames
        
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        frames_list = []
        for fpath in frame_files:
            frames_list.append(cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB))
            
        frames_arr = np.array(frames_list)
        
        try:
            import importlib
            ufo_pkg = importlib.import_module('ufo')
            model_path_for_ufo = str(Path(ufo_pkg.__file__).parent / 'weights' / 'video_best.pth')
        except Exception:
            model_path_for_ufo = 'weights/video_best.pth'
            
        if not os.path.exists(model_path_for_ufo):
            # Try downloader
            try:
                downloader = importlib.import_module('ufo.download_ufo_weights')
                model_path_for_ufo = str(downloader.main())
            except Exception:
                pass
                
        masks_arr = segment_frames(
            frames=frames_arr,
            device=device_str,
            model_path=model_path_for_ufo,
            group_size=5,
            img_size=224
        )
        
        for i, fname in enumerate(frame_files):
            mask_path = os.path.join(ufo_masks_dir, fname.name)
            mask_uint8 = (masks_arr[i] * 255.0).astype(np.uint8)
            cv2.imwrite(mask_path, mask_uint8)
            
    # Load masks
    masks = []
    for p in sorted(Path(ufo_masks_dir).glob("*.png")):
        masks.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        
    masks_arr = np.array(masks)
    if temporal_pool:
        pooled = np.max(masks_arr, axis=0)
        masks_arr = np.repeat(pooled[None, ...], len(masks_arr), axis=0)

    return masks_arr


def get_gt_masks(video_name: str, width: int, height: int, block_size: int,
                 reference_frames_dir: str, cache_dir: str, annotations_dir: str):
    """
    Loads ground-truth foreground masks for `video_name` from a per-video
    directory of palette-indexed (or plain grayscale) PNG masks at
    `annotations_dir/<video_name>/*.png` -- the DAVIS
    `Annotations/<resolution>/<video>/*.png` convention, which MOSEv2 and
    YouTube-VOS's per-frame instance masks also both happen to use.

    Returns the exact same (F, H, W) uint8 contract as `get_ufo_masks`: one
    grayscale frame per reference frame, values in [0, 255]. Unlike UFO's
    continuous saliency output, this is genuinely binary (0 or 255) -- DAVIS/
    MOSEv2/YouTube-VOS annotations are multi-instance (pixel value = object
    id, 0 = background), and any nonzero pixel of *any* instance is
    collapsed to a single foreground blob here, matching how the rest of the
    pipeline already treats UFO's single saliency mask ("is there a
    foreground object here", not "which one" -- see `get_removability_scores`,
    which only ever tests `resized_mask == 0`).

    Frame counts must match `reference_frames_dir` exactly; a mismatch means
    `annotations_dir` doesn't correspond to the same clip/frame-rate as the
    reference frames and is raised rather than silently truncated.
    """
    key_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}")
    gt_masks_dir = os.path.join(key_dir, "gt_masks")

    frame_files = sorted(Path(reference_frames_dir).glob("*.png"))

    if not os.path.exists(gt_masks_dir) or len(list(Path(gt_masks_dir).glob("*.png"))) != len(frame_files):
        os.makedirs(gt_masks_dir, exist_ok=True)

        video_annot_dir = os.path.join(annotations_dir, video_name)
        if not os.path.isdir(video_annot_dir):
            raise FileNotFoundError(
                f"mask_source='gt' requires ground-truth annotations for '{video_name}' at "
                f"{video_annot_dir} (a folder of palette-indexed or grayscale PNG masks, one "
                f"per frame -- the DAVIS Annotations/<video>/*.png convention). None found."
            )
        annot_paths = sorted(Path(video_annot_dir).glob("*.png"))
        if not annot_paths:
            raise FileNotFoundError(f"{video_annot_dir} exists but contains no .png masks")
        if len(annot_paths) != len(frame_files):
            raise ValueError(
                f"GT annotation count ({len(annot_paths)}) does not match reference frame "
                f"count ({len(frame_files)}) for '{video_name}' -- check that annotations_dir "
                f"points at the same clip/frame subsampling as the reference frames."
            )

        for fname, apath in zip(frame_files, annot_paths):
            # PIL, not cv2: cv2.imread (any flag, including IMREAD_UNCHANGED)
            # silently EXPANDS a palette-indexed ("P" mode) PNG through its
            # color table into a 3-channel BGR image -- it never returns the
            # raw index values, contrary to what IMREAD_UNCHANGED's name
            # suggests. That turned every real DAVIS annotation into an
            # all-zero mask here (object id 1's palette color happened to
            # have a zero blue channel) until caught by testing against real
            # `dataset/annotations/bear` data rather than only synthetic
            # single-channel test PNGs (which don't exercise "P" mode and so
            # didn't reveal this). PIL's `Image.open` + `np.array(...)`
            # preserves the raw indices for "P" mode and is the correct way
            # to read this format.
            idx_map = np.array(Image.open(apath))
            if idx_map.ndim == 3:
                # A genuine multi-channel mask (not palette-indexed) -- any
                # nonzero channel counts as foreground evidence.
                idx_map = idx_map.any(axis=-1).astype(np.uint8)
            binary = (idx_map > 0).astype(np.uint8)
            if binary.shape != (height, width):
                binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_uint8 = binary * 255
            cv2.imwrite(os.path.join(gt_masks_dir, fname.name), mask_uint8)

    masks = []
    for p in sorted(Path(gt_masks_dir).glob("*.png")):
        masks.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))

    return np.array(masks)


# Open-vocabulary YOLOE checkpoints (NOT the closed-vocabulary yolo11/yolo26
# -seg models, which only detect COCO's 80 classes): DAVIS/MOSEv2/YouTube-VOS
# foreground subjects range far outside that (e.g. "jellyfish", "surfboard",
# "kangaroo"). Default picked from the checkpoints staged at
# /home/itec/emanuele/Models/YOLO/: the mid-size "-11l-seg" text-prompted
# variant (not a "-pf" prompt-free checkpoint) so the mask-sensitivity study
# actually exercises prompt design, which is the point of D3.
DEFAULT_YOLO_MODEL_PATH = "/home/itec/emanuele/Models/YOLO/yoloe-11l-seg.pt"

# Generic, concrete foreground-object nouns fed to YOLOE's text encoder via
# `model.set_classes(prompts, model.get_text_pe(prompts))`. A single abstract
# prompt like "main subject" or "foreground object" was considered and
# rejected: YOLOE's text embeddings are trained on concrete noun phrases from
# image-caption/grounding data (see the Ultralytics YOLOE docs' prompt
# examples, which are all concrete nouns), so an abstract prompt sits far
# from the training distribution and under-fires. This short list is a
# judgment call, not a tuned result -- it has not been validated against
# real detections (see the D3 report caveat: `ultralytics` isn't installed in
# any environment on this host, so this path is untested end-to-end).
DEFAULT_YOLO_PROMPTS = ["person", "animal", "vehicle", "object"]


def get_yolo_masks(video_name: str, width: int, height: int, block_size: int,
                   reference_frames_dir: str, cache_dir: str,
                   model_path: str = None, class_prompts=None, conf: float = 0.25):
    """
    Uses YOLOE's open-vocabulary instance segmentation to produce a
    saliency-style foreground mask per frame, matching `get_ufo_masks`' and
    `get_gt_masks`' (F, H, W) uint8 contract.

    `model_path` defaults to `DEFAULT_YOLO_MODEL_PATH` (a text-prompted
    "-seg" checkpoint). Pointing it at a "-pf" (prompt-free) checkpoint
    instead skips prompt design entirely -- YOLOE's prompt-free variants
    detect from a built-in ~4585-class vocabulary with no `set_classes` call
    -- which is the alternative default strategy documented for D3 if the
    concrete-noun prompt list here turns out not to generalize across
    datasets.

    All detected instance masks in a frame are unioned into one binary blob
    (thresholded at `conf`), matching UFO/GT's single "is there a foreground
    object here" semantic rather than per-instance segmentation.

    Requires the `ultralytics` package (YOLOE support needs a version that
    ships it, e.g. >=8.3), which is NOT installed in the `presley` conda env
    on this host and must not be pip-installed into it (pinned research
    env -- see CLAUDE.md's Environment section); install it into a separate
    conda env or venv, the same pattern already used for `nrmetrics`
    (`~/.venvs/nrmetrics`). This function has not been exercised end-to-end
    on this host for exactly that reason -- see the D3 report caveat.
    """
    key_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}")
    yolo_masks_dir = os.path.join(key_dir, "yolo_masks")

    frame_files = sorted(Path(reference_frames_dir).glob("*.png"))

    if not os.path.exists(yolo_masks_dir) or len(list(Path(yolo_masks_dir).glob("*.png"))) != len(frame_files):
        os.makedirs(yolo_masks_dir, exist_ok=True)
        import torch
        from ultralytics import YOLOE

        ckpt = model_path or DEFAULT_YOLO_MODEL_PATH
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"YOLO checkpoint not found at {ckpt}")

        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = YOLOE(ckpt)
        model.to(device_str)

        is_prompt_free = '-pf' in os.path.basename(ckpt)
        if not is_prompt_free:
            prompts = class_prompts or DEFAULT_YOLO_PROMPTS
            model.set_classes(prompts, model.get_text_pe(prompts))

        for fpath in frame_files:
            img = cv2.imread(str(fpath))
            result = model.predict(img, conf=conf, verbose=False)[0]
            fg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            if result.masks is not None:
                for m in result.masks.data.cpu().numpy():
                    m_resized = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    fg = np.maximum(fg, (m_resized > 0.5).astype(np.uint8))
            cv2.imwrite(os.path.join(yolo_masks_dir, fpath.name), fg * 255)

    masks = []
    for p in sorted(Path(yolo_masks_dir).glob("*.png")):
        masks.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))

    return np.array(masks)


def resolve_masks(mask_source: str, video_name: str, width: int, height: int, block_size: int,
                  reference_frames_dir: str, cache_dir: str, dataset_dir: str,
                  temporal_pool: bool = False):
    """
    Single dispatch point for 'ufo' / 'gt' / 'yolo' mask_source, shared by
    `get_removability_scores` and every component's direct fg_protect mask
    lookup (elvis.py / presley_ai.py), so the two never disagree about which
    function a given mask_source resolves to.

    `mask_source='ufo'` is a byte-for-byte passthrough to the pre-existing
    `get_ufo_masks(..., temporal_pool=...)` call -- default behavior for
    every experiment that doesn't set `mask_source` is unchanged.

    `mask_source='gt'` reads `dataset_dir/annotations/<video_name>/*.png`
    (see `get_gt_masks`). `mask_source='yolo'` runs `get_yolo_masks` with its
    defaults. Both apply `temporal_pool` afterwards (max-pool across frames,
    matching what `get_ufo_masks` does internally for that branch).
    """
    mask_source = mask_source.lower()
    if mask_source == 'ufo':
        return get_ufo_masks(video_name, width, height, block_size, reference_frames_dir, cache_dir,
                             temporal_pool=temporal_pool)
    elif mask_source == 'gt':
        annotations_dir = os.path.join(dataset_dir, "annotations")
        masks = get_gt_masks(video_name, width, height, block_size, reference_frames_dir, cache_dir,
                             annotations_dir)
    elif mask_source == 'yolo':
        masks = get_yolo_masks(video_name, width, height, block_size, reference_frames_dir, cache_dir)
    else:
        raise ValueError(f"Unknown mask_source: {mask_source!r} (expected 'ufo', 'gt', or 'yolo')")

    if temporal_pool and len(masks) > 0:
        pooled = np.max(masks, axis=0)
        masks = np.repeat(pooled[None, ...], len(masks), axis=0)
    return masks


def get_removability_scores(video_name: str, width: int, height: int, block_size: int,
                            alpha: float, beta: float, dataset_dir: str, cache_dir: str,
                            mask_source: str = 'ufo'):
    """
    Returns combined removability scores (F, BY, BX). Caches to disk.

    `mask_source` selects which foreground mask feeds the BG-boost step
    below: 'ufo' (default, preserves all pre-existing cached scores and
    results exactly), 'gt' (ground-truth annotations), or 'yolo' (open-vocab
    YOLOE detections) -- see `resolve_masks`.
    """
    key_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}_bs{block_size}")
    os.makedirs(key_dir, exist_ok=True)

    # Non-'ufo' mask sources get their own cache filename so switching
    # mask_source can never silently return another source's stale cached
    # scores -- the 'ufo' filename is left exactly as before so every
    # existing cache entry (and every existing results/<hash>/ that depended
    # on it) stays valid.
    mask_suffix = "" if mask_source == "ufo" else f"_mask-{mask_source}"
    score_path = os.path.join(key_dir, f"removability_a{alpha:.2f}_b{beta:.2f}{mask_suffix}.npy")

    if os.path.exists(score_path):
        return np.load(score_path)

    raw_yuv_path, frames, _ = get_reference_frames(video_name, width, height, dataset_dir, cache_dir)
    ref_frames_dir = os.path.join(cache_dir, f"{video_name}_{width}x{height}", "reference_frames")

    temporal_3d, spatial_3d = get_evca_scores(video_name, width, height, block_size, raw_yuv_path, ref_frames_dir, cache_dir)
    ufo_masks = resolve_masks(mask_source, video_name, width, height, block_size, ref_frames_dir, cache_dir, dataset_dir)

    removability_scores = np.zeros_like(spatial_3d)
    removability_scores[:-1] = alpha * spatial_3d[:-1] + (1 - alpha) * temporal_3d[1:]
    removability_scores[-1] = spatial_3d[-1]
    
    num_blocks_x = width // block_size
    num_blocks_y = height // block_size
    num_frames = removability_scores.shape[0]
    
    for i in range(num_frames):
        mask = ufo_masks[i]
        resized_mask = cv2.resize(mask, (num_blocks_x, num_blocks_y), interpolation=cv2.INTER_NEAREST)
        background_blocks = resized_mask == 0
        removability_scores[i][background_blocks] *= 10.0
        
    if beta < 1.0 and num_frames >= 2:
        smoothed = np.zeros_like(removability_scores)
        smoothed[0] = removability_scores[0]
        smoothed[1:] = beta * removability_scores[1:] + (1 - beta) * removability_scores[:-1]
        removability_scores = smoothed
        
    removability_scores = normalize_array(removability_scores)
    np.save(score_path, removability_scores)
    
    return removability_scores
