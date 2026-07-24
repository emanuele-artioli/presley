"""Mask-ingestion and framerate-probing stage contracts (D2/D3/D4 of the
dataset/mask referee-response infrastructure).

Covers:
  - `get_gt_masks`' output contract must match `get_ufo_masks`' exactly
    (shape, dtype, values) so it is a true drop-in swap at the
    `get_removability_scores` call site.
  - `resolve_masks` dispatch (the single place mask_source is interpreted)
    and its failure mode for an unknown source.
  - `probe_framerate`'s dataset-registry / video-file / inference logic --
    the D4 fix for the old blanket "assume 24fps" default.
  - `get_removability_scores`' mask_source-keyed cache filename, which is
    the mechanism that stops a non-'ufo' mask_source from silently
    returning another source's stale cached scores.

Everything here is synthetic and CPU-only: no dataset/, no GPU, no UFO/YOLO
model weights.
"""

import subprocess

import cv2
import numpy as np
import pytest
from PIL import Image

from presley.preprocessing import (
    get_gt_masks,
    get_removability_scores,
    probe_framerate,
    resolve_masks,
)


def _write_ref_frames(ref_dir, n, height, width):
    ref_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(ref_dir / f"{i+1:05d}.png"),
                    np.zeros((height, width, 3), dtype=np.uint8))


def _write_palette_mask(path, height, width, *, fg_value, box=None):
    """A plain single-channel grayscale mask (as cv2.imwrite writes it):
    `fg_value` inside `box` (or everywhere if None), 0 elsewhere. Good enough
    for most tests here, but NOT a real palette-indexed PNG -- see
    `_write_true_palette_mask` below for that, which is the one that actually
    catches the cv2-vs-PIL decoding bug this module was regression-tested
    for."""
    arr = np.zeros((height, width), dtype=np.uint8)
    if box is None:
        arr[:] = fg_value
    else:
        y0, y1, x0, x1 = box
        arr[y0:y1, x0:x1] = fg_value
    cv2.imwrite(str(path), arr)


def _write_true_palette_mask(path, height, width, *, fg_index, box):
    """A genuine PIL "P"-mode (palette-indexed) PNG, matching DAVIS/MOSEv2/
    YouTube-VOS's actual on-disk annotation format -- as opposed to the
    plain-grayscale stand-in `_write_palette_mask` writes via cv2.imwrite.
    The distinction matters: cv2.imread (any flag, including
    IMREAD_UNCHANGED) silently expands a "P"-mode PNG through its color
    palette into a 3-channel BGR image instead of returning the raw index
    values -- this is exactly the bug found by testing get_gt_masks against
    real `dataset/annotations/bear` data (every real DAVIS mask decoded to
    all-zero). A palette entry is deliberately picked whose RGB channels are
    NOT all matching `fg_index`, so a regression back to cv2-based decoding
    reads a wrong/zero value here rather than accidentally passing anyway.
    """
    arr = np.zeros((height, width), dtype=np.uint8)
    y0, y1, x0, x1 = box
    arr[y0:y1, x0:x1] = fg_index
    img = Image.fromarray(arr, mode="P")
    # DAVIS-style palette: index 0 -> black, index 1 -> a color whose R/G/B
    # channels are unequal and none of them equal `fg_index` (128, 0, 64) --
    # so cv2's silent RGB-expansion would read back a wrong channel value,
    # not a lucky coincidence of the same number.
    palette = [0, 0, 0] + [128, 0, 64] + [0, 0, 0] * 254
    img.putpalette(palette)
    img.save(str(path))


# --- get_gt_masks ----------------------------------------------------------


def test_gt_masks_match_ufo_output_contract(tmp_path):
    """Shape (F,H,W), dtype uint8, values in {0,255} -- the exact contract
    `get_removability_scores` and every fg_protect call site rely on."""
    height, width, n_frames = 16, 24, 3
    ref_dir = tmp_path / "ref_frames"
    _write_ref_frames(ref_dir, n_frames, height, width)

    annot_dir = tmp_path / "annotations" / "myvideo"
    annot_dir.mkdir(parents=True)
    for i in range(n_frames):
        _write_palette_mask(annot_dir / f"{i+1:05d}.png", height, width,
                            fg_value=1, box=(4, 8, 4, 8))

    masks = get_gt_masks("myvideo", width, height, 8, str(ref_dir), str(tmp_path / "cache"),
                        str(tmp_path / "annotations"))

    assert masks.shape == (n_frames, height, width)
    assert masks.dtype == np.uint8
    assert set(np.unique(masks).tolist()) <= {0, 255}
    # the annotated box is genuinely foreground, everything else background
    assert masks[0, 5, 5] == 255
    assert masks[0, 0, 0] == 0


def test_gt_masks_collapse_multi_instance_to_single_fg_blob(tmp_path):
    """DAVIS-style multi-object masks (pixel value = object id) must collapse
    to one semantic FG blob, matching how UFO's single saliency mask is used
    (get_removability_scores only ever tests `resized_mask == 0`)."""
    height, width = 10, 10
    ref_dir = tmp_path / "ref_frames"
    _write_ref_frames(ref_dir, 1, height, width)

    annot_dir = tmp_path / "annotations" / "multiobj"
    annot_dir.mkdir(parents=True)
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[0:3, 0:3] = 1   # object 1
    arr[5:8, 5:8] = 2   # object 2 (different id, still foreground)
    cv2.imwrite(str(annot_dir / "00001.png"), arr)

    masks = get_gt_masks("multiobj", width, height, 8, str(ref_dir), str(tmp_path / "cache"),
                        str(tmp_path / "annotations"))

    assert masks[0, 1, 1] == 255  # object 1's region
    assert masks[0, 6, 6] == 255  # object 2's region (different id, still FG)
    assert masks[0, 9, 9] == 0    # background


def test_gt_masks_reads_true_palette_indexed_png_correctly(tmp_path):
    """Regression test: cv2.imread (incl. IMREAD_UNCHANGED) silently expands
    a "P"-mode (palette-indexed) PNG into a 3-channel BGR image instead of
    returning raw index values -- this made every real DAVIS annotation
    decode to an all-zero mask until caught by testing against actual
    `dataset/annotations/bear` data. The plain-grayscale masks the other
    tests in this file write via cv2.imwrite do NOT exercise "P" mode, so
    this is the one test that would fail if the fix regresses back to
    cv2-based decoding."""
    height, width, n_frames = 16, 24, 2
    ref_dir = tmp_path / "ref_frames"
    _write_ref_frames(ref_dir, n_frames, height, width)

    annot_dir = tmp_path / "annotations" / "palettevideo"
    annot_dir.mkdir(parents=True)
    for i in range(n_frames):
        _write_true_palette_mask(annot_dir / f"{i+1:05d}.png", height, width,
                                 fg_index=1, box=(4, 8, 4, 8))

    masks = get_gt_masks("palettevideo", width, height, 8, str(ref_dir), str(tmp_path / "cache"),
                        str(tmp_path / "annotations"))

    assert masks.shape == (n_frames, height, width)
    assert masks.dtype == np.uint8
    # the annotated box must read back as foreground -- a regression to
    # cv2-based decoding reads this as 0 (see docstring/palette above).
    assert masks[0, 5, 5] == 255
    assert masks[0, 0, 0] == 0
    assert (masks > 0).any(), "gt mask is all-zero -- likely decoding the palette PNG's RGB instead of its raw indices"


def test_gt_masks_resizes_to_target_resolution(tmp_path):
    """Annotations at a different resolution than the reference frames (the
    common case: DAVIS ships Full-Resolution annotations, experiments often
    run at a downscaled width/height) get resized, not rejected."""
    ref_h, ref_w = 8, 8
    ann_h, ann_w = 32, 32
    ref_dir = tmp_path / "ref_frames"
    _write_ref_frames(ref_dir, 1, ref_h, ref_w)

    annot_dir = tmp_path / "annotations" / "hires"
    annot_dir.mkdir(parents=True)
    _write_palette_mask(annot_dir / "00001.png", ann_h, ann_w, fg_value=1,
                        box=(0, 16, 0, 16))  # top-left quadrant at hi-res

    masks = get_gt_masks("hires", ref_w, ref_h, 4, str(ref_dir), str(tmp_path / "cache"),
                        str(tmp_path / "annotations"))

    assert masks.shape == (1, ref_h, ref_w)
    assert masks[0, 1, 1] == 255   # still inside the (resized) top-left quadrant
    assert masks[0, 6, 6] == 0     # still outside it


def test_gt_masks_missing_annotations_dir_raises(tmp_path):
    ref_dir = tmp_path / "ref_frames"
    _write_ref_frames(ref_dir, 2, 8, 8)

    with pytest.raises(FileNotFoundError):
        get_gt_masks("nosuchvideo", 8, 8, 4, str(ref_dir), str(tmp_path / "cache"),
                    str(tmp_path / "annotations"))


def test_gt_masks_frame_count_mismatch_raises(tmp_path):
    """A GT folder with a different frame count than the reference frames
    means annotations_dir doesn't match this clip's sampling -- must raise,
    not silently zip/truncate to the shorter length."""
    ref_dir = tmp_path / "ref_frames"
    _write_ref_frames(ref_dir, 3, 8, 8)

    annot_dir = tmp_path / "annotations" / "mismatched"
    annot_dir.mkdir(parents=True)
    _write_palette_mask(annot_dir / "00001.png", 8, 8, fg_value=1)  # only 1, not 3

    with pytest.raises(ValueError):
        get_gt_masks("mismatched", 8, 8, 4, str(ref_dir), str(tmp_path / "cache"),
                    str(tmp_path / "annotations"))


def test_gt_masks_are_cached_and_reentrant(tmp_path):
    """Second call must not error and must return the same result (the ufo
    cache-check pattern this mirrors: `len(glob) != len(frame_files)`)."""
    ref_dir = tmp_path / "ref_frames"
    _write_ref_frames(ref_dir, 2, 8, 8)
    annot_dir = tmp_path / "annotations" / "cached"
    annot_dir.mkdir(parents=True)
    for i in range(2):
        _write_palette_mask(annot_dir / f"{i+1:05d}.png", 8, 8, fg_value=1, box=(0, 4, 0, 4))

    cache_dir = str(tmp_path / "cache")
    first = get_gt_masks("cached", 8, 8, 4, str(ref_dir), cache_dir, str(tmp_path / "annotations"))
    second = get_gt_masks("cached", 8, 8, 4, str(ref_dir), cache_dir, str(tmp_path / "annotations"))

    assert np.array_equal(first, second)


# --- resolve_masks -----------------------------------------------------------


def test_resolve_masks_dispatches_gt(tmp_path):
    ref_dir = tmp_path / "ref_frames"
    _write_ref_frames(ref_dir, 1, 8, 8)
    dataset_dir = tmp_path / "dataset"
    annot_dir = dataset_dir / "annotations" / "gtvideo"
    annot_dir.mkdir(parents=True)
    _write_palette_mask(annot_dir / "00001.png", 8, 8, fg_value=1, box=(0, 4, 0, 4))

    masks = resolve_masks("gt", "gtvideo", 8, 8, 4, str(ref_dir), str(tmp_path / "cache"), str(dataset_dir))

    assert masks.shape == (1, 8, 8)
    assert masks[0, 1, 1] == 255


def test_resolve_masks_dispatches_ufo_by_default(tmp_path, monkeypatch):
    """mask_source='ufo' must be a byte-for-byte passthrough to
    get_ufo_masks so every pre-existing experiment (no mask_source set)
    behaves exactly as before this change."""
    import presley.preprocessing as pp

    called = {}

    def fake_ufo(video_name, width, height, block_size, reference_frames_dir, cache_dir, temporal_pool=False):
        called['args'] = (video_name, width, height, block_size, reference_frames_dir, cache_dir, temporal_pool)
        return np.zeros((1, height, width), dtype=np.uint8)

    monkeypatch.setattr(pp, "get_ufo_masks", fake_ufo)

    resolve_masks("ufo", "anyvideo", 8, 8, 4, "refdir", "cachedir", "datasetdir", temporal_pool=True)

    assert called['args'] == ("anyvideo", 8, 8, 4, "refdir", "cachedir", True)


def test_resolve_masks_dispatches_yolo(tmp_path, monkeypatch):
    import presley.preprocessing as pp

    called = {}

    def fake_yolo(video_name, width, height, block_size, reference_frames_dir, cache_dir, **kwargs):
        called['args'] = (video_name, width, height, block_size, reference_frames_dir, cache_dir)
        return np.zeros((1, height, width), dtype=np.uint8)

    monkeypatch.setattr(pp, "get_yolo_masks", fake_yolo)

    resolve_masks("yolo", "anyvideo", 8, 8, 4, "refdir", "cachedir", "datasetdir")

    assert called['args'] == ("anyvideo", 8, 8, 4, "refdir", "cachedir")


def test_resolve_masks_unknown_source_raises(tmp_path):
    with pytest.raises(ValueError):
        resolve_masks("not-a-real-source", "v", 8, 8, 4, "refdir", "cachedir", "datasetdir")


# --- probe_framerate ---------------------------------------------------------


def test_probe_framerate_known_dataset(tmp_path):
    source_dir = tmp_path / "dataset" / "bear"
    source_dir.mkdir(parents=True)
    assert probe_framerate("bear", str(tmp_path / "dataset")) == 24.0


def test_probe_framerate_infers_dataset_from_namespace_prefix(tmp_path, monkeypatch):
    import presley.preprocessing as pp
    monkeypatch.setitem(pp.KNOWN_DATASET_FRAMERATES, "mosev2", 30.0)
    source_dir = tmp_path / "dataset" / "mosev2" / "someclip"
    source_dir.mkdir(parents=True)

    assert probe_framerate("mosev2/someclip", str(tmp_path / "dataset")) == 30.0


def test_probe_framerate_unknown_dataset_raises(tmp_path):
    source_dir = tmp_path / "dataset" / "someclip"
    source_dir.mkdir(parents=True)
    with pytest.raises(ValueError):
        probe_framerate("someclip", str(tmp_path / "dataset"), dataset="not_a_known_dataset")


def test_probe_framerate_ffprobes_a_real_video_file(tmp_path):
    """When the source is an actual video file, its own stream framerate
    wins over any KNOWN_DATASET_FRAMERATES entry -- this is the only
    genuinely *measured* case, everything else is a documented constant."""
    source_dir = tmp_path / "dataset" / "realclip"
    source_dir.mkdir(parents=True)
    video_path = source_dir / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=16x16:r=30", "-frames:v", "3",
         str(video_path)],
        check=True,
    )

    fps = probe_framerate("realclip", str(tmp_path / "dataset"), dataset="davis")

    assert fps == pytest.approx(30.0)


# --- get_removability_scores mask_source cache isolation ---------------------


def test_removability_scores_cache_path_differs_by_mask_source(tmp_path, monkeypatch):
    """The exact bug this project asked not to repeat: switching mask_source
    must never return another source's stale cached removability scores."""
    import presley.preprocessing as pp

    def fake_reference_frames(video_name, width, height, dataset_dir, cache_dir):
        return "fake.yuv", [np.zeros((height, width, 3), dtype=np.uint8)], 24.0

    def fake_evca_scores(video_name, width, height, block_size, raw_yuv_path, reference_frames_dir, cache_dir):
        nb = height // block_size, width // block_size
        # Non-uniform so the mask's x10 BG-boost is actually visible in the
        # output (an all-zero base score stays all-zero after any boost).
        base = np.ones((1, *nb)) * 0.5
        return base, base

    def fake_resolve_masks(mask_source, *a, **kw):
        return np.full((1, height := 8, 8), 255 if mask_source == "gt" else 0, dtype=np.uint8)

    monkeypatch.setattr(pp, "get_reference_frames", fake_reference_frames)
    monkeypatch.setattr(pp, "get_evca_scores", fake_evca_scores)
    monkeypatch.setattr(pp, "resolve_masks", fake_resolve_masks)

    cache_dir = str(tmp_path / "cache")
    ufo_scores = get_removability_scores("v", 8, 8, 4, 0.5, 1.0, str(tmp_path / "dataset"), cache_dir,
                                        mask_source="ufo")
    gt_scores = get_removability_scores("v", 8, 8, 4, 0.5, 1.0, str(tmp_path / "dataset"), cache_dir,
                                        mask_source="gt")

    key_dir = tmp_path / "cache" / "v_8x8_bs4"
    assert (key_dir / "removability_a0.50_b1.00.npy").exists()          # 'ufo' filename unchanged
    assert (key_dir / "removability_a0.50_b1.00_mask-gt.npy").exists()  # 'gt' gets its own file
    # different masks (all-BG-boosted vs all-FG) must not collapse to identical scores
    assert not np.array_equal(ufo_scores, gt_scores)
