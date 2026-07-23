"""Frame and mask loading.

Mask loading is the risky half: it pads short mask sequences and falls back to
an all-foreground mask when it finds nothing. Both behaviours are silent, and
both change what every region metric measures, so they are pinned here.
"""

import cv2
import numpy as np
import pytest

from presley.io import (
    clear_directory,
    get_frame_paths,
    load_frame,
    load_frames,
    load_masks,
    save_frame,
    save_frames,
)


@pytest.fixture
def frames(rng):
    return [rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8) for _ in range(3)]


def write_mask(path, height, width, *, filled):
    cv2.imwrite(str(path), np.full((height, width), 255 if filled else 0, dtype=np.uint8))


# --- frames --------------------------------------------------------------------


def test_a_saved_frame_round_trips(tmp_path, frames):
    path = tmp_path / "nested" / "frame.png"
    save_frame(frames[0], str(path))

    assert np.array_equal(load_frame(str(path)), frames[0])


def test_loading_a_missing_frame_raises(tmp_path):
    with pytest.raises(IOError):
        load_frame(str(tmp_path / "absent.png"))


def test_frames_load_back_in_saved_order(tmp_path, frames):
    save_frames(frames, str(tmp_path))
    loaded = load_frames(str(tmp_path))

    assert len(loaded) == 3
    for original, roundtripped in zip(frames, loaded):
        assert np.array_equal(original, roundtripped)


def test_frame_numbering_starts_at_one_by_default(tmp_path, frames):
    paths = save_frames(frames, str(tmp_path))
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["00001.png", "00002.png", "00003.png"]


def test_frames_sort_numerically_not_lexically(tmp_path, rng):
    """Zero-padded names are what keep frame 10 after frame 9.

    A pattern change that dropped the padding would reorder the sequence and
    silently compare frame N of one video against frame M of another.
    """
    frames = [rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8) for _ in range(11)]
    save_frames(frames, str(tmp_path))

    loaded = load_frames(str(tmp_path))
    assert np.array_equal(loaded[9], frames[9])
    assert np.array_equal(loaded[10], frames[10])


def test_load_frames_falls_back_to_other_extensions(tmp_path, frames):
    """The default pattern is *.png, but a jpg directory must still load."""
    save_frames(frames, str(tmp_path), pattern="%05d.jpg")
    assert len(load_frames(str(tmp_path))) == 3


def test_loading_a_missing_directory_raises(tmp_path):
    with pytest.raises(ValueError):
        load_frames(str(tmp_path / "absent"))


def test_get_frame_paths_ignores_non_images(tmp_path, frames):
    save_frames(frames, str(tmp_path))
    (tmp_path / "notes.txt").write_text("not a frame")

    assert len(get_frame_paths(str(tmp_path))) == 3


def test_get_frame_paths_on_a_missing_directory_is_empty(tmp_path):
    assert get_frame_paths(str(tmp_path / "absent")) == []


def test_clear_directory_removes_only_images(tmp_path, frames):
    save_frames(frames, str(tmp_path))
    keep = tmp_path / "result.json"
    keep.write_text("{}")

    clear_directory(str(tmp_path))

    assert get_frame_paths(str(tmp_path)) == []
    assert keep.exists()


# --- masks ---------------------------------------------------------------------


def test_masks_are_thresholded_into_complementary_regions(tmp_path):
    write_mask(tmp_path / "00001.png", 8, 12, filled=True)

    fg, bg = load_masks(str(tmp_path), width=12, height=8, expected_count=1)

    assert fg[0].all()
    assert not bg[0].any()
    assert np.array_equal(bg[0], ~fg[0])


def test_masks_are_resized_to_the_frame(tmp_path):
    """A mask at the wrong resolution must be resized, not silently misaligned."""
    write_mask(tmp_path / "00001.png", 16, 24, filled=True)

    fg, _ = load_masks(str(tmp_path), width=12, height=8, expected_count=1)

    assert fg[0].shape == (8, 12)


def test_a_short_mask_sequence_repeats_the_last_mask(tmp_path):
    """Fewer masks than frames pads with the last one rather than failing.

    Worth pinning because it is silent: the tail frames are scored against a
    stale foreground, which shows up as a metric drift rather than an error.
    """
    write_mask(tmp_path / "00001.png", 8, 12, filled=True)

    fg, _ = load_masks(str(tmp_path), width=12, height=8, expected_count=4)

    assert len(fg) == 4
    assert all(m.all() for m in fg)


def test_no_masks_at_all_falls_back_to_all_foreground(tmp_path):
    """The fallback makes every metric a whole-frame metric.

    That is the intended safe default, but it means a missing mask directory
    produces plausible "foreground" numbers instead of an error — so a caller
    must never take an FG number on faith when masks may be absent.
    """
    fg, bg = load_masks(str(tmp_path), width=12, height=8, expected_count=2)

    assert len(fg) == 2
    assert all(m.all() for m in fg)
    assert not any(m.any() for m in bg)


def test_zero_expected_frames_yields_no_masks(tmp_path):
    assert load_masks(str(tmp_path), width=12, height=8, expected_count=0) == ([], [])
