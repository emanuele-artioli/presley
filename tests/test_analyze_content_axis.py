"""Wave 2A asks whether content predicts transport choice, and answers no.

A negative is only publishable if the machinery could have detected a positive
and refused the ways of manufacturing one. The failure modes below are each a
claim the article would otherwise make and could not defend:

  * **Calling a correlation significant on too few videos.** Hard rule 2b
    requires n>=6 videos; a per-cell unit of analysis would have given n=84
    from 23 videos and made noise look real.
  * **Not correcting for the candidates tried.** Four attributes were
    pre-registered; Holm must divide by four even when an attribute fails to
    compute, or the survivors get a free pass.
  * **Reporting an association that is really run history.** A video's cell
    count is a record of what was run, not a property of its content. The
    adjudication withdraws an attribute that tracks it.
  * **Missing a real signal.** A genuinely predictive attribute must still
    come out predictive, otherwise the negative is just a broken test.
  * **Testing against a constant.** T1's winner distribution is degenerate;
    an attribute cannot predict something with no variance, and the tool must
    say so rather than run a test.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "analyze_content_axis.py"
_spec = importlib.util.spec_from_file_location("analyze_content_axis", _TOOL)
aca = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = aca
_spec.loader.exec_module(aca)


def cell(video, verdict, winner=None):
    return {"op": [video, "svtav1", 50, 640, 360], "verdict": verdict,
            "separable": verdict == "separable", "winner": winner}


# --- statistics the verdict rests on -----------------------------------------


def test_spearman_is_monotone_not_linear():
    # Perfectly monotone but strongly non-linear: Pearson would not give 1.0.
    assert aca.spearman([1, 2, 3, 4], [1, 8, 27, 64]) == pytest.approx(1.0)
    assert aca.spearman([1, 2, 3, 4], [64, 27, 8, 1]) == pytest.approx(-1.0)


def test_spearman_undefined_against_a_constant():
    """T1's situation: no variance in the outcome means nothing to predict."""
    assert np.isnan(aca.spearman([1, 2, 3, 4], [7, 7, 7, 7]))


def test_spearman_shares_ranks_across_ties():
    assert aca.rankdata([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_permutation_p_never_returns_zero():
    """A permutation test licenses no p below 1/(n_perm+1)."""
    _, p = aca.permutation_p(list(range(8)), list(range(8)), n_perm=200)
    assert p == pytest.approx(1 / 201)
    assert p > 0


def test_permutation_p_is_reproducible_and_two_tailed():
    a = aca.permutation_p([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5], seed=0, n_perm=500)
    b = aca.permutation_p([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5], seed=0, n_perm=500)
    assert a == b
    # Sign must not change the p-value: the test is two-tailed.
    pos = aca.permutation_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], seed=0, n_perm=500)
    neg = aca.permutation_p([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], seed=0, n_perm=500)
    assert pos[1] == neg[1]


def test_holm_corrects_for_candidates_tried_not_candidates_computed():
    """Three computable attributes still pay the full candidate correction --
    expressed against N_CANDIDATES rather than a literal, because round 2 raised
    it from 4 to 7 and a hard-coded expectation would have had to be edited to
    match rather than checking anything."""
    computed = aca.holm([0.01, 0.4, 0.9], n_tests=aca.N_CANDIDATES)
    uncorrected = aca.holm([0.01, 0.4, 0.9])
    assert computed[0] == pytest.approx(0.01 * aca.N_CANDIDATES)
    assert computed[0] > uncorrected[0]


def test_every_declared_attribute_is_corrected_for():
    """The correction must cover every attribute in ATTRIBUTES. Adding an
    attribute without raising k is a silent way to make the survivors of a
    larger search look better than they are."""
    assert aca.N_CANDIDATES == len(aca.ATTRIBUTES)


def test_holm_is_monotone_and_clipped():
    adj = aca.holm([0.01, 0.02, 0.03, 0.04])
    assert adj == sorted(adj)
    assert all(p <= 1.0 for p in adj)


def test_holm_refuses_fewer_tests_than_pvalues():
    with pytest.raises(ValueError):
        aca.holm([0.1, 0.2, 0.3], n_tests=2)


# --- the unit of analysis is the video ---------------------------------------


def test_target_rates_aggregate_per_video_not_per_cell():
    cells = [cell("bear", "separable"), cell("bear", "tie_within_threshold"),
             cell("bear", "tie_within_threshold"), cell("dog", "separable")]
    rates = aca.target_rates(cells, {"separable"},
                             {"separable", "tie_within_threshold"}, "t2")
    assert rates.rates == {"bear": pytest.approx(1 / 3), "dog": 1.0}
    assert rates.counts == {"bear": 3, "dog": 1}


def test_target_rates_exclude_verdicts_outside_the_denominator():
    """Contested cells are separable+tie; the other two verdicts pose no choice."""
    cells = [cell("bear", "separable"), cell("bear", "no_eligible_arm"),
             cell("bear", "single_eligible_arm")]
    rates = aca.target_rates(cells, {"separable"},
                             {"separable", "tie_within_threshold"}, "t2")
    assert rates.counts == {"bear": 1}


def test_n_below_six_videos_can_never_be_predictive():
    """Hard rule 2b: no significance claim below six videos, however clean."""
    cells, attrs = [], {}
    for i, v in enumerate(["a", "b", "c", "d", "e"]):
        cells += [cell(v, "separable")] * (i + 1)
        cells += [cell(v, "tie_within_threshold")] * (5 - i)
        attrs[v] = {a: float(i) for a in aca.ATTRIBUTES}
    rates = aca.target_rates(cells, {"separable"},
                             {"separable", "tie_within_threshold"}, "t2")
    rows = aca.analyse_target(rates, attrs, n_perm=500)
    assert all(not r["predictive"] for r in rows)
    assert all(r["n_videos"] < aca.MIN_VIDEOS_FOR_SIGNIFICANCE for r in rows)


# --- it must still be able to find a real signal ------------------------------


def test_a_genuinely_predictive_attribute_is_detected():
    cells, attrs = [], {}
    videos = [f"v{i}" for i in range(10)]
    for i, v in enumerate(videos):
        # rate rises with i; A1 rises with i, the rest are noise-free constants
        # perturbed so they are not degenerate.
        cells += [cell(v, "separable")] * i
        cells += [cell(v, "tie_within_threshold")] * (10 - i)
        attrs[v] = {"A1_motion": float(i), "A2_hole_instability": float(i % 3),
                    "A3_bg_texture": float((i * 7) % 5),
                    "A4_residual_info": float((i * 3) % 4)}
    rates = aca.target_rates(cells, {"separable"},
                             {"separable", "tie_within_threshold"}, "t2")
    rows = {r["attribute"]: r for r in aca.analyse_target(rates, attrs, n_perm=2000)}
    assert rows["A1_motion"]["predictive"], "a perfect predictor must be detected"
    assert rows["A1_motion"]["rho"] == pytest.approx(1.0)
    assert not rows["A2_hole_instability"]["predictive"]


# --- the confounds that withdrew the one attribute that fired -----------------


def test_coverage_confound_exposes_outcome_tracking_cell_count():
    """Single-cell videos have rate 0 or 1; denser videos drift toward the mean.

    That alone makes the rate track the cell count, which is run history.
    """
    cells = []
    for v, n_sep, n_tie in [("a", 0, 1), ("b", 0, 1), ("c", 1, 3), ("d", 3, 3)]:
        cells += [cell(v, "separable")] * n_sep
        cells += [cell(v, "tie_within_threshold")] * n_tie
    rates = aca.target_rates(cells, {"separable"},
                             {"separable", "tie_within_threshold"}, "t2")
    attrs = {v: {a: float(i) for a in aca.ATTRIBUTES}
             for i, v in enumerate(["a", "b", "c", "d"])}
    cov = aca.coverage_confound(rates, attrs)
    assert cov["outcome_vs_cellcount"] > 0.5


def test_restrict_drops_thin_and_named_videos():
    cells = [cell("bear", "separable"), cell("bear", "tie_within_threshold"),
             cell("dog", "separable")]
    rates = aca.target_rates(cells, {"separable"},
                             {"separable", "tie_within_threshold"}, "t2")
    assert set(aca.restrict(rates, min_cells=2).rates) == {"bear"}
    assert set(aca.restrict(rates, drop=("bear",)).rates) == {"dog"}


# --- T1 is a constant, and the tool must say so rather than test it -----------


def test_winner_degeneracy_flags_the_real_distribution():
    cells = [cell("v", "separable", "downsample+realesrgan") for _ in range(18)]
    cells.append(cell("v", "separable", "blackout+propainter"))
    deg = aca.winner_degeneracy(cells)
    assert deg["n_separable"] == 19
    assert deg["modal_share"] == pytest.approx(18 / 19)
    assert deg["degenerate"]


def test_winner_degeneracy_clears_when_winners_are_mixed():
    cells = [cell("v", "separable", "a") for _ in range(10)]
    cells += [cell("v", "separable", "b") for _ in range(10)]
    assert not aca.winner_degeneracy(cells)["degenerate"]


def test_winner_degeneracy_ignores_non_separable_cells():
    cells = [cell("v", "separable", "a"),
             cell("v", "tie_within_threshold", "b"),
             cell("v", "no_eligible_arm", None)]
    assert aca.winner_degeneracy(cells)["winners"] == {"a": 1}


# --- attribute construction ---------------------------------------------------


def test_foreground_blocks_follow_the_mask():
    mask = np.zeros((16, 32), dtype=np.uint8)
    mask[:8, :8] = 1  # exactly the top-left 8x8 block
    fg = aca.foreground_block_mask(mask, n_blocks=8, width=32, height=16)
    assert fg.tolist() == [True, False, False, False, False, False, False, False]


def test_foreground_blocks_resample_a_mask_of_a_different_resolution():
    """Annotations are native-resolution; EVCA grids are not."""
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    mask[:540, :960] = 1  # top-left quadrant
    fg = aca.foreground_block_mask(mask, n_blocks=8, width=32, height=16)
    assert fg.reshape(2, 4).tolist() == [[True, True, False, False],
                                         [False, False, False, False]]


def test_foreground_blocks_need_majority_coverage():
    mask = np.zeros((16, 32), dtype=np.uint8)
    mask[:3, :8] = 1  # 3/8 of the first block -- below the 0.5 threshold
    assert not aca.foreground_block_mask(mask, 8, 32, 16).any()


# --- round 2 attributes (W5) --------------------------------------------------


def _fake_cache(tmp_path, tc, sc, mask):
    """A minimal EVCA cache + annotation dir for one video at 32x16, bs8."""
    cdir = tmp_path / "cache" / "clip_32x16_bs8"
    cdir.mkdir(parents=True)
    for name, arr in (("evca_TC_blocks.csv", tc), ("evca_SC_blocks.csv", sc)):
        with open(cdir / name, "w") as fh:
            fh.write("f0\n")
            fh.write("\n".join(str(v) for v in arr) + "\n")
    ann = tmp_path / "annotations" / "clip"
    ann.mkdir(parents=True)
    from PIL import Image
    Image.fromarray(mask * 255).save(ann / "00000.png")
    return str(tmp_path / "cache"), str(tmp_path / "annotations")


def test_bg_motion_is_the_complement_of_hole_instability(tmp_path):
    """A5 exists to decompose A1, so it must be computed off the same block
    flags as A2 -- otherwise the pair is two motion numbers that merely look
    like a decomposition."""
    tc = [10.0] + [2.0] * 7          # block 0 is the foreground block
    mask = np.zeros((16, 32), dtype=np.uint8)
    mask[:8, :8] = 1
    cache, annotations = _fake_cache(tmp_path, tc, [1.0] * 8, mask)

    attrs = aca.content_attributes("clip", cache, annotations)

    assert attrs["A2_hole_instability"] == pytest.approx(10.0)
    assert attrs["A5_bg_motion"] == pytest.approx(2.0)
    assert attrs["A1_motion"] == pytest.approx(sum(tc) / 8)


def test_fg_fraction_counts_blocks_not_pixels(tmp_path):
    """A7 is measured on the block grid the transports actually operate on."""
    mask = np.zeros((16, 32), dtype=np.uint8)
    mask[:8, :16] = 1  # two of eight blocks
    cache, annotations = _fake_cache(tmp_path, [1.0] * 8, [1.0] * 8, mask)

    assert aca.content_attributes("clip", cache, annotations)["A7_fg_fraction"] \
        == pytest.approx(0.25)


def test_duration_survives_a_video_with_no_evca_cache(tmp_path):
    """A6 comes from the run record, not the cache. A video missing EVCA scores
    still has a length, and dropping it would quietly shrink A6's n to match
    the other attributes' n."""
    attrs = aca.content_attributes("clip", str(tmp_path), str(tmp_path),
                                   frames={"clip": 2440.0})

    assert attrs["A6_duration"] == 2440.0
    assert attrs["A1_motion"] is None
