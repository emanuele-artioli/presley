"""The corrected selection objective, and the leak it is built to refuse.

The dangerous failure here is not a crash. It is a damage model that trained on
the very clip it is scoring: the arm under test would then be predicting its
own answer, the race would flatter it, and no downstream check could tell. So
the refusal is tested first and hardest.

The rest pins the contract `presley_ai` depends on: the corrected score keeps
the same orientation as the removability score it replaces (higher = better to
degrade), so it can be handed straight to `select_removal_mask_global` without
disturbing the clustering blur, the hard foreground exclusion or the budget.
"""

import json

import numpy as np
import pytest

from presley.damagemodel import (FEATURES, DamagePredictor, block_features,
                                 load, save)


@pytest.fixture
def models():
    return [
        DamagePredictor(beta=np.arange(len(FEATURES) + 1, dtype=float),
                        held_out_video="bear", n_train_runs=87),
        DamagePredictor(beta=np.ones(len(FEATURES) + 1), held_out_video="camel",
                        n_train_runs=93),
    ]


def test_refuses_a_model_that_trained_on_the_clip_being_scored(tmp_path, models):
    """The leak that would be undetectable downstream."""
    path = tmp_path / "m.json"
    save(models, str(path))
    with pytest.raises(KeyError, match="no damage model that excludes"):
        load(str(path), "dog")            # no fold held dog out


def test_load_returns_the_fold_that_excludes_the_video(tmp_path, models):
    path = tmp_path / "m.json"
    save(models, str(path))
    m = load(str(path), "bear")
    assert m.held_out_video == "bear"
    assert m.n_train_runs == 87


def test_round_trips_through_disk(tmp_path, models):
    path = tmp_path / "m.json"
    save(models, str(path))
    np.testing.assert_allclose(load(str(path), "camel").beta, models[1].beta)


def test_rejects_a_model_file_fit_on_different_features(tmp_path, models):
    """A silently-reordered feature list would apply coefficients to the wrong
    columns and still produce plausible-looking numbers."""
    path = tmp_path / "m.json"
    save(models, str(path))
    payload = json.loads(path.read_text())
    payload["features"] = ["sc_mean", "tc_mean"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="was fit on features"):
        load(str(path), "bear")


def test_predict_rejects_the_wrong_feature_count(models):
    with pytest.raises(ValueError, match="expected 5 features"):
        models[0].predict(np.zeros((10, 3)))


def test_corrected_score_keeps_the_orientation_of_the_removability_score(rng):
    """Higher must still mean "better to degrade", or selection inverts."""
    beta = np.zeros(len(FEATURES) + 1)      # a constant damage prediction
    model = DamagePredictor(beta=beta, held_out_video="x", n_train_runs=1)
    removability = rng.random(50)
    features = rng.random((50, len(FEATURES)))

    corrected = model.corrected_scores(removability, features)
    # Constant denominator -> the ranking must be exactly preserved.
    assert np.array_equal(np.argsort(corrected), np.argsort(removability))


def test_high_predicted_damage_demotes_a_block():
    """The whole point: two blocks equal on bits, ordered by restorability."""
    beta = np.zeros(len(FEATURES) + 1)
    beta[0] = 1.0                            # damage rises with sc_mean
    model = DamagePredictor(beta=beta, held_out_video="x", n_train_runs=1)

    removability = np.array([0.5, 0.5])      # identical bits proxy
    features = np.zeros((2, len(FEATURES)))
    features[:, 0] = [0.0, 10.0]             # second block restores far worse

    corrected = model.corrected_scores(removability, features)
    assert corrected[0] > corrected[1]


def test_corrected_score_is_finite_when_predictions_go_negative(rng):
    """A ridge fit of a rank target can predict below zero; a ratio through
    zero would be a division by ~0 and silently produce inf."""
    beta = -np.ones(len(FEATURES) + 1) * 5.0
    model = DamagePredictor(beta=beta, held_out_video="x", n_train_runs=1)
    out = model.corrected_scores(rng.random(40), rng.random((40, len(FEATURES))))
    assert np.isfinite(out).all()


def test_block_features_shape_and_order():
    frames, rows, cols = 4, 3, 5
    spatial = np.zeros((frames, rows, cols))
    temporal = np.zeros((frames, rows, cols))
    spatial[1, 2, 3] = 7.0
    temporal[1, 2, 3] = 9.0

    feats = block_features(spatial, temporal, frame_index=1)
    assert feats.shape == (rows * cols, len(FEATURES))
    flat = 2 * cols + 3
    assert feats[flat, 0] == 7.0        # sc_mean
    assert feats[flat, 1] == 9.0        # tc_mean


def test_frame_edge_marks_only_the_border():
    spatial = np.zeros((2, 4, 4))
    feats = block_features(spatial, spatial, 0)
    edge = feats[:, FEATURES.index("frame_edge")].reshape(4, 4)
    assert edge[0].all() and edge[-1].all()
    assert edge[:, 0].all() and edge[:, -1].all()
    assert not edge[1:3, 1:3].any()


def test_variance_features_are_across_frames_not_within_one():
    """A per-frame variance would be a different quantity under the same name,
    and the model was fit on the across-frames one."""
    spatial = np.zeros((3, 2, 2))
    spatial[:, 0, 0] = [0.0, 10.0, 20.0]     # this block varies over time
    feats = block_features(spatial, spatial, 0)
    sc_var = feats[:, FEATURES.index("sc_var")].reshape(2, 2)
    assert sc_var[0, 0] > 0
    assert sc_var[0, 1] == 0
