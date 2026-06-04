"""
Tests for utils.checkpoint.
"""

import numpy as np
import pytest

from nn.linear import Linear
from nn.model import Sequential
from utils.checkpoint import checkpoint_path_from_hyperparams, load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint_restores_parameters(tmp_path):
    model = Sequential([
        Linear(3, 4, seed=0),
        Linear(4, 2, seed=1),
    ])
    original_params = [param.copy() for param, _ in model.params_and_grads()]

    checkpoint_path = tmp_path / "model.npz"
    saved_path = save_checkpoint(model, checkpoint_path, metadata={"epoch": 3})

    for param, _ in model.params_and_grads():
        param[...] = 0.0

    metadata = load_checkpoint(model, saved_path)

    assert metadata == {"epoch": 3}
    for (param, _), original_param in zip(model.params_and_grads(), original_params):
        np.testing.assert_allclose(param, original_param)


def test_load_checkpoint_raises_on_parameter_count_mismatch(tmp_path):
    source_model = Sequential([
        Linear(3, 4, seed=0),
    ])
    target_model = Sequential([
        Linear(3, 4, seed=0),
        Linear(4, 2, seed=1),
    ])

    checkpoint_path = tmp_path / "model.npz"
    save_checkpoint(source_model, checkpoint_path)

    with pytest.raises(ValueError):
        load_checkpoint(target_model, checkpoint_path)


def test_load_checkpoint_raises_on_shape_mismatch(tmp_path):
    source_model = Sequential([
        Linear(3, 4, seed=0),
    ])
    target_model = Sequential([
        Linear(3, 5, seed=0),
    ])

    checkpoint_path = tmp_path / "model.npz"
    save_checkpoint(source_model, checkpoint_path)

    with pytest.raises(ValueError):
        load_checkpoint(target_model, checkpoint_path)


def test_checkpoint_path_from_hyperparams_contains_experiment_settings(tmp_path):
    hyperparams = {
        "MODEL_TYPE": "cnn",
        "BATCH_SIZE": 64,
        "HIDDEN_DIMS": [128],
        "LEARNING_RATE": 0.01,
        "OPTIMIZER": "momentum",
        "ACTIVATION": "ReLU",
        "RANDOM_SEED": 42,
        "WEIGHT_INIT": "he",
        "MOMENTUM": 0.9,
    }

    path = checkpoint_path_from_hyperparams(hyperparams, save_dir=tmp_path)

    assert path.parent == tmp_path
    assert path.name == (
        "model_type-cnn_bs-64_hidden-128_lr-0.01_opt-momentum_act-ReLU_"
        "seed-42_init-he_momentum-0.9.npz"
    )
