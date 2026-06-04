"""
Tests for the browser handwritten digit demo helpers.
"""

import numpy as np
import pytest

from config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_TYPE
from demo_handwritten_digit import metadata_title, prepare_pixels


def test_prepare_pixels_returns_model_input_shape():
    pixels = np.zeros(IMAGE_HEIGHT * IMAGE_WIDTH, dtype=np.float32)

    X_model = prepare_pixels(pixels.tolist())

    if MODEL_TYPE == "cnn":
        assert X_model.shape == (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    else:
        assert X_model.shape == (1, IMAGE_HEIGHT * IMAGE_WIDTH)


def test_prepare_pixels_rejects_wrong_size():
    with pytest.raises(ValueError):
        prepare_pixels([0.0, 1.0])


def test_metadata_title_includes_core_hyperparams():
    title = metadata_title(
        {
            "MODEL_TYPE": "cnn",
            "OPTIMIZER": "momentum",
            "ACTIVATION": "ReLU",
            "LEARNING_RATE": 0.01,
            "BATCH_SIZE": 64,
            "HIDDEN_DIMS": [128],
            "WEIGHT_INIT": "he",
            "RANDOM_SEED": 42,
            "CNN_OUT_CHANNELS": 8,
            "CNN_KERNEL_SIZE": 3,
        }
    )

    assert "MODEL_TYPE=cnn" in title
    assert "OPTIMIZER=momentum" in title
    assert "CNN_OUT_CHANNELS=8" in title
