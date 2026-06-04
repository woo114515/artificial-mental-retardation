"""
Tests for train.py data/model configuration helpers.
"""

import numpy as np
import pytest

import train


def test_prepare_data_keeps_flat_inputs_for_mlp(monkeypatch):
    monkeypatch.setattr(train, "MODEL_TYPE", "mlp")
    X_train = np.zeros((2, 784), dtype=np.float32)
    X_val = np.zeros((1, 784), dtype=np.float32)
    X_test = np.zeros((1, 784), dtype=np.float32)

    out_train, out_val, out_test = train.prepare_data(X_train, X_val, X_test)

    assert out_train.shape == (2, 784)
    assert out_val.shape == (1, 784)
    assert out_test.shape == (1, 784)


def test_prepare_data_uses_configured_image_shape_for_cnn(monkeypatch):
    monkeypatch.setattr(train, "MODEL_TYPE", "cnn")
    monkeypatch.setattr(train, "DATASET", "mnist")
    X_train = np.zeros((2, 784), dtype=np.float32)
    X_val = np.zeros((1, 784), dtype=np.float32)
    X_test = np.zeros((1, 784), dtype=np.float32)

    out_train, out_val, out_test = train.prepare_data(X_train, X_val, X_test)

    assert out_train.shape == (2, 1, 28, 28)
    assert out_val.shape == (1, 1, 28, 28)
    assert out_test.shape == (1, 1, 28, 28)


def test_prepare_data_uses_cifar10_image_shape_for_cnn(monkeypatch):
    monkeypatch.setattr(train, "MODEL_TYPE", "cnn")
    monkeypatch.setattr(train, "DATASET", "cifar10")
    X_train = np.zeros((2, 3072), dtype=np.float32)
    X_val = np.zeros((1, 3072), dtype=np.float32)
    X_test = np.zeros((1, 3072), dtype=np.float32)

    out_train, out_val, out_test = train.prepare_data(X_train, X_val, X_test)

    assert out_train.shape == (2, 3, 32, 32)
    assert out_val.shape == (1, 3, 32, 32)
    assert out_test.shape == (1, 3, 32, 32)


def test_prepare_data_rejects_cnn_feature_shape_mismatch(monkeypatch):
    monkeypatch.setattr(train, "MODEL_TYPE", "cnn")
    monkeypatch.setattr(train, "DATASET", "mnist")
    X_train = np.zeros((2, 3072), dtype=np.float32)
    X_val = np.zeros((1, 3072), dtype=np.float32)
    X_test = np.zeros((1, 3072), dtype=np.float32)

    with pytest.raises(ValueError, match="image_shape=\\(1, 28, 28\\)"):
        train.prepare_data(X_train, X_val, X_test)


def test_get_image_shape_from_dataset():
    assert train.get_image_shape("mnist") == (1, 28, 28)
    assert train.get_image_shape("fashionmnist") == (1, 28, 28)
    assert train.get_image_shape("cifar10") == (3, 32, 32)


def test_get_image_shape_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        train.get_image_shape("unknown")
