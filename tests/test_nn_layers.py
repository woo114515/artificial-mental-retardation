"""
Tests for nn.layers.Linear.

Convention:
    X:    (batch_size, in_features)
    W:    (in_features, out_features)
    b:    (out_features,)
    out:  (batch_size, out_features)
"""

import numpy as np
import pytest

from nn.layers import Linear


def test_linear_forward_shape():
    layer = Linear(in_features=4, out_features=3, seed=0)

    X = np.random.randn(5, 4)
    out = layer.forward(X)

    assert out.shape == (5, 3)


def test_linear_backward_shape():
    layer = Linear(in_features=4, out_features=3, seed=0)

    X = np.random.randn(5, 4)
    out = layer.forward(X)

    dout = np.random.randn(5, 3)
    dX = layer.backward(dout)

    assert dX.shape == X.shape
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape


def test_linear_forward_values():
    layer = Linear(in_features=2, out_features=3, seed=0)

    layer.W = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    layer.b = np.array([0.5, 1.0, -1.0])

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    out = layer.forward(X)

    expected = np.array([
        [9.5, 13.0, 14.0],
        [19.5, 27.0, 32.0],
    ])

    np.testing.assert_allclose(out, expected)


def test_linear_backward_values():
    layer = Linear(in_features=2, out_features=3, seed=0)

    layer.W = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    layer.b = np.array([0.5, 1.0, -1.0])

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    dout = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 3.0],
    ])

    layer.forward(X)
    dX = layer.backward(dout)

    expected_dW = np.array([
        [1.0, 3.0, 11.0],
        [2.0, 4.0, 16.0],
    ])

    expected_db = np.array([1.0, 1.0, 5.0])

    expected_dX = np.array([
        [7.0, 16.0],
        [11.0, 23.0],
    ])

    np.testing.assert_allclose(layer.dW, expected_dW)
    np.testing.assert_allclose(layer.db, expected_db)
    np.testing.assert_allclose(dX, expected_dX)


def test_linear_backward_before_forward_raises_error():
    layer = Linear(in_features=4, out_features=3, seed=0)

    dout = np.random.randn(5, 3)

    with pytest.raises(RuntimeError):
        layer.backward(dout)


def test_linear_params_and_grads():
    layer = Linear(in_features=4, out_features=3, seed=0)

    X = np.random.randn(5, 4)
    dout = np.random.randn(5, 3)

    layer.forward(X)
    layer.backward(dout)

    params_and_grads = layer.params_and_grads()

    assert len(params_and_grads) == 2

    W, dW = params_and_grads[0]
    b, db = params_and_grads[1]

    assert W is layer.W
    assert dW is layer.dW
    assert b is layer.b
    assert db is layer.db

    assert W.shape == dW.shape
    assert b.shape == db.shape