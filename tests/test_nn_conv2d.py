"""
Tests for nn.conv2d.Conv2D.
"""

import numpy as np
import pytest

from nn.conv2d import Conv2D


def test_conv2d_forward_shape_with_padding():
    layer = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, seed=0)

    X = np.random.randn(4, 1, 28, 28)
    out = layer.forward(X)

    assert out.shape == (4, 2, 28, 28)


def test_conv2d_forward_values_simple_kernel():
    layer = Conv2D(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, seed=0)
    layer.W[...] = 1.0
    layer.b[...] = 0.0

    X = np.array([
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ]
    ])

    out = layer.forward(X)

    expected = np.array([
        [
            [
                [12.0, 16.0],
                [24.0, 28.0],
            ]
        ]
    ])
    np.testing.assert_allclose(out, expected)


def test_conv2d_backward_shapes():
    layer = Conv2D(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, seed=0)

    X = np.random.randn(5, 2, 8, 8)
    out = layer.forward(X)
    dout = np.random.randn(*out.shape)
    dX = layer.backward(dout)

    assert dX.shape == X.shape
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape


def test_conv2d_params_and_grads():
    layer = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, seed=0)

    X = np.random.randn(4, 1, 5, 5)
    out = layer.forward(X)
    dout = np.random.randn(*out.shape)
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


def test_conv2d_backward_before_forward_raises_error():
    layer = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, seed=0)

    dout = np.random.randn(2, 1, 5, 5)

    with pytest.raises(RuntimeError):
        layer.backward(dout)


def test_conv2d_forward_invalid_input_dimension_raises_error():
    layer = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, seed=0)

    X = np.random.randn(4, 28, 28)

    with pytest.raises(ValueError):
        layer.forward(X)


def test_conv2d_backward_dW_matches_numerical_gradient():
    np.random.seed(0)
    layer = Conv2D(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, seed=1)

    X = np.random.randn(1, 1, 3, 3)
    out = layer.forward(X)
    dout = np.random.randn(*out.shape)
    layer.backward(dout)

    eps = 1e-5
    max_error = 0.0

    for index in np.ndindex(layer.W.shape):
        old_value = layer.W[index]

        layer.W[index] = old_value + eps
        loss_plus = np.sum(layer.forward(X) * dout)

        layer.W[index] = old_value - eps
        loss_minus = np.sum(layer.forward(X) * dout)

        layer.W[index] = old_value
        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        max_error = max(max_error, abs(numerical_grad - layer.dW[index]))

    assert max_error < 1e-8


def test_conv2d_backward_dX_matches_numerical_gradient():
    np.random.seed(1)
    layer = Conv2D(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, seed=2)

    X = np.random.randn(1, 1, 3, 3)
    out = layer.forward(X)
    dout = np.random.randn(*out.shape)
    dX = layer.backward(dout)

    eps = 1e-5
    max_error = 0.0

    for index in np.ndindex(X.shape):
        old_value = X[index]

        X[index] = old_value + eps
        loss_plus = np.sum(layer.forward(X) * dout)

        X[index] = old_value - eps
        loss_minus = np.sum(layer.forward(X) * dout)

        X[index] = old_value
        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        max_error = max(max_error, abs(numerical_grad - dX[index]))

    assert max_error < 1e-8

