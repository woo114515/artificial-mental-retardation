"""
Tests for nn.maxpool2d.MaxPool2D.
"""

import numpy as np
import pytest

from nn.maxpool2d import MaxPool2D


def test_maxpool2d_forward_shape():
    pool = MaxPool2D(kernel_size=2, stride=2)

    X = np.random.randn(4, 3, 8, 8)
    out = pool.forward(X)

    assert out.shape == (4, 3, 4, 4)


def test_maxpool2d_forward_values():
    pool = MaxPool2D(kernel_size=2, stride=2)

    X = np.array([
        [
            [
                [1.0, 2.0, 5.0, 4.0],
                [3.0, 4.0, 6.0, 1.0],
                [9.0, 8.0, 7.0, 6.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        ]
    ])

    out = pool.forward(X)

    expected = np.array([
        [
            [
                [4.0, 6.0],
                [9.0, 7.0],
            ]
        ]
    ])
    np.testing.assert_allclose(out, expected)


def test_maxpool2d_backward_values():
    pool = MaxPool2D(kernel_size=2, stride=2)

    X = np.array([
        [
            [
                [1.0, 2.0, 5.0, 4.0],
                [3.0, 4.0, 6.0, 1.0],
                [9.0, 8.0, 7.0, 6.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        ]
    ])

    out = pool.forward(X)
    dout = np.ones_like(out)
    dX = pool.backward(dout)

    expected = np.array([
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    ])
    np.testing.assert_allclose(dX, expected)


def test_maxpool2d_backward_shape():
    pool = MaxPool2D(kernel_size=2, stride=2)

    X = np.random.randn(5, 2, 6, 6)
    out = pool.forward(X)
    dout = np.random.randn(*out.shape)
    dX = pool.backward(dout)

    assert dX.shape == X.shape


def test_maxpool2d_default_stride_equals_kernel_size():
    pool = MaxPool2D(kernel_size=2)

    X = np.random.randn(2, 1, 4, 4)
    out = pool.forward(X)

    assert out.shape == (2, 1, 2, 2)


def test_maxpool2d_params_and_grads_empty():
    pool = MaxPool2D(kernel_size=2, stride=2)

    assert pool.params_and_grads() == []


def test_maxpool2d_backward_before_forward_raises_error():
    pool = MaxPool2D(kernel_size=2, stride=2)

    dout = np.ones((1, 1, 2, 2))

    with pytest.raises(RuntimeError):
        pool.backward(dout)


def test_maxpool2d_invalid_input_dimension_raises_error():
    pool = MaxPool2D(kernel_size=2, stride=2)

    X = np.random.randn(4, 28, 28)

    with pytest.raises(ValueError):
        pool.forward(X)


def test_maxpool2d_invalid_dout_shape_raises_error():
    pool = MaxPool2D(kernel_size=2, stride=2)

    X = np.random.randn(1, 1, 4, 4)
    pool.forward(X)

    dout = np.ones((1, 1, 3, 3))

    with pytest.raises(ValueError):
        pool.backward(dout)


def test_maxpool2d_invalid_kernel_or_stride_raises_error():
    with pytest.raises(ValueError):
        MaxPool2D(kernel_size=0)

    with pytest.raises(ValueError):
        MaxPool2D(kernel_size=2, stride=0)

