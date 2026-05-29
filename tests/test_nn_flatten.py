"""
Tests for nn.flatten.Flatten.
"""

import numpy as np
import pytest

from nn.flatten import Flatten


def test_flatten_forward_shape():
    flatten = Flatten()

    X = np.random.randn(2, 3, 4, 5)
    out = flatten.forward(X)

    assert out.shape == (2, 60)


def test_flatten_forward_values_preserve_order():
    flatten = Flatten()

    X = np.arange(24).reshape(2, 3, 4)
    out = flatten.forward(X)

    expected = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    ])
    np.testing.assert_array_equal(out, expected)


def test_flatten_backward_shape():
    flatten = Flatten()

    X = np.random.randn(2, 3, 4, 5)
    out = flatten.forward(X)
    dout = np.random.randn(*out.shape)
    dX = flatten.backward(dout)

    assert dX.shape == X.shape


def test_flatten_backward_values_preserve_order():
    flatten = Flatten()

    X = np.zeros((2, 3, 4))
    flatten.forward(X)

    dout = np.arange(24).reshape(2, 12)
    dX = flatten.backward(dout)

    expected = np.arange(24).reshape(2, 3, 4)
    np.testing.assert_array_equal(dX, expected)


def test_flatten_params_and_grads_empty():
    flatten = Flatten()

    assert flatten.params_and_grads() == []


def test_flatten_backward_before_forward_raises_error():
    flatten = Flatten()

    dout = np.ones((2, 12))

    with pytest.raises(RuntimeError):
        flatten.backward(dout)


def test_flatten_forward_invalid_input_dimension_raises_error():
    flatten = Flatten()

    X = np.array(1.0)

    with pytest.raises(ValueError):
        flatten.forward(X)


def test_flatten_backward_invalid_dout_shape_raises_error():
    flatten = Flatten()

    X = np.random.randn(2, 3, 4)
    flatten.forward(X)

    dout = np.ones((2, 10))

    with pytest.raises(ValueError):
        flatten.backward(dout)

