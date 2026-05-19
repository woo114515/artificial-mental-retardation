"""
Tests for nn.activations.Sigmoid.
"""

import numpy as np
import pytest

from nn.activations import Sigmoid


def test_sigmoid_forward_basic():
    sigmoid = Sigmoid()

    x = np.array([
        [-1.0, 0.0, 1.0],
        [2.0, -2.0, 3.0],
    ])

    out = sigmoid.forward(x)
    expected = 1 / (1 + np.exp(-x))

    np.testing.assert_allclose(out, expected)


def test_sigmoid_backward_basic():
    sigmoid = Sigmoid()

    x = np.array([
        [-1.0, 0.0, 1.0],
        [2.0, -2.0, 3.0],
    ])
    dout = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    out = sigmoid.forward(x)
    dx = sigmoid.backward(dout)
    expected = dout * out * (1 - out)

    np.testing.assert_allclose(dx, expected)


def test_sigmoid_forward_preserves_shape():
    sigmoid = Sigmoid()

    x = np.random.randn(2, 3, 4)
    out = sigmoid.forward(x)

    assert out.shape == x.shape


def test_sigmoid_output_range():
    sigmoid = Sigmoid()

    x = np.array([-10.0, 0.0, 10.0])
    out = sigmoid.forward(x)

    assert np.all(out > 0.0)
    assert np.all(out < 1.0)


def test_sigmoid_params_and_grads_empty():
    sigmoid = Sigmoid()

    assert sigmoid.params_and_grads() == []


def test_sigmoid_backward_before_forward_raises_error():
    sigmoid = Sigmoid()

    dout = np.ones((2, 3))

    with pytest.raises(RuntimeError):
        sigmoid.backward(dout)

