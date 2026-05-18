from pathlib import Path
import sys

import numpy as np
import pytest

# 让 pytest 能从项目根目录找到 nn/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.activations import ReLU


def test_relu_forward_basic():
    relu = ReLU()

    x = np.array([
        [-1.0, 0.0, 2.0],
        [3.0, -4.0, 5.0],
    ])

    out = relu.forward(x)

    expected = np.array([
        [0.0, 0.0, 2.0],
        [3.0, 0.0, 5.0],
    ])

    np.testing.assert_array_equal(out, expected)


def test_relu_backward_basic():
    relu = ReLU()

    x = np.array([
        [-1.0, 0.0, 2.0],
        [3.0, -4.0, 5.0],
    ])

    dout = np.array([
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
    ])

    relu.forward(x)
    dx = relu.backward(dout)

    expected = np.array([
        [0.0, 0.0, 10.0],
        [20.0, 0.0, 20.0],
    ])

    np.testing.assert_array_equal(dx, expected)


def test_relu_forward_preserves_shape():
    relu = ReLU()

    x = np.random.randn(2, 3, 4)
    out = relu.forward(x)

    assert out.shape == x.shape


def test_relu_params_and_grads_empty():
    relu = ReLU()

    assert relu.params_and_grads() == []


def test_relu_backward_before_forward_raises_error():
    relu = ReLU()

    dout = np.ones((2, 3))

    with pytest.raises(RuntimeError):
        relu.backward(dout)