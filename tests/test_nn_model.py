"""
Tests for nn.model.Sequential.

Convention:
    X:       (batch_size, input_dim)
    logits:  (batch_size, num_classes)
    dlogits: (batch_size, num_classes)
"""

import numpy as np

from nn.model import Sequential
from nn.linear import Linear
from nn.activations import ReLU


def test_sequential_forward_shape():
    model = Sequential([
        Linear(4, 5, seed=0),
        ReLU(),
        Linear(5, 3, seed=1),
    ])

    X = np.random.randn(6, 4)
    logits = model.forward(X)

    assert logits.shape == (6, 3)


def test_sequential_backward_shape():
    model = Sequential([
        Linear(4, 5, seed=0),
        ReLU(),
        Linear(5, 3, seed=1),
    ])

    X = np.random.randn(6, 4)
    logits = model.forward(X)

    dlogits = np.random.randn(6, 3)
    dX = model.backward(dlogits)

    assert dX.shape == X.shape


def test_sequential_params_and_grads_count():
    model = Sequential([
        Linear(4, 5, seed=0),
        ReLU(),
        Linear(5, 3, seed=1),
    ])

    X = np.random.randn(6, 4)
    logits = model.forward(X)

    dlogits = np.random.randn(6, 3)
    model.backward(dlogits)

    params_and_grads = model.params_and_grads()

    # Two Linear layers.
    # Each Linear layer has W and b.
    assert len(params_and_grads) == 4


def test_sequential_params_and_grads_shapes():
    model = Sequential([
        Linear(4, 5, seed=0),
        ReLU(),
        Linear(5, 3, seed=1),
    ])

    X = np.random.randn(6, 4)
    logits = model.forward(X)

    dlogits = np.random.randn(6, 3)
    model.backward(dlogits)

    params_and_grads = model.params_and_grads()

    for param, grad in params_and_grads:
        assert param.shape == grad.shape


def test_sequential_forward_backward_with_known_shapes():
    model = Sequential([
        Linear(2, 4, seed=0),
        ReLU(),
        Linear(4, 3, seed=1),
    ])

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    logits = model.forward(X)

    assert logits.shape == (2, 3)

    dlogits = np.ones_like(logits)
    dX = model.backward(dlogits)

    assert dX.shape == (2, 2)


def test_sequential_collects_only_trainable_params():
    linear1 = Linear(4, 5, seed=0)
    relu = ReLU()
    linear2 = Linear(5, 3, seed=1)

    model = Sequential([
        linear1,
        relu,
        linear2,
    ])

    X = np.random.randn(6, 4)
    logits = model.forward(X)

    dlogits = np.random.randn(6, 3)
    model.backward(dlogits)

    params_and_grads = model.params_and_grads()

    expected = [
        (linear1.W, linear1.dW),
        (linear1.b, linear1.db),
        (linear2.W, linear2.dW),
        (linear2.b, linear2.db),
    ]

    assert len(params_and_grads) == len(expected)

    for (param, grad), (expected_param, expected_grad) in zip(params_and_grads, expected):
        assert param is expected_param
        assert grad is expected_grad


def test_sequential_forward_returns_output_not_none():
    model = Sequential([
        Linear(4, 3, seed=0),
    ])

    X = np.random.randn(5, 4)
    out = model.forward(X)

    assert out is not None
    assert out.shape == (5, 3)


def test_sequential_backward_returns_dx_not_none():
    model = Sequential([
        Linear(4, 3, seed=0),
    ])

    X = np.random.randn(5, 4)
    logits = model.forward(X)

    dlogits = np.random.randn(5, 3)
    dX = model.backward(dlogits)

    assert dX is not None
    assert dX.shape == X.shape
