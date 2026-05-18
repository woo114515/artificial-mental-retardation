"""
Tests for nn.optimizers.SGD.

SGD update rule:
    param = param - lr * grad

Important:
    Optimizer must update parameters in-place.
"""

import numpy as np

from nn.optimizers import SGD


def test_sgd_updates_single_parameter():
    optimizer = SGD(lr=0.1)

    param = np.array([1.0, 2.0, 3.0])
    grad = np.array([0.1, 0.2, 0.3])

    optimizer.step([(param, grad)])

    expected = np.array([0.99, 1.98, 2.97])

    np.testing.assert_allclose(param, expected)


def test_sgd_updates_multiple_parameters():
    optimizer = SGD(lr=0.01)

    W = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    b = np.array([1.0, 2.0])

    dW = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    db = np.array([0.5, 0.6])

    optimizer.step([
        (W, dW),
        (b, db),
    ])

    expected_W = np.array([
        [0.999, 1.998],
        [2.997, 3.996],
    ])

    expected_b = np.array([0.995, 1.994])

    np.testing.assert_allclose(W, expected_W)
    np.testing.assert_allclose(b, expected_b)


def test_sgd_updates_in_place():
    optimizer = SGD(lr=0.1)

    param = np.array([1.0, 2.0, 3.0])
    grad = np.array([1.0, 1.0, 1.0])

    original_id = id(param)

    optimizer.step([(param, grad)])

    assert id(param) == original_id

    expected = np.array([0.9, 1.9, 2.9])
    np.testing.assert_allclose(param, expected)


def test_sgd_skips_none_gradient():
    optimizer = SGD(lr=0.1)

    param = np.array([1.0, 2.0, 3.0])

    optimizer.step([(param, None)])

    expected = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(param, expected)


def test_sgd_accepts_params_and_grads_as_tuples():
    optimizer = SGD(lr=0.1)

    W = np.array([1.0, 2.0])
    dW = np.array([0.5, 1.0])

    params_and_grads = [
        (W, dW),
    ]

    optimizer.step(params_and_grads)

    expected = np.array([0.95, 1.9])

    np.testing.assert_allclose(W, expected)