"""
Tests for nn.optimizers.Momentum.

Momentum update rule:
    v = momentum * v - lr * grad
    param = param + v

Important:
    Optimizer must update parameters in-place.
"""

import numpy as np

from nn.optimizers import Momentum


def test_momentum_first_step_matches_sgd():
    optimizer = Momentum(lr=0.1, momentum=0.9)

    param = np.array([1.0, 2.0, 3.0])
    grad = np.array([0.1, 0.2, 0.3])

    optimizer.step([(param, grad)])

    expected = np.array([0.99, 1.98, 2.97])
    np.testing.assert_allclose(param, expected)


def test_momentum_accumulates_velocity_across_steps():
    optimizer = Momentum(lr=0.1, momentum=0.9)

    param = np.array([1.0, 2.0])
    grad = np.array([0.5, 1.0])

    optimizer.step([(param, grad)])
    optimizer.step([(param, grad)])

    expected = np.array([0.855, 1.71])
    np.testing.assert_allclose(param, expected)


def test_momentum_updates_multiple_parameters_independently():
    optimizer = Momentum(lr=0.1, momentum=0.9)

    W = np.array([1.0, 2.0])
    b = np.array([3.0])

    dW = np.array([0.5, 1.0])
    db = np.array([0.2])

    optimizer.step([
        (W, dW),
        (b, db),
    ])

    np.testing.assert_allclose(W, np.array([0.95, 1.9]))
    np.testing.assert_allclose(b, np.array([2.98]))

    assert id(W) in optimizer.velocities
    assert id(b) in optimizer.velocities


def test_momentum_updates_in_place():
    optimizer = Momentum(lr=0.1, momentum=0.9)

    param = np.array([1.0, 2.0])
    grad = np.array([1.0, 1.0])

    original_id = id(param)

    optimizer.step([(param, grad)])

    assert id(param) == original_id
    np.testing.assert_allclose(param, np.array([0.9, 1.9]))


def test_momentum_skips_none_gradient():
    optimizer = Momentum(lr=0.1, momentum=0.9)

    param = np.array([1.0, 2.0])

    optimizer.step([(param, None)])

    np.testing.assert_allclose(param, np.array([1.0, 2.0]))
    assert id(param) not in optimizer.velocities

