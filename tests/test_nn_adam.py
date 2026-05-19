"""
Tests for nn.optimizers.Adam.

Adam update rule:
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(v_hat) + eps)

Important:
    Optimizer must update parameters in-place.
"""

import numpy as np

from nn.optimizers import Adam


def test_adam_first_step_updates_parameter():
    optimizer = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

    param = np.array([1.0, 2.0])
    grad = np.array([0.5, 1.0])

    optimizer.step([(param, grad)])

    expected = np.array([0.9, 1.9])
    np.testing.assert_allclose(param, expected, rtol=1e-7, atol=1e-7)


def test_adam_accumulates_moments_across_steps():
    optimizer = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

    param = np.array([1.0, 2.0])
    grad = np.array([0.5, 1.0])

    optimizer.step([(param, grad)])
    optimizer.step([(param, grad)])

    expected = np.array([0.8, 1.8])
    np.testing.assert_allclose(param, expected, rtol=1e-7, atol=1e-7)
    assert optimizer.t == 2


def test_adam_updates_multiple_parameters_independently():
    optimizer = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

    W = np.array([1.0, 2.0])
    b = np.array([3.0])

    dW = np.array([0.5, 1.0])
    db = np.array([0.2])

    optimizer.step([
        (W, dW),
        (b, db),
    ])

    np.testing.assert_allclose(W, np.array([0.9, 1.9]), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(b, np.array([2.9]), rtol=1e-7, atol=1e-7)

    assert id(W) in optimizer.m
    assert id(W) in optimizer.v
    assert id(b) in optimizer.m
    assert id(b) in optimizer.v


def test_adam_updates_in_place():
    optimizer = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

    param = np.array([1.0, 2.0])
    grad = np.array([1.0, 1.0])

    original_id = id(param)

    optimizer.step([(param, grad)])

    assert id(param) == original_id
    np.testing.assert_allclose(param, np.array([0.9, 1.9]), rtol=1e-7, atol=1e-7)


def test_adam_skips_none_gradient():
    optimizer = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

    param = np.array([1.0, 2.0])

    optimizer.step([(param, None)])

    np.testing.assert_allclose(param, np.array([1.0, 2.0]))
    assert id(param) not in optimizer.m
    assert id(param) not in optimizer.v
    assert optimizer.t == 1

