"""
tests/test_init.py

Unit tests for nn/init.py.

These tests verify:
1. returned parameter shapes
2. bias initialization
3. reproducibility with fixed rng seed
4. normal / Xavier / He standard deviation logic
5. input validation
"""

import numpy as np
import pytest

from nn.init import initialize_parameters


def test_initialize_parameters_shapes():
    fan_in = 784
    fan_out = 128

    rng = np.random.default_rng(42)
    W, b = initialize_parameters(
        fan_in=fan_in,
        fan_out=fan_out,
        method="he",
        rng=rng,
    )

    assert W.shape == (fan_in, fan_out)
    assert b.shape == (1, fan_out)


def test_bias_is_initialized_to_zero():
    rng = np.random.default_rng(42)
    _, b = initialize_parameters(
        fan_in=10,
        fan_out=5,
        method="he",
        rng=rng,
    )

    assert np.allclose(b, 0.0)


def test_reproducible_with_same_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    W1, b1 = initialize_parameters(
        fan_in=20,
        fan_out=10,
        method="xavier",
        rng=rng1,
    )

    W2, b2 = initialize_parameters(
        fan_in=20,
        fan_out=10,
        method="xavier",
        rng=rng2,
    )

    assert np.allclose(W1, W2)
    assert np.allclose(b1, b2)


def test_normal_initialization_uses_given_std():
    fan_in = 1000
    fan_out = 500
    std = 0.01

    rng = np.random.default_rng(42)
    W, _ = initialize_parameters(
        fan_in=fan_in,
        fan_out=fan_out,
        method="normal",
        rng=rng,
        std=std,
    )

    assert np.isclose(W.mean(), 0.0, atol=1e-3)
    assert np.isclose(W.std(), std, rtol=0.1)


def test_xavier_initialization_std():
    fan_in = 1000
    fan_out = 500

    expected_std = np.sqrt(2.0 / (fan_in + fan_out))

    rng = np.random.default_rng(42)
    W, _ = initialize_parameters(
        fan_in=fan_in,
        fan_out=fan_out,
        method="xavier",
        rng=rng,
    )

    assert np.isclose(W.mean(), 0.0, atol=1e-3)
    assert np.isclose(W.std(), expected_std, rtol=0.1)


def test_he_initialization_std():
    fan_in = 1000
    fan_out = 500

    expected_std = np.sqrt(2.0 / fan_in)

    rng = np.random.default_rng(42)
    W, _ = initialize_parameters(
        fan_in=fan_in,
        fan_out=fan_out,
        method="he",
        rng=rng,
    )

    assert np.isclose(W.mean(), 0.0, atol=1e-3)
    assert np.isclose(W.std(), expected_std, rtol=0.1)


def test_invalid_method_raises_error():
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        initialize_parameters(
            fan_in=10,
            fan_out=5,
            method="invalid",
            rng=rng,
        )


def test_invalid_fan_in_raises_error():
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        initialize_parameters(
            fan_in=0,
            fan_out=5,
            method="he",
            rng=rng,
        )


def test_invalid_fan_out_raises_error():
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        initialize_parameters(
            fan_in=10,
            fan_out=0,
            method="he",
            rng=rng,
        )


def test_non_integer_dimensions_raise_error():
    rng = np.random.default_rng(42)

    with pytest.raises(TypeError):
        initialize_parameters(
            fan_in=10.5,
            fan_out=5,
            method="he",
            rng=rng,
        )

    with pytest.raises(TypeError):
        initialize_parameters(
            fan_in=10,
            fan_out=5.5,
            method="he",
            rng=rng,
        )