"""
Tests for data.transforms.
"""

import numpy as np
import pytest

from data.transforms import to_flat_images, to_nchw_images


def test_to_nchw_images_shape_for_mnist():
    X = np.zeros((5, 784), dtype=np.float32)

    out = to_nchw_images(X)

    assert out.shape == (5, 1, 28, 28)
    assert out.dtype == X.dtype


def test_to_nchw_images_preserves_values_order():
    X = np.arange(2 * 6).reshape(2, 6)

    out = to_nchw_images(X, channels=1, height=2, width=3)

    expected = np.array([
        [
            [
                [0, 1, 2],
                [3, 4, 5],
            ]
        ],
        [
            [
                [6, 7, 8],
                [9, 10, 11],
            ]
        ],
    ])
    np.testing.assert_array_equal(out, expected)


def test_to_flat_images_shape_and_values():
    X = np.arange(2 * 1 * 2 * 3).reshape(2, 1, 2, 3)

    out = to_flat_images(X)

    expected = np.array([
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
    ])
    assert out.shape == (2, 6)
    np.testing.assert_array_equal(out, expected)


def test_round_trip_flat_to_nchw_to_flat():
    X = np.random.randn(4, 784)

    images = to_nchw_images(X)
    flat = to_flat_images(images)

    np.testing.assert_allclose(flat, X)


def test_to_nchw_images_invalid_dimension_raises_error():
    X = np.zeros((5, 1, 28, 28))

    with pytest.raises(ValueError):
        to_nchw_images(X)


def test_to_nchw_images_invalid_feature_count_raises_error():
    X = np.zeros((5, 783))

    with pytest.raises(ValueError):
        to_nchw_images(X)


def test_to_flat_images_invalid_dimension_raises_error():
    X = np.zeros((5, 784))

    with pytest.raises(ValueError):
        to_flat_images(X)
