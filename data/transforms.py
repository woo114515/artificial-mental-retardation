"""
Data shape transforms shared by MLP and CNN experiments.
"""

from __future__ import annotations

import numpy as np


def to_nchw_images(
    X: np.ndarray,
    channels: int = 1,
    height: int = 28,
    width: int = 28,
) -> np.ndarray:
    """
    Convert flat image vectors to NCHW image tensors.

    Args:
        X: Input images with shape (num_samples, channels * height * width).
        channels: Number of image channels.
        height: Image height.
        width: Image width.

    Returns:
        Images with shape (num_samples, channels, height, width).
    """
    if X.ndim != 2:
        raise ValueError(f"to_nchw_images expected a 2D array, got shape {X.shape}.")

    expected_features = channels * height * width
    if X.shape[1] != expected_features:
        raise ValueError(
            f"to_nchw_images expected feature dimension {expected_features}, got {X.shape[1]}."
        )

    return X.reshape(X.shape[0], channels, height, width)


def to_flat_images(X: np.ndarray) -> np.ndarray:
    """
    Convert NCHW image tensors back to row-major flat vectors.

    Args:
        X: Input images with shape (num_samples, channels, height, width).

    Returns:
        Flat images with shape (num_samples, channels * height * width).
    """
    if X.ndim != 4:
        raise ValueError(f"to_flat_images expected a 4D array, got shape {X.shape}.")

    return X.reshape(X.shape[0], -1)
