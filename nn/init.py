'''
参数初始化。
'''

from __future__ import annotations

from typing import Optional, Tuple 
# 导入类型标注工具。Optional[X] 表示 X 或 None
# Tuple[A, B] 表示返回一个元组，里面有两个元素。

import numpy as np


__all__ = ["initialize_parameters", "initialize_parameters_conv2d"] # 这个变量用于控制： from init import * 时，哪些函数会被导入。


def initialize_parameters(
    fan_in: int,
    fan_out: int,
    method: str = "he",
    rng: Optional[np.random.Generator] = None,
    std: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize weights and biases for a dense layer.

    Parameters
    ----------
    fan_in:
        Number of input features.

    fan_out:
        Number of output features.

    method:
        Initialization method.
        Supported values:
            - "normal"
            - "xavier"
            - "he"

    rng:
        NumPy random generator. If None, a new default generator is created.

    std:
        Standard deviation used only when method == "normal".

    Returns
    -------
    W:
        Weight matrix of shape (fan_in, fan_out).

    b:
        Bias vector of shape (1, fan_out).
    """
    _validate_dimensions(fan_in, fan_out)
    _validate_method(method)

    if rng is None:
        rng = np.random.default_rng()

    weight_std = _compute_weight_std(
        fan_in=fan_in,
        fan_out=fan_out,
        method=method,
        std=std,
    )

    W = rng.normal(
        loc=0.0,
        scale=weight_std,
        size=(fan_in, fan_out),
    )
    # 正态分布

    b = np.zeros((1, fan_out))

    return W, b


def initialize_parameters_conv2d(
    in_channel: int,
    out_channel: int,
    kernel_size: int,
    method: str = "he",
    rng: Optional[np.random.Generator] = None,
    std: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize weights and biases for a conv2D layer.

    Parameters
    ----------
    in_channels:
        Number of input channels.

    out_channels:
        Number of convolution filters.

    kernel_size
        Height and width of each square convolution kernel.

    method:
        Initialization method.
        Supported values:
            - "normal"
            - "xavier"
            - "he"

    rng:
        NumPy random generator. If None, a new default generator is created.

    std:
        Standard deviation used only when method == "normal".

    Returns
    -------
    W:
        Weight matrix of shape (in_channel, out_channel, kernel_size, kernel_size).

    b:
        Bias vector of shape (out_channel,).
    """
    _validate_method(method)

    fan_in = in_channel * kernel_size * kernel_size
    fan_out = out_channel * kernel_size * kernel_size

    if rng is None:
        rng = np.random.default_rng()

    weight_std = _compute_weight_std(
        fan_in=fan_in,
        fan_out=fan_out,
        method=method,
        std=std,
    )

    W = rng.normal(
        loc=0.0,
        scale=weight_std,
        size=(out_channel, in_channel, kernel_size, kernel_size),
    )
    # 正态分布

    b = np.zeros(out_channel)

    return W, b


def _compute_weight_std(
    fan_in: int,
    fan_out: int,
    method: str,
    std: float,
) -> float:
    """
    Compute the standard deviation for weight initialization.

    This function is internal. Other modules should call
    initialize_parameters() instead.
    """
    if method == "normal":
        return std

    if method == "xavier":
        return np.sqrt(2.0 / (fan_in + fan_out))

    if method == "he":
        return np.sqrt(2.0 / fan_in)

    raise ValueError(f"Unknown initialization method: {method}")


def _validate_dimensions(fan_in: int, fan_out: int) -> None:
    """
    Validate layer dimensions.
    """
    if not isinstance(fan_in, int):
        raise TypeError(f"fan_in must be int, got {type(fan_in)}")

    if not isinstance(fan_out, int):
        raise TypeError(f"fan_out must be int, got {type(fan_out)}")

    if fan_in <= 0:
        raise ValueError(f"fan_in must be positive, got {fan_in}")

    if fan_out <= 0:
        raise ValueError(f"fan_out must be positive, got {fan_out}")


def _validate_method(method: str) -> None:
    """
    Validate initialization method.
    """
    supported_methods = {"normal", "xavier", "he"}

    if method not in supported_methods:
        raise ValueError(
            f"Unsupported initialization method: {method}. "
            f"Expected one of {supported_methods}."
        )