from __future__ import annotations

import numpy as np

from .layer import layer
from .init import initialize_parameters_conv2d


class Conv2D(layer):
    """
    2D convolution layer using NCHW format.

    Forward:
        Z = conv2d(X, W) + b

    Shapes:
        X:    (batch_size, in_channels, H_in, W_in)
        W:    (out_channels, in_channels, kernel_size, kernel_size)
        b:    (out_channels,)
        Z:    (batch_size, out_channels, H_out, W_out)

        dout: (batch_size, out_channels, H_out, W_out)
        dW:   (out_channels, in_channels, kernel_size, kernel_size)
        db:   (out_channels,)
        dX:   (batch_size, in_channels, H_in, W_in)

    Output size:
        H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        W_out = (W_in + 2 * padding - kernel_size) // stride + 1
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        method: str = "he",
        seed: int | None = None,
        std: float = 0.01
    ):
        """
        Initialize a 2D convolution layer.

        Args:
            in_channels:
                Number of input channels.

            out_channels:
                Number of convolution filters.

            kernel_size:
                Height and width of each square convolution kernel.

            stride:
                Step size of the convolution window.

            padding:
                Number of zeros padded to both sides of height and width.

            method:
                Weight initialization method.

            seed:
                Random seed for reproducible initialization.
        """
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        rng = np.random.default_rng(seed)

        W, b = initialize_parameters_conv2d(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            method=method,
            rng=rng,
            std=std,
            )
        
        self.W = W
        self.b = b

        self.X: np.ndarray | None = None
        self.X_col: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self.dX: np.ndarray | None = None
        self.H_out: int | None = None
        self.W_out: int | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass of the 2D convolution layer.

        Forward:
            Z = conv2d(X, W) + b

        Shapes:
            X:        (batch_size, in_channels, H_in, W_in)
            W:        (out_channels, in_channels, kernel_size, kernel_size)
            b:        (out_channels,)
            Z:        (batch_size, out_channels, H_out, W_out)

        Output size:
            H_out = (H_in + 2 * padding - kernel_size) // stride + 1
            W_out = (W_in + 2 * padding - kernel_size) // stride + 1

        Cache:
            X and X_col are saved for backward().
        """
        if X.ndim != 4:
            raise ValueError(f"conv2D.forward expected a 4D array, got shape {X.shape}.")

        N, C_in, H_in, W_in = X.shape
        C_out, C_weight, K_h, K_w = self.W.shape
        P = self.padding
        S = self.stride

        if C_in != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, got {C_in}."
            )

        if C_weight != C_in:
            raise ValueError(
                f"Weight expects {C_weight} input channels, but X has {C_in}."
            )

        if K_h != K_w:
            raise ValueError("Only square kernels are currently supported.")

        H_out = (H_in + 2 * P - K_h) // S + 1
        W_out = (W_in + 2 * P - K_w) // S + 1

        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"Invalid output size ({H_out}, {W_out}). Check kernel_size, stride, and padding."
            )

        X_col = _im2col_sliding(X=X, kernel_size=K_h, stride=S, padding=P)

        self.X = X
        self.X_col = X_col
        self.H_out = H_out
        self.W_out = W_out

        W_col = self.W.reshape(self.out_channels, -1)
        Z_col = X_col @ W_col.T + self.b

        Z = Z_col.reshape(N, H_out, W_out, C_out)
        Z = Z.transpose(0, 3, 1, 2)
        return Z

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backpropagate through the 2D convolution layer.

        Forward:
            X_col = im2col(X)
            W_col = W.reshape(out_channels, -1)
            Z_col = X_col @ W_col.T + b

        Backward:
            dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, out_channels)

            dW_col = dout_col.T @ X_col
            db     = sum(dout_col, axis=0)
            dX_col = dout_col @ W_col

            dW = dW_col.reshape(W.shape)
            dX = col2im(dX_col)

        Shapes:
            dout:     (batch_size, out_channels, H_out, W_out)
            X_col:    (batch_size * H_out * W_out,
                       in_channels * kernel_size * kernel_size)
            W_col:    (out_channels,
                       in_channels * kernel_size * kernel_size)
            dout_col: (batch_size * H_out * W_out, out_channels)
            dW:       (out_channels, in_channels, kernel_size, kernel_size)
            db:       (out_channels,)
            dX:       (batch_size, in_channels, H_in, W_in)

        Returns:
            Gradient with respect to input X, shape (batch_size, in_channels, H_in, W_in).
        """
        if self.X is None or self.X_col is None:
            raise RuntimeError("Cannot call backward before forward.")

        if dout.ndim != 4:
            raise ValueError(f"Conv2D.backward expected a 4D array, got shape {dout.shape}.")

        N, C_out, H_out, W_out = dout.shape
        if C_out != self.out_channels:
            raise ValueError(
                f"Conv2D.backward expected {self.out_channels} output channels, got {C_out}."
            )

        if H_out != self.H_out or W_out != self.W_out:
            raise ValueError(
                f"Conv2D.backward expected spatial shape ({self.H_out}, {self.W_out}), "
                f"got ({H_out}, {W_out})."
            )

        dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, C_out)
        W_col = self.W.reshape(self.out_channels, -1)

        dW_col = dout_col.T @ self.X_col
        self.dW = dW_col.reshape(self.W.shape)
        self.db = np.sum(dout, axis=(0, 2, 3))

        dX_col = dout_col @ W_col
        self.dX = _col2im_sliding(
            dX_col=dX_col,
            X_shape=self.X.shape,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            H_out=H_out,
            W_out=W_out,
        )

        return self.dX

    def params_and_grads(self):
        """
        Return trainable parameters and their corresponding gradients.
        """
        return [
            (self.W, self.dW),
            (self.b, self.db),
        ]



        
def _im2col_sliding(X: np.ndarray, kernel_size: int, stride: int, padding: int) -> np.ndarray:
    N, C, H, W = X.shape
    K = kernel_size
    P = padding
    S = stride

    X_padded = np.pad(
        X,
        pad_width=((0, 0), (0, 0), (P, P), (P, P)),
        mode="constant",
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        X_padded,
        window_shape=(K, K),
        axis=(2, 3),
    )

    # windows shape:
    # (N, C, H_out_raw, W_out_raw, K, K)

    windows = windows[:, :, ::S, ::S, :, :]

    # Move dimensions into:
    # (N, H_out, W_out, C, K, K)
    windows = windows.transpose(0, 2, 3, 1, 4, 5)

    N, H_out, W_out, C, K_h, K_w = windows.shape

    X_col = windows.reshape(N * H_out * W_out, C * K_h * K_w)

    return X_col


def _col2im_sliding(
    dX_col: np.ndarray,
    X_shape: tuple[int, int, int, int],
    kernel_size: int,
    stride: int,
    padding: int,
    H_out: int,
    W_out: int,
) -> np.ndarray:
    N, C, H, W = X_shape
    K = kernel_size
    S = stride
    P = padding

    dX_padded = np.zeros((N, C, H + 2 * P, W + 2 * P), dtype=dX_col.dtype)
    dX_windows = dX_col.reshape(N, H_out, W_out, C, K, K)

    for h_out in range(H_out):
        h_start = h_out * S
        h_end = h_start + K

        for w_out in range(W_out):
            w_start = w_out * S
            w_end = w_start + K

            dX_padded[:, :, h_start:h_end, w_start:w_end] += dX_windows[:, h_out, w_out]

    if P == 0:
        return dX_padded

    return dX_padded[:, :, P:-P, P:-P]
