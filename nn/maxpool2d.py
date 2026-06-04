from __future__ import annotations

import numpy as np

from .layer import layer


class MaxPool2D(layer):
    """
    2D max pooling layer using NCHW format.

    Shapes:
        X:    (batch_size, channels, H_in, W_in)
        out:  (batch_size, channels, H_out, W_out)
        dout: (batch_size, channels, H_out, W_out)
        dX:   (batch_size, channels, H_in, W_in)
    """

    def __init__(self, kernel_size: int = 2, stride: int | None = None):
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")

        if stride is None:
            stride = kernel_size

        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")

        self.kernel_size = kernel_size
        self.stride = stride

        self.X: np.ndarray | None = None
        self.max_indices: np.ndarray | None = None
        self.H_out: int | None = None
        self.W_out: int | None = None
        self.dX: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply max pooling.
        """
        if X.ndim != 4:
            raise ValueError(f"MaxPool2D.forward expected a 4D array, got shape {X.shape}.")

        N, C, H, W = X.shape
        K = self.kernel_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"Invalid output size ({H_out}, {W_out}). Check kernel_size and stride."
            )

        out = np.empty((N, C, H_out, W_out), dtype=X.dtype)
        max_indices = np.empty((N, C, H_out, W_out), dtype=np.int64)

        for h_out in range(H_out):
            h_start = h_out * S
            h_end = h_start + K

            for w_out in range(W_out):
                w_start = w_out * S
                w_end = w_start + K

                window = X[:, :, h_start:h_end, w_start:w_end]
                window_flat = window.reshape(N, C, -1)

                out[:, :, h_out, w_out] = np.max(window_flat, axis=2)
                max_indices[:, :, h_out, w_out] = np.argmax(window_flat, axis=2)

        self.X = X
        self.max_indices = max_indices
        self.H_out = H_out
        self.W_out = W_out

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backpropagate through max pooling.
        """
        if self.X is None or self.max_indices is None:
            raise RuntimeError("Cannot call backward before forward.")

        if dout.ndim != 4:
            raise ValueError(f"MaxPool2D.backward expected a 4D array, got shape {dout.shape}.")

        N, C, H, W = self.X.shape
        K = self.kernel_size
        S = self.stride

        expected_shape = (N, C, self.H_out, self.W_out)
        if dout.shape != expected_shape:
            raise ValueError(
                f"MaxPool2D.backward expected dout shape {expected_shape}, got {dout.shape}."
            )

        dX = np.zeros_like(self.X)

        for h_out in range(self.H_out):
            h_start = h_out * S

            for w_out in range(self.W_out):
                w_start = w_out * S
                indices = self.max_indices[:, :, h_out, w_out]

                row_offsets = indices // K
                col_offsets = indices % K

                n_idx = np.arange(N)[:, None]
                c_idx = np.arange(C)[None, :]

                dX[
                    n_idx,
                    c_idx,
                    h_start + row_offsets,
                    w_start + col_offsets,
                ] += dout[:, :, h_out, w_out]

        self.dX = dX
        return dX

    def params_and_grads(self):
        """
        MaxPool2D has no trainable parameters.
        """
        return []
