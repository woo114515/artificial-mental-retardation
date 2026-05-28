from __future__ import annotations

import numpy as np

from .layer import layer


class Flatten(layer):
    """
    Flatten all non-batch dimensions into one feature dimension.

    Shapes:
        X:    (batch_size, ...)
        out:  (batch_size, flattened_features)
        dout: (batch_size, flattened_features)
        dX:   same shape as X
    """

    def __init__(self):
        self.input_shape: tuple[int, ...] | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten input while preserving the batch dimension.
        """
        if X.ndim < 2:
            raise ValueError(f"Flatten.forward expected at least a 2D array, got shape {X.shape}.")

        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Restore gradient to the input shape saved during forward().
        """
        if self.input_shape is None:
            raise RuntimeError("Cannot call backward before forward.")

        expected_shape = (self.input_shape[0], int(np.prod(self.input_shape[1:])))
        if dout.shape != expected_shape:
            raise ValueError(
                f"Flatten.backward expected dout shape {expected_shape}, got {dout.shape}."
            )

        return dout.reshape(self.input_shape)

    def params_and_grads(self):
        """
        Flatten has no trainable parameters.
        """
        return []
