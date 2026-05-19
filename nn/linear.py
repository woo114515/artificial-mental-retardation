"""
Linear layers for the NumPy neural network.

This module uses row-major batch convention:

    X:    (batch_size, in_features)
    W:    (in_features, out_features)
    b:    (out_features,)
    Z:    (batch_size, out_features)
    dout: (batch_size, out_features)

So the forward pass is:

    Z = X @ W + b
"""

from __future__ import annotations

import numpy as np

from .init import initialize_parameters

from .layer import layer


class Linear(layer):
    """
    Fully connected linear layer.

    Forward:
        Z = X @ W + b

    Shapes:
        X:    (batch_size, in_features)
        W:    (in_features, out_features)
        b:    (out_features,)
        Z:    (batch_size, out_features)
        dout: (batch_size, out_features)
        dW:   (in_features, out_features)
        db:   (out_features,)
        dX:   (batch_size, in_features)
    """

    def __init__(
        self,
        in_features: int | None = None,
        out_features: int | None = None,
        *,
        fan_in: int | None = None,
        fan_out: int | None = None,
        method: str = "he",
        seed: int | None = None,
        std: float = 0.01,
    ):
        if in_features is None:
            in_features = fan_in

        if out_features is None:
            out_features = fan_out

        if in_features is None:
            raise TypeError("missing required argument: 'in_features' or 'fan_in'")

        if out_features is None:
            raise TypeError("missing required argument: 'out_features' or 'fan_out'")

        self.in_features = in_features
        self.out_features = out_features

        rng = np.random.default_rng(seed)

        W, b = initialize_parameters(
            fan_in=in_features,
            fan_out=out_features,
            method=method,
            rng=rng,
            std=std,
        )

        # initialize_parameters is expected to return:
        #   W: (in_features, out_features)
        #   b: (1, out_features) or (out_features,)
        # Linear keeps the same row-major convention:
        #   W: (in_features, out_features)
        #   b: (out_features,)
        self.W = W
        self.b = b.reshape(out_features)

        self.X: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self.dX: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear transformation.

        Args:
            X: Input batch, shape (batch_size, in_features).

        Returns:
            Output batch, shape (batch_size, out_features).
        """
        if X.ndim != 2:
            raise ValueError(f"Linear.forward expected a 2D array, got shape {X.shape}.")

        if X.shape[1] != self.in_features:
            raise ValueError(
                f"Linear.forward expected input feature dimension {self.in_features}, "
                f"got {X.shape[1]}."
            )

        self.X = X
        return X @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backpropagate through the linear layer.

        Args:
            dout: Upstream gradient, shape (batch_size, out_features).

        Returns:
            Gradient with respect to input X, shape (batch_size, in_features).
        """
        if self.X is None:
            raise RuntimeError("Cannot call backward before forward.")

        if dout.ndim != 2:
            raise ValueError(f"Linear.backward expected a 2D array, got shape {dout.shape}.")

        if dout.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"Linear.backward batch size mismatch: forward batch size was {self.X.shape[0]}, "
                f"but dout batch size is {dout.shape[0]}."
            )

        if dout.shape[1] != self.out_features:
            raise ValueError(
                f"Linear.backward expected output gradient dimension {self.out_features}, "
                f"got {dout.shape[1]}."
            )

        self.dW = self.X.T @ dout
        self.db = np.sum(dout, axis=0)
        self.dX = dout @ self.W.T

        return self.dX

    def params_and_grads(self):
        """
        Return trainable parameters and their corresponding gradients.
        """
        return [
            (self.W, self.dW),
            (self.b, self.db),
        ]
