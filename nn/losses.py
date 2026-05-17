"""
最后一层 loss。

前向：
    根据 logits 和真实标签计算 softmax cross entropy loss

反向：
    计算 loss 对 logits 的梯度
"""

from __future__ import annotations

import numpy as np


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.logits = None
        self.labels = None
        self.probs = None
        self.loss = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        logits: shape = (batch_size, num_classes)
        labels: shape = (batch_size,)
                每个元素是整数类别，例如 0, 1, ..., 9

        return:
            scalar loss
        """

        if logits.ndim != 2:
            raise ValueError("logits must be a 2D array with shape (batch_size, num_classes).")

        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array with shape (batch_size,).")

        batch_size, num_classes = logits.shape

        if labels.shape[0] != batch_size:
            raise ValueError("labels length must match logits batch size.")

        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError("labels contain class index out of range.")

        self.logits = logits
        self.labels = labels

        # Numerical stability:
        # softmax(logits) == softmax(logits - max(logits))
        shifted = logits - np.max(logits, axis=1, keepdims=True)

        exp_scores = np.exp(shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_probs = self.probs[np.arange(batch_size), labels]

        # Add epsilon to avoid log(0)
        per_sample_loss = -np.log(correct_probs + 1e-12)

        self.loss = np.mean(per_sample_loss)

        return self.loss

    def backward(self) -> np.ndarray:
        """
        return:
            dlogits, shape = (batch_size, num_classes)
        """

        if self.logits is None:
            raise RuntimeError("Cannot call backward before forward.")

        batch_size = self.logits.shape[0]

        dlogits = self.probs.copy()
        dlogits[np.arange(batch_size), self.labels] -= 1
        dlogits /= batch_size

        return dlogits