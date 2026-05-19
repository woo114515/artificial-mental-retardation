"""
Evaluation metrics for the NumPy MLP project.
"""

from __future__ import annotations

import numpy as np


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy from logits and integer labels.

    Args:
        logits: Array with shape (batch_size, num_classes).
        labels: Integer labels with shape (batch_size,).

    Returns:
        Scalar accuracy as a Python float.
    """
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == labels))


def evaluate(model, X: np.ndarray, y: np.ndarray, criterion=None, batch_size: int = 256) -> dict:
    """
    Run model evaluation in non-shuffled batches.

    Args:
        model: Object with a forward(X_batch) method.
        X: Input data, shape (num_samples, num_features).
        y: Integer labels, shape (num_samples,).
        criterion: Optional loss object with forward(logits, labels).
        batch_size: Number of samples per evaluation batch.

    Returns:
        {"accuracy": acc} if criterion is None.
        {"loss": avg_loss, "accuracy": acc} if criterion is provided.
    """
    num_samples = X.shape[0]
    all_preds = []
    total_loss = 0.0
    total_count = 0

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]

        logits = model.forward(X_batch)
        all_preds.append(np.argmax(logits, axis=1))

        if criterion is not None:
            batch_loss = criterion.forward(logits, y_batch)
            total_loss += batch_loss * len(y_batch)

        total_count += len(y_batch)

    preds = np.concatenate(all_preds)
    acc = float(np.mean(preds == y))

    if criterion is None:
        return {"accuracy": acc}

    return {
        "loss": float(total_loss / total_count),
        "accuracy": acc,
    }
