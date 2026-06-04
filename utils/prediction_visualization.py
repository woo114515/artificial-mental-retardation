"""
Visualization helpers for model predictions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def plot_test_predictions(
    model,
    X_display: np.ndarray,
    X_model: np.ndarray,
    y: np.ndarray,
    save_path: str = "experiments/predictions/test_predictions.png",
    num_samples: int = 16,
    seed: int = 42,
    image_shape: tuple[int, int] = (28, 28),
) -> Path:
    """
    Plot test images with true labels, predicted labels, and confidence.

    Args:
        model: Trained model exposing forward().
        X_display: Flat image data for plotting, shape (num_samples_total, H * W).
        X_model: Model input data, e.g. (N, 784) for MLP or (N, 1, 28, 28) for CNN.
        y: Integer labels, shape (num_samples_total,).
        save_path: Output image path.
        num_samples: Number of samples to visualize.
        seed: Random seed for reproducible sample selection.
        image_shape: Shape used to display flat images.

    Returns:
        Path to the saved visualization.
    """
    if X_display.shape[0] != X_model.shape[0] or X_display.shape[0] != y.shape[0]:
        raise ValueError("X_display, X_model, and y must contain the same number of samples.")

    if X_display.ndim != 2:
        raise ValueError(f"X_display must be flat 2D image data, got shape {X_display.shape}.")

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    rng = np.random.default_rng(seed)
    sample_count = min(num_samples, X_display.shape[0])
    indices = rng.choice(X_display.shape[0], size=sample_count, replace=False)

    logits = model.forward(X_model[indices])
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    cols = int(np.ceil(np.sqrt(sample_count)))
    rows = int(np.ceil(sample_count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.array(axes).reshape(-1)

    for axis_index, ax in enumerate(axes):
        ax.axis("off")

        if axis_index >= sample_count:
            continue

        sample_index = indices[axis_index]
        image = X_display[sample_index].reshape(image_shape)

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"y={int(y[sample_index])}, pred={int(preds[axis_index])}\n"
            f"conf={confidences[axis_index]:.2f}",
            fontsize=9,
        )

    fig.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_error_predictions(
    X_display: np.ndarray,
    y: np.ndarray,
    preds: np.ndarray,
    confidences: np.ndarray,
    error_indices: np.ndarray,
    save_path: str = "experiments/predictions/error/error_predictions.png",
    num_samples: int = 9,
    image_shape: tuple[int, int] = (28, 28),
    title: str | None = None,
) -> Path:
    """
    Plot misclassified test images.

    Args:
        X_display: Flat image data for plotting, shape (num_samples_total, H * W).
        y: Integer labels, shape (num_samples_total,).
        preds: Predicted labels, shape (num_samples_total,).
        confidences: Prediction confidences, shape (num_samples_total,).
        error_indices: Indices where predictions are incorrect.
        save_path: Output image path.
        num_samples: Maximum number of errors to visualize.
        image_shape: Shape used to display flat images.
        title: Optional figure title, e.g. model hyperparameters.

    Returns:
        Path to the saved visualization.
    """
    if X_display.ndim != 2:
        raise ValueError(f"X_display must be flat 2D image data, got shape {X_display.shape}.")

    num_total = X_display.shape[0]
    if y.shape[0] != num_total or preds.shape[0] != num_total or confidences.shape[0] != num_total:
        raise ValueError("X_display, y, preds, and confidences must contain the same samples.")

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    sample_count = min(num_samples, len(error_indices))
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sample_count == 0:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No misclassified samples found.", ha="center", va="center")
        if title:
            fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    selected_indices = error_indices[:sample_count]
    cols = int(np.ceil(np.sqrt(sample_count)))
    rows = int(np.ceil(sample_count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    if title:
        fig.suptitle(title, fontsize=9)

    for axis_index, ax in enumerate(axes):
        ax.axis("off")

        if axis_index >= sample_count:
            continue

        sample_index = selected_indices[axis_index]
        image = X_display[sample_index].reshape(image_shape)

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"idx={int(sample_index)}\n"
            f"y={int(y[sample_index])}, pred={int(preds[sample_index])}\n"
            f"conf={confidences[sample_index]:.2f}",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path
