"""
Evaluate a saved checkpoint and visualize misclassified test samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import nn
from config import CHECKPOINT_PATH
from train import build_model_from_config, get_image_shape, prepare_data
from utils.checkpoint import load_checkpoint
from utils.metrics import evaluate as evaluate_metrics
from utils.prediction_visualization import plot_error_predictions, softmax


def predict(model, X: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run batched prediction and return logits, predicted labels, and confidences.
    """
    logits_batches = []

    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        logits_batches.append(model.forward(X[start:end]))

    logits = np.vstack(logits_batches)
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    return logits, preds, confidences


def metadata_title(metadata: dict) -> str:
    """
    Build a compact hyperparameter title for prediction figures.
    """
    title_keys = [
        "MODEL_TYPE",
        "OPTIMIZER",
        "ACTIVATION",
        "LEARNING_RATE",
        "BATCH_SIZE",
        "HIDDEN_DIMS",
        "WEIGHT_INIT",
        "RANDOM_SEED",
    ]

    if metadata.get("MODEL_TYPE") == "cnn":
        title_keys.extend(["CNN_OUT_CHANNELS", "CNN_KERNEL_SIZE"])

    parts = []
    for key in title_keys:
        if key in metadata:
            parts.append(f"{key}={metadata[key]}")

    return ", ".join(parts) if parts else "checkpoint metadata unavailable"


def print_metadata(metadata: dict) -> None:
    """
    Print checkpoint hyperparameters and training metadata.
    """
    if not metadata:
        print("Checkpoint metadata: unavailable")
        return

    print("Checkpoint metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MNIST checkpoint and visualize misclassified samples."
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_PATH,
        help=f"Checkpoint path to load. Default: {CHECKPOINT_PATH}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used during evaluation.",
    )
    parser.add_argument(
        "--num-errors",
        type=int,
        default=9,
        help="Number of misclassified examples to visualize.",
    )
    parser.add_argument(
        "--output",
        default="experiments/predictions/error/error_predictions.png",
        help="Path for the misclassification visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from data.mnist import X_test, X_train, X_val, y_test

    image_shape = get_image_shape("mnist")
    input_dim = X_test.shape[1]
    _, _, X_test_model = prepare_data(X_train, X_val, X_test, image_shape=image_shape)

    model = build_model_from_config(input_dim=input_dim, image_shape=image_shape)
    metadata = load_checkpoint(model, args.checkpoint)
    print(f"Loaded checkpoint: {Path(args.checkpoint)}")
    print_metadata(metadata)

    criterion = nn.SoftmaxCrossEntropyLoss()
    metrics = evaluate_metrics(
        model,
        X_test_model,
        y_test,
        criterion=criterion,
        batch_size=args.batch_size,
    )

    _, preds, confidences = predict(model, X_test_model, batch_size=args.batch_size)
    error_indices = np.where(preds != y_test)[0]
    error_rate = len(error_indices) / len(y_test)

    print(
        "Test metrics: "
        f"loss={metrics['loss']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"errors={len(error_indices)}/{len(y_test)} "
        f"({error_rate:.2%})"
    )

    output_path = plot_error_predictions(
        X_display=X_test,
        y=y_test,
        preds=preds,
        confidences=confidences,
        error_indices=error_indices,
        save_path=args.output,
        num_samples=args.num_errors,
        image_shape=(image_shape[1], image_shape[2]),
        title=metadata_title(metadata),
    )
    print(f"Saved error visualization to {output_path}")


if __name__ == "__main__":
    main()
