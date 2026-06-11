"""
Dataset interface smoke experiments.

This script runs the same small MLP training pipeline on MNIST,
Fashion-MNIST, and CIFAR-10. The goal is not to find the best model, but to
verify that all dataset modules expose the same row-major interface:

    X_train, y_train, X_val, y_val, X_test, y_test

Results are saved in:

    experiments/dataset_interfaces/<dataset>/

Each dataset folder contains:
    - config.json
    - results.json
    - loss_curve.png
    - accuracy_curve.png

The summary is saved to:
    experiments/dataset_interfaces_summary.csv
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import nn
from data.dataloader import create_mini_batches
from utils.metrics import evaluate


EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENTS_DIR / "dataset_interfaces"
SUMMARY_PATH = EXPERIMENTS_DIR / "dataset_interfaces_summary.csv"

DATASETS = {
    "mnist": "data.mnist",
    "fashionmnist": "data.fashionmnist",
    "cifar10": "data.cifar10",
}

NUM_CLASSES = 10
HIDDEN_DIMS = [64]
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 0.01
WEIGHT_INIT = "he"
RANDOM_SEED = 42
TRAIN_LIMIT = 5000
VAL_LIMIT = 1000
TEST_LIMIT = 1000


def load_dataset(dataset_name: str):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {', '.join(DATASETS)}")

    try:
        module = importlib.import_module(DATASETS[dataset_name])
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Failed to load dataset '{dataset_name}'. "
            "Please check that its raw data files exist under data/raw/."
        ) from exc

    return (
        module.X_train,
        module.y_train,
        module.X_val,
        module.y_val,
        module.X_test,
        module.y_test,
    )


def take_subset(X: np.ndarray, y: np.ndarray, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(X):
        return X, y
    return X[:limit], y[:limit]


def build_mlp(input_dim: int, hidden_dims: list[int], seed: int):
    layer_dims = [input_dim] + hidden_dims + [NUM_CLASSES]
    layers = []

    for i in range(len(layer_dims) - 1):
        layers.append(
            nn.Linear(
                layer_dims[i],
                layer_dims[i + 1],
                method=WEIGHT_INIT,
                seed=seed + i,
            )
        )
        if i < len(layer_dims) - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(layers)


def train_one_epoch(model, criterion, optimizer, X: np.ndarray, y: np.ndarray):
    total_loss = 0.0
    total_count = 0

    for X_batch, y_batch in create_mini_batches(X, y, batch_size=BATCH_SIZE):
        logits = model.forward(X_batch)
        loss = criterion.forward(logits, y_batch)
        model.backward(criterion.backward())
        optimizer.step(model.params_and_grads())

        total_loss += loss * len(y_batch)
        total_count += len(y_batch)

    return float(total_loss / total_count)


def save_curves(history: dict, save_dir: Path, title_suffix: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title(f"Loss Curve ({title_suffix})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_dir / "loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title(f"Accuracy Curve ({title_suffix})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(save_dir / "accuracy_curve.png")
    plt.close()


def run_one_dataset(
    dataset_name: str,
    num_epochs: int,
    train_limit: int,
    val_limit: int,
    test_limit: int,
) -> dict:
    np.random.seed(RANDOM_SEED)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name)
    full_shapes = {
        "X_train": list(X_train.shape),
        "y_train": list(y_train.shape),
        "X_val": list(X_val.shape),
        "y_val": list(y_val.shape),
        "X_test": list(X_test.shape),
        "y_test": list(y_test.shape),
    }

    X_train, y_train = take_subset(X_train, y_train, train_limit)
    X_val, y_val = take_subset(X_val, y_val, val_limit)
    X_test, y_test = take_subset(X_test, y_test, test_limit)

    input_dim = X_train.shape[1]
    model = build_mlp(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, seed=RANDOM_SEED)
    criterion = nn.SoftmaxCrossEntropyLoss()
    optimizer = nn.SGD(LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start_time = time.time()

    print(f"\n--- dataset = {dataset_name} ---")
    print(
        f"input_dim={input_dim}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, X_train, y_train)
        train_metrics = evaluate(model, X_train, y_train, batch_size=256)
        val_metrics = evaluate(model, X_val, y_val, criterion=criterion, batch_size=256)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"  Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    elapsed = time.time() - start_time
    test_metrics = evaluate(model, X_test, y_test, criterion=criterion, batch_size=256)
    print(
        f"  Final test loss={test_metrics['loss']:.4f}, "
        f"test accuracy={test_metrics['accuracy']:.4f} ({elapsed:.1f}s)"
    )

    config = {
        "dataset": dataset_name,
        "purpose": "dataset interface smoke experiment",
        "model_type": "mlp",
        "input_dim": input_dim,
        "num_classes": NUM_CLASSES,
        "hidden_dims": HIDDEN_DIMS,
        "batch_size": BATCH_SIZE,
        "num_epochs": num_epochs,
        "learning_rate": LEARNING_RATE,
        "optimizer": "sgd",
        "activation": "ReLU",
        "weight_init": WEIGHT_INIT,
        "seed": RANDOM_SEED,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "test_limit": test_limit,
        "full_dataset_shapes": full_shapes,
    }
    results = {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_test_loss": test_metrics["loss"],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "test_accuracy": test_metrics["accuracy"],
        "training_time_seconds": elapsed,
    }

    save_dir = OUTPUT_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    save_curves(history, save_dir, title_suffix=dataset_name)

    return {**config, **results}


def save_summary(rows: list[dict]):
    keys = [
        "dataset",
        "model_type",
        "input_dim",
        "train_limit",
        "val_limit",
        "test_limit",
        "num_epochs",
        "final_train_loss",
        "final_val_loss",
        "final_test_loss",
        "final_train_acc",
        "final_val_acc",
        "test_accuracy",
        "training_time_seconds",
    ]

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            values = []
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, float):
                    value = f"{value:.6f}"
                values.append(str(value))
            f.write(",".join(values) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run quick interface experiments on MNIST, Fashion-MNIST, and CIFAR-10."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Datasets to run.",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--train-limit", type=int, default=TRAIN_LIMIT)
    parser.add_argument("--val-limit", type=int, default=VAL_LIMIT)
    parser.add_argument("--test-limit", type=int, default=TEST_LIMIT)
    return parser.parse_args()


def run_all():
    args = parse_args()
    rows = []

    print("=" * 60)
    print("Dataset interface smoke experiments")
    print("=" * 60)

    for dataset_name in args.datasets:
        row = run_one_dataset(
            dataset_name=dataset_name,
            num_epochs=args.epochs,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            test_limit=args.test_limit,
        )
        rows.append(row)

    save_summary(rows)
    print("\n" + "=" * 60)
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"Summary saved to {SUMMARY_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
