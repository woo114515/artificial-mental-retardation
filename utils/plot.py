"""
Plot training curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history: dict, save_dir: str = "reports/figures") -> None:
    """
    Save loss and accuracy curves from a training history dictionary.

    Expected keys:
        train_loss, val_loss, train_acc, val_acc
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path / "loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(save_path / "accuracy_curve.png")
    plt.close()
