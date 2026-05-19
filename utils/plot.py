"""
Plot training curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _format_hidden_dims(hidden_dims) -> str:
    if not hidden_dims:
        return "none"
    return "-".join(str(dim) for dim in hidden_dims)


def _format_hyperparams(hyperparams: dict | None) -> str:
    if not hyperparams:
        return ""

    return (
        f"hidden={_format_hidden_dims(hyperparams['HIDDEN_DIMS'])}, "
        f"lr={hyperparams['LEARNING_RATE']}, "
        f"batch={hyperparams['BATCH_SIZE']}, "
        f"seed={hyperparams['RANDOM_SEED']}, "
        f"init={hyperparams['WEIGHT_INIT']}"
    )


def _make_run_dir(save_dir: str, hyperparams: dict | None) -> Path:
    base_path = Path(save_dir)

    if not hyperparams:
        return base_path

    run_name = (
        f"hidden-{_format_hidden_dims(hyperparams['HIDDEN_DIMS'])}"
        f"_lr-{hyperparams['LEARNING_RATE']}"
        f"_bs-{hyperparams['BATCH_SIZE']}"
        f"_seed-{hyperparams['RANDOM_SEED']}"
        f"_init-{hyperparams['WEIGHT_INIT']}"
    )
    return base_path / run_name


def plot_history(
    history: dict,
    save_dir: str = "experiments/records",
    hyperparams: dict | None = None,
) -> Path:
    """
    Save loss and accuracy curves from a training history dictionary.

    Expected keys:
        train_loss, val_loss, train_acc, val_acc
    """
    save_path = _make_run_dir(save_dir, hyperparams)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    hyperparam_text = _format_hyperparams(hyperparams)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title(f"Loss Curve\n{hyperparam_text}" if hyperparam_text else "Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path / "loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title(f"Accuracy Curve\n{hyperparam_text}" if hyperparam_text else "Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(save_path / "accuracy_curve.png")
    plt.close()

    return save_path
