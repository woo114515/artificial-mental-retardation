"""
优化器及学习率超参数对比实验。

基于主模型 784 -> 128 -> 10 (hidden_dims=[128])，共四组实验：
  1. 优化器类型对比（SGD / Momentum / Adam）
  2. Momentum 系数对比（0.5 / 0.9 / 0.99）
  3. Adam beta1 对比（0.5 / 0.9 / 0.99）
  4. 学习率对比（0.001 / 0.01 / 0.1）

每组实验结果保存到 experiments/<组名>/<子文件夹>/ 下，包含：
  - config.json     实验配置
  - results.json    训练结果
  - loss_curve.png / accuracy_curve.png  训练曲线

同时生成 experiments/summary.csv 汇总所有结果。
"""

from __future__ import annotations

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
from data.mnist import X_test, X_train, X_val, y_test, y_train, y_val
from utils.metrics import evaluate

# ---------------------------------------------------------------------------
# 固定配置（与 experiment.md 基线一致）
# ---------------------------------------------------------------------------
NUM_EPOCHS = 150
RANDOM_SEED = 42
HIDDEN_DIMS = [128]
BATCH_SIZE = 64
ACTIVATION = "ReLU"
WEIGHT_INIT = "he"
INPUT_DIM = 784
NUM_CLASSES = 10
EXPERIMENTS_DIR = Path(__file__).resolve().parent


def build_mlp(seed: int):
    layer_dims = [INPUT_DIM] + HIDDEN_DIMS + [NUM_CLASSES]
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1], method=WEIGHT_INIT, seed=seed + i))
        if i < len(layer_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(layers)


def train_one_epoch(model, criterion, optimizer, X, y):
    total_loss = 0.0
    total_count = 0
    for Xb, yb in create_mini_batches(X, y, batch_size=BATCH_SIZE):
        logits = model.forward(Xb)
        loss = criterion.forward(logits, yb)
        model.backward(criterion.backward())
        optimizer.step(model.params_and_grads())
        total_loss += loss * len(yb)
        total_count += len(yb)
    return float(total_loss / total_count)


def save_curves(history: dict, save_dir: Path, title_suffix: str = ""):
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title(f"Loss Curve{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_dir / "loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title(f"Accuracy Curve{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(save_dir / "accuracy_curve.png")
    plt.close()


def run_one_experiment(save_dir: Path, optimizer_name: str, lr: float,
                       momentum: float | None = None,
                       beta1: float | None = None,
                       beta2: float = 0.999, eps: float = 1e-8) -> dict:
    """运行单次实验，保存曲线和结果，返回指标字典。"""
    np.random.seed(RANDOM_SEED)

    model = build_mlp(RANDOM_SEED)
    criterion = nn.SoftmaxCrossEntropyLoss()

    if optimizer_name == "sgd":
        optimizer = nn.SGD(lr)
    elif optimizer_name == "momentum":
        optimizer = nn.Momentum(lr, momentum=momentum if momentum is not None else 0.9)
    elif optimizer_name == "adam":
        optimizer = nn.Adam(lr, beta1=beta1 if beta1 is not None else 0.9,
                            beta2=beta2, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, criterion, optimizer, X_train, y_train)
        train_metrics = evaluate(model, X_train, y_train, batch_size=256)
        val_metrics = evaluate(model, X_val, y_val, criterion=criterion, batch_size=256)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                  f"train_acc={train_metrics['accuracy']:.4f}, val_acc={val_metrics['accuracy']:.4f}")

    elapsed = time.time() - start_time
    test_metrics = evaluate(model, X_test, y_test, batch_size=256)
    print(f"  Final test accuracy = {test_metrics['accuracy']:.4f}  ({elapsed:.0f}s)")

    config = {
        "hidden_dims": HIDDEN_DIMS,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "activation": ACTIVATION,
        "weight_init": WEIGHT_INIT,
        "seed": RANDOM_SEED,
        "optimizer": optimizer_name,
        "learning_rate": lr,
    }
    if optimizer_name == "momentum":
        config["momentum"] = momentum
    if optimizer_name == "adam":
        config["beta1"] = beta1
        config["beta2"] = beta2

    results = {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "test_accuracy": test_metrics["accuracy"],
        "training_time_seconds": elapsed,
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    title_suffix = f" ({optimizer_name}"
    if momentum is not None:
        title_suffix += f", momentum={momentum}"
    if beta1 is not None:
        title_suffix += f", beta1={beta1}"
    title_suffix += f", lr={lr})"
    save_curves(history, save_dir, title_suffix)

    return {**config, **results}


# ---------------------------------------------------------------------------
# 实验组定义
# ---------------------------------------------------------------------------

def run_all():
    all_rows = []

    # =====================================================================
    # 实验 1：优化器类型对比
    # =====================================================================
    group = "optimizer"
    print(f"\n{'='*60}")
    print(f"  实验组 1: 优化器类型对比")
    print(f"{'='*60}")

    experiments = [
        # (label, optimizer_name, lr, momentum, beta1)
        ("sgd",      "sgd",      0.01,  None, None),
        ("momentum", "momentum", 0.01,  0.9,  None),
        ("adam",     "adam",     0.001, None, 0.9),
    ]
    for label, opt_name, lr, mom, b1 in experiments:
        print(f"\n--- {label} ---")
        row = run_one_experiment(
            save_dir=EXPERIMENTS_DIR / group / label,
            optimizer_name=opt_name, lr=lr, momentum=mom, beta1=b1,
        )
        row["experiment_group"] = group
        all_rows.append(row)

    # =====================================================================
    # 实验 2：Momentum 系数对比
    # =====================================================================
    group = "momentum"
    print(f"\n{'='*60}")
    print(f"  实验组 2: Momentum 系数对比")
    print(f"{'='*60}")

    for momentum in [0.5, 0.9, 0.99]:
        label = f"momentum-{momentum}"
        print(f"\n--- momentum = {momentum} ---")
        row = run_one_experiment(
            save_dir=EXPERIMENTS_DIR / group / label,
            optimizer_name="momentum", lr=0.01, momentum=momentum,
        )
        row["experiment_group"] = group
        all_rows.append(row)

    # =====================================================================
    # 实验 3：Adam beta1 对比
    # =====================================================================
    group = "adam"
    print(f"\n{'='*60}")
    print(f"  实验组 3: Adam beta1 对比")
    print(f"{'='*60}")

    for beta1 in [0.5, 0.9, 0.99]:
        label = f"beta1-{beta1}"
        print(f"\n--- beta1 = {beta1} ---")
        row = run_one_experiment(
            save_dir=EXPERIMENTS_DIR / group / label,
            optimizer_name="adam", lr=0.001, beta1=beta1,
        )
        row["experiment_group"] = group
        all_rows.append(row)

    # =====================================================================
    # 实验 4：学习率对比
    # =====================================================================
    group = "learning_rate"
    print(f"\n{'='*60}")
    print(f"  实验组 4: 学习率对比")
    print(f"{'='*60}")

    for lr in [0.001, 0.01, 0.1]:
        label = f"lr-{lr}"
        print(f"\n--- lr = {lr} ---")
        row = run_one_experiment(
            save_dir=EXPERIMENTS_DIR / group / label,
            optimizer_name="sgd", lr=lr,
        )
        row["experiment_group"] = group
        all_rows.append(row)

    # =====================================================================
    # 保存汇总 CSV
    # =====================================================================
    summary_path = EXPERIMENTS_DIR / "summary.csv"
    keys = [
        "experiment_group", "optimizer", "learning_rate",
        "momentum", "beta1",
        "final_train_loss", "final_val_loss",
        "final_train_acc", "final_val_acc", "test_accuracy",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in all_rows:
            vals = [
                row.get("experiment_group", ""),
                row.get("optimizer", ""),
                str(row.get("learning_rate", "")),
                str(row.get("momentum", "")),
                str(row.get("beta1", "")),
                f"{row['final_train_loss']:.6f}",
                f"{row['final_val_loss']:.6f}",
                f"{row['final_train_acc']:.6f}",
                f"{row['final_val_acc']:.6f}",
                f"{row['test_accuracy']:.6f}",
            ]
            f.write(",".join(vals) + "\n")

    print(f"\n{'='*60}")
    print(f"  汇总已保存到 {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all()
