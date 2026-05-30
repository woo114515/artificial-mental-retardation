"""
超参数对比实验。

四组实验：
  1. Batch Size 对比（32 / 64 / 128 / 256）
  2. 隐藏层宽度对比（64 / 128 / 256 / 512）
  3. 激活函数对比（ReLU / Sigmoid）
  4. 隐藏层数对比（1层 / 2层 / 3层）

固定参数：
  - 学习率 lr = 0.01
  - 优化器 = SGD
  - 权重初始化 = he
  - 随机种子 = 42
  - 训练轮数 = 20

每组实验结果保存到 experiments/<组名>/<子文件夹>/ 下，包含：
  - config.json     实验配置
  - results.json    实验结果
  - loss_curve.png  / accuracy_curve.png  训练曲线

同时生成 experiments/hyperparameter_summary.csv 汇总所有结果。
"""

from __future__ import annotations

import json
import sys
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
# 固定配置
# ---------------------------------------------------------------------------
NUM_EPOCHS = 20
RANDOM_SEED = 42
LEARNING_RATE = 0.01
OPTIMIZER = "sgd"
WEIGHT_INIT = "he"
INPUT_DIM = 784
NUM_CLASSES = 10
EXPERIMENTS_DIR = Path(__file__).resolve().parent


def build_mlp(hidden_dims: list[int], activation: str, seed: int):
    """根据隐藏层结构和激活函数类型构建 MLP。"""
    layer_dims = [INPUT_DIM] + hidden_dims + [NUM_CLASSES]
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
            if activation == "ReLU":
                layers.append(nn.ReLU())
            elif activation == "Sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}")
    return nn.Sequential(layers)


def train_one_epoch(model, criterion, optimizer, X, y, batch_size):
    """训练一个 epoch，返回加权平均 loss。"""
    total_loss = 0.0
    total_count = 0
    for Xb, yb in create_mini_batches(X, y, batch_size=batch_size):
        logits = model.forward(Xb)
        loss = criterion.forward(logits, yb)
        model.backward(criterion.backward())
        optimizer.step(model.params_and_grads())
        total_loss += loss * len(yb)
        total_count += len(yb)
    return float(total_loss / total_count)


def save_curves(history: dict, save_dir: Path, title_suffix: str = ""):
    """保存 loss 曲线和 accuracy 曲线。"""
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


def run_one_experiment(
    save_dir: Path,
    hidden_dims: list[int],
    activation: str,
    batch_size: int,
) -> dict:
    """运行单次实验，保存曲线和结果，返回指标字典。"""
    np.random.seed(RANDOM_SEED)

    model = build_mlp(hidden_dims, activation, RANDOM_SEED)
    criterion = nn.SoftmaxCrossEntropyLoss()
    optimizer = nn.SGD(LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, criterion, optimizer, X_train, y_train, batch_size
        )
        train_metrics = evaluate(model, X_train, y_train, batch_size=256)
        val_metrics = evaluate(
            model, X_val, y_val, criterion=criterion, batch_size=256
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"  Epoch {epoch + 1:2d}: "
            f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, val_acc={val_metrics['accuracy']:.4f}"
        )

    test_metrics = evaluate(model, X_test, y_test, batch_size=256)
    print(f"  Final test accuracy = {test_metrics['accuracy']:.4f}")

    # ---- 确保保存目录存在 ----
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 保存 config.json ----
    config = {
        "hidden_dims": hidden_dims,
        "activation": activation,
        "batch_size": batch_size,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER,
        "weight_init": WEIGHT_INIT,
        "seed": RANDOM_SEED,
    }
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ---- 保存 results.json ----
    results = {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "test_accuracy": test_metrics["accuracy"],
    }
    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- 保存曲线图 ----
    hidden_str = "-".join(str(d) for d in hidden_dims)
    title_suffix = (
        f" (bs={batch_size}, hidden=[{hidden_str}], act={activation})"
    )
    save_curves(history, save_dir, title_suffix)

    return {
        "hidden_dims": hidden_str,
        "activation": activation,
        "batch_size": batch_size,
        **results,
    }


def run_all():
    all_rows = []

    # =====================================================================
    # 实验 1：Batch Size 对比
    # =====================================================================
    group = "batch_size"
    print(f"\n{'='*60}")
    print(f"  实验组 1: Batch Size 对比")
    print(f"  （固定: lr=0.01, sgd, hidden=[128], ReLU）")
    print(f"{'='*60}")

    for bs in [32, 64, 128, 256]:
        label = f"bs-{bs}"
        print(f"\n--- batch_size = {bs} ---")
        row = run_one_experiment(
            save_dir=EXPERIMENTS_DIR / group / label,
            hidden_dims=[128],
            activation="ReLU",
            batch_size=bs,
        )
        row["experiment_group"] = group
        row["experiment_label"] = f"batch_size={bs}"
        all_rows.append(row)

    # =====================================================================
    # 实验 2：隐藏层宽度对比
    # =====================================================================
    group = "hidden_width"
    print(f"\n{'='*60}")
    print(f"  实验组 2: 隐藏层宽度对比")
    print(f"  （固定: lr=0.01, sgd, bs=64, ReLU, 1层隐藏层）")
    print(f"{'='*60}")

    for width in [64, 128, 256, 512]:
        label = f"width-{width}"
        print(f"\n--- hidden_dims = [{width}] ---")
        row = run_one_experiment(
            save_dir=EXPERIMENTS_DIR / group / label,
            hidden_dims=[width],
            activation="ReLU",
            batch_size=64,
        )
        row["experiment_group"] = group
        row["experiment_label"] = f"hidden_width={width}"
        all_rows.append(row)

    # =====================================================================
    # 实验 3：激活函数对比
    # =====================================================================
    group = "activation"
    print(f"\n{'='*60}")
    print(f"  实验组 3: 激活函数对比")
    print(f"  （固定: lr=0.01, sgd, bs=64, hidden=[128]）")
    print(f"{'='*60}")

    for act in ["ReLU", "Sigmoid"]:
        label = f"act-{act}"
        print(f"\n--- activation = {act} ---")
        row = run_one_experiment(
            save_dir=EXPERIMENTS_DIR / group / label,
            hidden_dims=[128],
            activation=act,
            batch_size=64,
        )
        row["experiment_group"] = group
        row["experiment_label"] = f"activation={act}"
        all_rows.append(row)

    # =====================================================================
    # 实验 4：隐藏层数（深度）对比
    # =====================================================================
    group = "hidden_depth"
    print(f"\n{'='*60}")
    print(f"  实验组 4: 隐藏层数对比")
    print(f"  （固定: lr=0.01, sgd, bs=64, ReLU）")
    print(f"{'='*60}")

    depth_configs = [
        ([128],         "1层 → [128]"),
        ([256, 128],    "2层 → [256, 128]"),
        ([256, 128, 64],"3层 → [256, 128, 64]"),
    ]

    for hidden_dims, description in depth_configs:
        hidden_str = "-".join(str(d) for d in hidden_dims)
        label = f"depth-{hidden_str}"
        print(f"\n--- {description} ---")
        row = run_one_experiment(
            save_dir=EXPERIMENTS_DIR / group / label,
            hidden_dims=hidden_dims,
            activation="ReLU",
            batch_size=64,
        )
        row["experiment_group"] = group
        row["experiment_label"] = f"hidden_dims=[{hidden_str}]"
        all_rows.append(row)

    # =====================================================================
    # 保存汇总 CSV
    # =====================================================================
    summary_path = EXPERIMENTS_DIR / "hyperparameter_summary.csv"
    keys = [
        "experiment_group",
        "experiment_label",
        "hidden_dims",
        "activation",
        "batch_size",
        "final_train_loss",
        "final_val_loss",
        "final_train_acc",
        "final_val_acc",
        "test_accuracy",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in all_rows:
            vals = [
                row.get("experiment_group", ""),
                row.get("experiment_label", ""),
                str(row.get("hidden_dims", "")),
                row.get("activation", ""),
                str(row.get("batch_size", "")),
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
