"""
Training entry point for the NumPy MNIST MLP.
"""

from __future__ import annotations

import numpy as np

import nn
from config import (
    BATCH_SIZE,
    CNN_KERNEL_SIZE,
    CNN_OUT_CHANNELS,
    CNN_PADDING,
    CNN_STRIDE,
    DATASET,
    HIDDEN_DIMS,
    LEARNING_RATE,
    MODEL_TYPE,
    NUM_CLASSES,
    NUM_EPOCHS,
    OPTIMIZER,
    POOL_KERNEL_SIZE,
    POOL_STRIDE,
    MOMENTUM,
    BETA1,
    BETA2,
    EPS,
    ACTIVATION,
    RANDOM_SEED,
    WEIGHT_INIT,
)
from data.dataloader import create_mini_batches
from data.transforms import to_nchw_images
from utils.metrics import evaluate
from utils.plot import plot_history


DATASET_IMAGE_SHAPES = {
    "mnist": (1, 28, 28),
    "fashionmnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
}


def get_image_shape(dataset: str | None = None) -> tuple[int, int, int]:
    """
    Return image shape as (channels, height, width) for a supported dataset.
    """
    if dataset is None:
        dataset = DATASET

    try:
        return DATASET_IMAGE_SHAPES[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Supported: {', '.join(DATASET_IMAGE_SHAPES)}"
        ) from exc


def build_activation():
    '''
    Build the activation
    '''

    if ACTIVATION == "ReLU":
        return nn.ReLU()
    elif ACTIVATION == "Sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {ACTIVATION}")
    
def build_optimizer():
    '''
    Build the optimizer
    '''

    if OPTIMIZER == "sgd":
        return nn.SGD(LEARNING_RATE)
    elif OPTIMIZER == "momentum":
        return nn.Momentum(lr=LEARNING_RATE, momentum=MOMENTUM)
    elif OPTIMIZER == "adam":
        return nn.Adam(lr=LEARNING_RATE, beta1=BETA1, beta2=BETA2, eps=EPS)
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")


def build_model(input_dim: int):
    """
    Build an MLP from config.py.

    Args:
        input_dim: 输入维度，由数据集决定（MNIST=784, CIFAR-10=3072）

    Example:
        HIDDEN_DIMS = [128] builds input_dim -> 128 -> 10.
    """
    layers = []
    layer_dims = [input_dim] + HIDDEN_DIMS + [NUM_CLASSES]

    for layer_index in range(len(layer_dims) - 1):
        layers.append(
            nn.Linear(
                in_features=layer_dims[layer_index],
                out_features=layer_dims[layer_index + 1],
                method=WEIGHT_INIT,
                seed=RANDOM_SEED + layer_index,
            )
        )

        if layer_index < len(layer_dims) - 2:
            layers.append(build_activation())

    return nn.Sequential(layers)


def build_cnn(image_shape: tuple[int, int, int]):
    """
    Build a small CNN from the dataset image shape.

    Structure:
        Conv2D -> activation -> MaxPool2D -> Flatten -> MLP head
    """
    image_channels, image_height, image_width = image_shape
    conv_height = (image_height + 2 * CNN_PADDING - CNN_KERNEL_SIZE) // CNN_STRIDE + 1
    conv_width = (image_width + 2 * CNN_PADDING - CNN_KERNEL_SIZE) // CNN_STRIDE + 1
    pooled_height = (conv_height - POOL_KERNEL_SIZE) // POOL_STRIDE + 1
    pooled_width = (conv_width - POOL_KERNEL_SIZE) // POOL_STRIDE + 1

    if pooled_height <= 0 or pooled_width <= 0:
        raise ValueError(
            f"Invalid CNN spatial size after conv/pool: ({pooled_height}, {pooled_width})."
        )

    flattened_dim = CNN_OUT_CHANNELS * pooled_height * pooled_width
    layer_dims = [flattened_dim] + HIDDEN_DIMS + [NUM_CLASSES]

    layers = [
        nn.Conv2D(
            in_channels=image_channels,
            out_channels=CNN_OUT_CHANNELS,
            kernel_size=CNN_KERNEL_SIZE,
            stride=CNN_STRIDE,
            padding=CNN_PADDING,
            method=WEIGHT_INIT,
            seed=RANDOM_SEED,
        ),
        build_activation(),
        nn.MaxPool2D(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE),
        nn.Flatten(),
    ]

    for layer_index in range(len(layer_dims) - 1):
        layers.append(
            nn.Linear(
                in_features=layer_dims[layer_index],
                out_features=layer_dims[layer_index + 1],
                method=WEIGHT_INIT,
                seed=RANDOM_SEED + layer_index + 1,
            )
        )

        if layer_index < len(layer_dims) - 2:
            layers.append(build_activation())

    return nn.Sequential(layers)


def build_model_from_config(input_dim: int, image_shape: tuple[int, int, int] | None = None):
    if MODEL_TYPE == "mlp":
        return build_model(input_dim=input_dim)
    elif MODEL_TYPE == "cnn":
        return build_cnn(image_shape=image_shape or get_image_shape())
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")


def prepare_data(X_train, X_val, X_test, image_shape: tuple[int, int, int] | None = None):
    if MODEL_TYPE == "mlp":
        return X_train, X_val, X_test
    elif MODEL_TYPE == "cnn":
        image_channels, image_height, image_width = image_shape or get_image_shape()
        expected_features = image_channels * image_height * image_width
        actual_features = X_train.shape[1]
        if actual_features != expected_features:
            raise ValueError(
                f"MODEL_TYPE='cnn' with DATASET='{DATASET}' expects flat images with "
                f"{expected_features} features from image_shape="
                f"{(image_channels, image_height, image_width)}; "
                f"got {actual_features}."
            )

        return (
            to_nchw_images(X_train, channels=image_channels, height=image_height, width=image_width),
            to_nchw_images(X_val, channels=image_channels, height=image_height, width=image_width),
            to_nchw_images(X_test, channels=image_channels, height=image_height, width=image_width),
        )
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")


def train_one_epoch(model, criterion, optimizer, X_train, y_train, batch_size):
    """
    Train for one epoch and return the average batch loss weighted by batch size.
    """
    total_loss = 0.0
    total_count = 0

    for X_batch, y_batch in create_mini_batches(X_train, y_train, batch_size=batch_size):
        logits = model.forward(X_batch)
        loss = criterion.forward(logits, y_batch)

        dlogits = criterion.backward()
        model.backward(dout=dlogits)

        optimizer.step(model.params_and_grads())

        total_loss += loss * len(y_batch)
        total_count += len(y_batch)

    return float(total_loss / total_count)


def get_hyperparams():
    hyperparams = {
        "DATASET": DATASET,
        "MODEL_TYPE": MODEL_TYPE,
        "BATCH_SIZE": BATCH_SIZE,
        "HIDDEN_DIMS": HIDDEN_DIMS,
        "LEARNING_RATE": LEARNING_RATE,
        "OPTIMIZER": OPTIMIZER,
        "ACTIVATION": ACTIVATION,
        "RANDOM_SEED": RANDOM_SEED,
        "WEIGHT_INIT": WEIGHT_INIT,
    }

    if MODEL_TYPE == "cnn":
        image_channels, image_height, image_width = get_image_shape()
        hyperparams["CNN_OUT_CHANNELS"] = CNN_OUT_CHANNELS
        hyperparams["CNN_KERNEL_SIZE"] = CNN_KERNEL_SIZE
        hyperparams["CNN_STRIDE"] = CNN_STRIDE
        hyperparams["CNN_PADDING"] = CNN_PADDING
        hyperparams["POOL_KERNEL_SIZE"] = POOL_KERNEL_SIZE
        hyperparams["POOL_STRIDE"] = POOL_STRIDE
        hyperparams["IMAGE_CHANNELS"] = image_channels
        hyperparams["IMAGE_HEIGHT"] = image_height
        hyperparams["IMAGE_WIDTH"] = image_width

    if OPTIMIZER == "momentum":
        hyperparams["MOMENTUM"] = MOMENTUM
    elif OPTIMIZER == "adam":
        hyperparams["BETA1"] = BETA1
        hyperparams["BETA2"] = BETA2
        hyperparams["EPS"] = EPS

    return hyperparams


def main():
    np.random.seed(RANDOM_SEED)

    # 根据 DATASET 配置动态导入对应的数据模块
    if DATASET == "mnist":
        from data.mnist import X_test, X_train, X_val, y_test, y_train, y_val
    elif DATASET == "fashionmnist":
        from data.fashionmnist import X_test, X_train, X_val, y_test, y_train, y_val
    elif DATASET == "cifar10":
        from data.cifar10 import X_test, X_train, X_val, y_test, y_train, y_val
    else:
        raise ValueError(f"Unsupported dataset: {DATASET}. "
                         f"Supported: 'mnist', 'fashionmnist', 'cifar10'")

    # 自动推断输入维度（MNIST/Fashion-MNIST=784, CIFAR-10=3072）
    input_dim = X_train.shape[1]
    image_shape = get_image_shape(DATASET)

    X_train, X_val, X_test = prepare_data(X_train, X_val, X_test, image_shape=image_shape)

    if MODEL_TYPE == "mlp":
        input_info = f"input_dim={input_dim}"
    else:
        input_info = f"input_shape={X_train.shape[1:]}"

    print(f"Dataset: {DATASET}, model_type={MODEL_TYPE}, {input_info}, "
          f"num_train={X_train.shape[0]}, num_val={X_val.shape[0]}, "
          f"num_test={X_test.shape[0]}")

    model = build_model_from_config(input_dim=input_dim, image_shape=image_shape)
    criterion = nn.SoftmaxCrossEntropyLoss()
    optimizer = build_optimizer()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            X_train=X_train,
            y_train=y_train,
            batch_size=BATCH_SIZE,
        )

        train_metrics = evaluate(model, X_train, y_train, batch_size=256)
        val_metrics = evaluate(model, X_val, y_val, criterion=criterion, batch_size=256)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    test_metrics = evaluate(model, X_test, y_test, batch_size=256)
    print(f"Final test accuracy={test_metrics['accuracy']:.4f}")

    figure_dir = plot_history(history, hyperparams=get_hyperparams())
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
