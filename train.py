"""
Training entry point for the NumPy MNIST MLP.
"""

from __future__ import annotations

import numpy as np

import nn
from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CNN_KERNEL_SIZE,
    CNN_OUT_CHANNELS,
    CNN_PADDING,
    CNN_STRIDE,
    HIDDEN_DIMS,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INPUT_DIM,
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
from utils.checkpoint import checkpoint_path_from_hyperparams, save_checkpoint
from utils.metrics import evaluate
from utils.plot import plot_history

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


def build_mlp():
    """
    Build an MLP from config.py.

    Example:
        HIDDEN_DIMS = [128] builds 784 -> 128 -> 10.
    """
    layers = []
    layer_dims = [INPUT_DIM] + HIDDEN_DIMS + [NUM_CLASSES]

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


def build_cnn():
    """
    Build a small CNN for MNIST.

    Structure:
        Conv2D -> activation -> MaxPool2D -> Flatten -> MLP head
    """
    conv_height = (IMAGE_HEIGHT + 2 * CNN_PADDING - CNN_KERNEL_SIZE) // CNN_STRIDE + 1
    conv_width = (IMAGE_WIDTH + 2 * CNN_PADDING - CNN_KERNEL_SIZE) // CNN_STRIDE + 1
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
            in_channels=IMAGE_CHANNELS,
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


def build_model():
    if MODEL_TYPE == "mlp":
        return build_mlp()
    elif MODEL_TYPE == "cnn":
        return build_cnn()
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")


def prepare_data(X_train, X_val, X_test):
    if MODEL_TYPE == "mlp":
        return X_train, X_val, X_test
    elif MODEL_TYPE == "cnn":
        return (
            to_nchw_images(X_train, channels=IMAGE_CHANNELS, height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            to_nchw_images(X_val, channels=IMAGE_CHANNELS, height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            to_nchw_images(X_test, channels=IMAGE_CHANNELS, height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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
        hyperparams["CNN_OUT_CHANNELS"] = CNN_OUT_CHANNELS
        hyperparams["CNN_KERNEL_SIZE"] = CNN_KERNEL_SIZE
        hyperparams["CNN_STRIDE"] = CNN_STRIDE
        hyperparams["CNN_PADDING"] = CNN_PADDING
        hyperparams["POOL_KERNEL_SIZE"] = POOL_KERNEL_SIZE
        hyperparams["POOL_STRIDE"] = POOL_STRIDE

    if OPTIMIZER == "momentum":
        hyperparams["MOMENTUM"] = MOMENTUM
    elif OPTIMIZER == "adam":
        hyperparams["BETA1"] = BETA1
        hyperparams["BETA2"] = BETA2
        hyperparams["EPS"] = EPS

    return hyperparams


def save_model_checkpoints(model, metadata, interrupted: bool = False):
    latest_checkpoint_path = save_checkpoint(
        model,
        CHECKPOINT_PATH,
        metadata=metadata,
    )
    print(f"Saved latest checkpoint to {latest_checkpoint_path}")

    suffix = "_interrupted.npz" if interrupted else ".npz"
    named_checkpoint_path = checkpoint_path_from_hyperparams(get_hyperparams(), suffix=suffix)
    save_checkpoint(
        model,
        named_checkpoint_path,
        metadata=metadata,
    )
    print(f"Saved named checkpoint to {named_checkpoint_path}")


def main():
    np.random.seed(RANDOM_SEED)

    from data.mnist import X_test, X_train, X_val, y_test, y_train, y_val
    X_train, X_val, X_test = prepare_data(X_train, X_val, X_test)

    model = build_model()
    criterion = nn.SoftmaxCrossEntropyLoss()
    optimizer = build_optimizer()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    completed_epochs = 0

    try:
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
            completed_epochs = epoch + 1

            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"train_acc={train_metrics['accuracy']:.4f}, "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model checkpoint...")
        interrupted_metadata = {
            **get_hyperparams(),
            "completed_epochs": completed_epochs,
            "interrupted": True,
        }
        save_model_checkpoints(model, interrupted_metadata, interrupted=True)
        return

    test_metrics = evaluate(model, X_test, y_test, batch_size=256)
    print(f"Final test accuracy={test_metrics['accuracy']:.4f}")

    figure_dir = plot_history(history, hyperparams=get_hyperparams())
    print(f"Saved figures to {figure_dir}")

    checkpoint_metadata = {
        **get_hyperparams(),
        "completed_epochs": completed_epochs,
        "test_accuracy": test_metrics["accuracy"],
        "interrupted": False,
    }

    save_model_checkpoints(model, checkpoint_metadata)


if __name__ == "__main__":
    main()
