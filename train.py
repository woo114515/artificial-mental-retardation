"""
Training entry point for the NumPy MNIST MLP.
"""

from __future__ import annotations

import numpy as np

import nn
from config import (
    BATCH_SIZE,
    HIDDEN_DIMS,
    INPUT_DIM,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    OPTIMIZER,
    MOMENTUM,
    BETA1,
    BETA2,
    EPS,
    ACTIVATION,
    RANDOM_SEED,
    WEIGHT_INIT,
)
from data.dataloader import create_mini_batches
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


def build_model():
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
        "BATCH_SIZE": BATCH_SIZE,
        "HIDDEN_DIMS": HIDDEN_DIMS,
        "LEARNING_RATE": LEARNING_RATE,
        "OPTIMIZER": OPTIMIZER,
        "ACTIVATION": ACTIVATION,
        "RANDOM_SEED": RANDOM_SEED,
        "WEIGHT_INIT": WEIGHT_INIT,
    }

    if OPTIMIZER == "momentum":
        hyperparams["MOMENTUM"] = MOMENTUM
    elif OPTIMIZER == "adam":
        hyperparams["BETA1"] = BETA1
        hyperparams["BETA2"] = BETA2
        hyperparams["EPS"] = EPS

    return hyperparams


def main():
    np.random.seed(RANDOM_SEED)

    from data.mnist import X_test, X_train, X_val, y_test, y_train, y_val

    model = build_model()
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
