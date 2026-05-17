"""
Integration tests for the nn/ module.

These tests verify that the whole neural network core can run as a pipeline:

    model.forward(X)
    criterion.forward(logits, y)
    criterion.backward()
    model.backward(dlogits)
    optimizer.step(model.params_and_grads())

Convention:
    X:       (batch_size, input_dim)
    y:       (batch_size,)
    logits:  (batch_size, num_classes)
    dlogits: (batch_size, num_classes)
"""

import numpy as np

from nn.layers import Linear
from nn.activations import ReLU
from nn.losses import SoftmaxCrossEntropyLoss
from nn.optimizers import SGD
from nn.model import Sequential


def test_nn_forward_backward_update_pipeline():
    """
    Verify that the full nn pipeline runs without shape errors.
    """

    np.random.seed(0)

    batch_size = 8
    input_dim = 4
    hidden_dim = 6
    num_classes = 3

    X = np.random.randn(batch_size, input_dim)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])

    model = Sequential([
        Linear(input_dim, hidden_dim, seed=0),
        ReLU(),
        Linear(hidden_dim, num_classes, seed=1),
    ])

    criterion = SoftmaxCrossEntropyLoss()
    optimizer = SGD(lr=0.01)

    logits = model.forward(X)

    assert logits.shape == (batch_size, num_classes)

    loss = criterion.forward(logits, y)

    assert np.isscalar(loss)
    assert np.isfinite(loss)

    dlogits = criterion.backward()

    assert dlogits.shape == logits.shape
    assert np.all(np.isfinite(dlogits))

    dX = model.backward(dlogits)

    assert dX.shape == X.shape
    assert np.all(np.isfinite(dX))

    params_and_grads = model.params_and_grads()

    assert len(params_and_grads) == 4

    for param, grad in params_and_grads:
        assert param.shape == grad.shape
        assert np.all(np.isfinite(grad))

    old_params = [param.copy() for param, _ in params_and_grads]

    optimizer.step(params_and_grads)

    new_params = [param for param, _ in params_and_grads]

    changed = [
        not np.allclose(old_param, new_param)
        for old_param, new_param in zip(old_params, new_params)
    ]

    assert all(changed)


def test_loss_decreases_on_tiny_linear_problem():
    """
    Verify that a tiny linear classifier can reduce loss after several SGD steps.

    This checks that:
        - Linear.forward works
        - SoftmaxCrossEntropyLoss works
        - backward propagation works
        - SGD updates parameters in-place
    """

    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    y = np.array([0, 1, 0, 1])

    model = Sequential([
        Linear(2, 2, seed=0),
    ])

    # Make the initial state deterministic and simple.
    linear = model.layers[0]
    linear.W[...] = 0.0
    linear.b[...] = 0.0

    criterion = SoftmaxCrossEntropyLoss()
    optimizer = SGD(lr=0.1)

    logits = model.forward(X)
    initial_loss = criterion.forward(logits, y)

    for _ in range(100):
        logits = model.forward(X)
        loss = criterion.forward(logits, y)
        dlogits = criterion.backward()
        model.backward(dlogits)
        optimizer.step(model.params_and_grads())

    logits = model.forward(X)
    final_loss = criterion.forward(logits, y)

    assert np.isfinite(initial_loss)
    assert np.isfinite(final_loss)
    assert final_loss < initial_loss


def test_sequential_params_match_trainable_layers_only():
    """
    ReLU should not contribute trainable parameters.
    Two Linear layers should contribute:
        W1, b1, W2, b2
    """

    model = Sequential([
        Linear(4, 5, seed=0),
        ReLU(),
        Linear(5, 3, seed=1),
    ])

    X = np.random.randn(7, 4)
    y = np.array([0, 1, 2, 0, 1, 2, 0])

    criterion = SoftmaxCrossEntropyLoss()

    logits = model.forward(X)
    loss = criterion.forward(logits, y)
    dlogits = criterion.backward()
    model.backward(dlogits)

    params_and_grads = model.params_and_grads()

    assert len(params_and_grads) == 4

    for param, grad in params_and_grads:
        assert param is not None
        assert grad is not None
        assert param.shape == grad.shape


def test_integer_labels_are_required_for_loss():
    """
    The project convention is:
        labels.shape = (batch_size,)
    not:
        labels.shape = (batch_size, num_classes)
    """

    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.2, 3.0, 0.5],
    ])

    labels = np.array([0, 1])

    loss = criterion.forward(logits, labels)

    assert np.isscalar(loss)
    assert np.isfinite(loss)

    dlogits = criterion.backward()

    assert dlogits.shape == logits.shape