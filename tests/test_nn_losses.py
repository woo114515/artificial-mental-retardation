"""
Tests for nn.losses.SoftmaxCrossEntropyLoss.

Convention:
    logits: (batch_size, num_classes)
    labels: (batch_size,)
    loss: scalar
    dlogits: (batch_size, num_classes)
"""

import numpy as np
import pytest

from nn.losses import SoftmaxCrossEntropyLoss


def test_forward_returns_scalar_loss():
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.2, 3.0, 0.5],
    ])

    labels = np.array([0, 1])

    loss = criterion.forward(logits, labels)

    assert np.isscalar(loss)
    assert np.isfinite(loss)
    assert loss > 0


def test_softmax_probs_shape_and_row_sum():
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.2, 3.0, 0.5],
    ])

    labels = np.array([0, 1])

    criterion.forward(logits, labels)

    assert criterion.probs.shape == logits.shape

    row_sums = np.sum(criterion.probs, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(logits.shape[0]))


def test_forward_loss_value():
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.2, 3.0, 0.5],
    ])

    labels = np.array([0, 1])

    loss = criterion.forward(logits, labels)

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    correct_probs = np.array([
        probs[0, 0],
        probs[1, 1],
    ])

    expected_loss = np.mean(-np.log(correct_probs + 1e-12))

    np.testing.assert_allclose(loss, expected_loss)


def test_backward_shape():
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.2, 3.0, 0.5],
    ])

    labels = np.array([0, 1])

    criterion.forward(logits, labels)
    dlogits = criterion.backward()

    assert dlogits.shape == logits.shape


def test_backward_value():
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.2, 3.0, 0.5],
    ])

    labels = np.array([0, 1])

    criterion.forward(logits, labels)
    dlogits = criterion.backward()

    batch_size = logits.shape[0]

    expected = criterion.probs.copy()
    expected[np.arange(batch_size), labels] -= 1
    expected /= batch_size

    np.testing.assert_allclose(dlogits, expected)


def test_backward_before_forward_raises_error():
    criterion = SoftmaxCrossEntropyLoss()

    with pytest.raises(RuntimeError):
        criterion.backward()


def test_numerical_stability_with_large_logits():
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([
        [1000.0, 1001.0, 999.0],
        [2000.0, 1999.0, 1998.0],
    ])

    labels = np.array([1, 0])

    loss = criterion.forward(logits, labels)
    dlogits = criterion.backward()

    assert np.isfinite(loss)
    assert np.all(np.isfinite(criterion.probs))
    assert np.all(np.isfinite(dlogits))


def test_invalid_logits_dimension_raises_error():
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([1.0, 2.0, 3.0])
    labels = np.array([0])

    with pytest.raises(ValueError):
        criterion.forward(logits, labels)


def test_invalid_labels_dimension_raises_error():
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])

    labels = np.array([
        [0],
        [1],
    ])

    with pytest.raises(ValueError):
        criterion.forward(logits, labels)


def test_batch_size_mismatch_raises_error():
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])

    labels = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        criterion.forward(logits, labels)


def test_label_out_of_range_raises_error():
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])

    labels = np.array([0, 3])

    with pytest.raises(ValueError):
        criterion.forward(logits, labels)