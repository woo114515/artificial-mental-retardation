"""
Tests for prediction visualization helpers.
"""

import numpy as np
import pytest

from utils.prediction_visualization import plot_error_predictions, plot_test_predictions, softmax


class DummyModel:
    def forward(self, X):
        logits = np.zeros((X.shape[0], 3))
        logits[:, 1] = 2.0
        return logits


def test_softmax_rows_sum_to_one():
    logits = np.array([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
    ])

    probs = softmax(logits)

    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(2))


def test_plot_test_predictions_creates_file(tmp_path):
    X_display = np.arange(4 * 4).reshape(4, 4)
    X_model = X_display.copy()
    y = np.array([0, 1, 2, 1])
    save_path = tmp_path / "predictions.png"

    output_path = plot_test_predictions(
        model=DummyModel(),
        X_display=X_display,
        X_model=X_model,
        y=y,
        save_path=save_path,
        num_samples=4,
        seed=0,
        image_shape=(2, 2),
    )

    assert output_path == save_path
    assert save_path.exists()


def test_plot_test_predictions_rejects_mismatched_sample_counts(tmp_path):
    X_display = np.zeros((4, 4))
    X_model = np.zeros((3, 4))
    y = np.zeros(4, dtype=int)

    with pytest.raises(ValueError):
        plot_test_predictions(
            model=DummyModel(),
            X_display=X_display,
            X_model=X_model,
            y=y,
            save_path=tmp_path / "predictions.png",
            image_shape=(2, 2),
        )


def test_plot_test_predictions_rejects_non_flat_display_data(tmp_path):
    X_display = np.zeros((4, 1, 2, 2))
    X_model = np.zeros((4, 4))
    y = np.zeros(4, dtype=int)

    with pytest.raises(ValueError):
        plot_test_predictions(
            model=DummyModel(),
            X_display=X_display,
            X_model=X_model,
            y=y,
            save_path=tmp_path / "predictions.png",
            image_shape=(2, 2),
        )


def test_plot_error_predictions_creates_file(tmp_path):
    X_display = np.arange(4 * 4).reshape(4, 4)
    y = np.array([0, 1, 2, 1])
    preds = np.array([0, 2, 2, 0])
    confidences = np.array([0.9, 0.8, 0.7, 0.6])
    error_indices = np.where(preds != y)[0]
    save_path = tmp_path / "error_predictions.png"

    output_path = plot_error_predictions(
        X_display=X_display,
        y=y,
        preds=preds,
        confidences=confidences,
        error_indices=error_indices,
        save_path=save_path,
        num_samples=2,
        image_shape=(2, 2),
        title="MODEL_TYPE=mlp, OPTIMIZER=sgd",
    )

    assert output_path == save_path
    assert save_path.exists()


def test_plot_error_predictions_creates_file_when_no_errors(tmp_path):
    X_display = np.arange(4 * 4).reshape(4, 4)
    y = np.array([0, 1, 2, 1])
    preds = y.copy()
    confidences = np.array([0.9, 0.8, 0.7, 0.6])
    save_path = tmp_path / "no_errors.png"

    output_path = plot_error_predictions(
        X_display=X_display,
        y=y,
        preds=preds,
        confidences=confidences,
        error_indices=np.array([], dtype=int),
        save_path=save_path,
        num_samples=3,
        image_shape=(2, 2),
    )

    assert output_path == save_path
    assert save_path.exists()


def test_plot_error_predictions_rejects_mismatched_sample_counts(tmp_path):
    X_display = np.zeros((4, 4))
    y = np.zeros(4, dtype=int)
    preds = np.zeros(3, dtype=int)
    confidences = np.ones(4)

    with pytest.raises(ValueError):
        plot_error_predictions(
            X_display=X_display,
            y=y,
            preds=preds,
            confidences=confidences,
            error_indices=np.array([0]),
            save_path=tmp_path / "error_predictions.png",
            image_shape=(2, 2),
        )
