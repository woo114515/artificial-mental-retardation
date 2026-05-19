import numpy as np

from utils.metrics import accuracy, evaluate


class IdentityLogitsModel:
    def forward(self, X):
        return X


class IndexedDummyModel:
    def forward(self, X):
        logits = np.zeros((len(X), 2))
        labels = X[:, 0].astype(int)
        logits[np.arange(len(X)), labels] = 1.0
        return logits


class MeanSquaredLogitLoss:
    def forward(self, logits, labels):
        return float(np.mean((logits[:, 0] - labels) ** 2))


def test_accuracy_from_logits_and_integer_labels():
    logits = np.array([
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 1.0],
        [1.0, 2.0, 4.0],
        [5.0, 1.0, 0.0],
    ])
    labels = np.array([0, 1, 1, 0])

    assert accuracy(logits, labels) == 0.75


def test_evaluate_returns_accuracy_without_loss():
    X = np.array([
        [0],
        [1],
        [0],
        [1],
        [1],
    ])
    y = np.array([0, 1, 0, 1, 1])

    result = evaluate(IndexedDummyModel(), X, y, batch_size=2)

    assert result == {"accuracy": 1.0}


def test_evaluate_returns_weighted_average_loss_and_accuracy():
    X = np.array([
        [2.0, 0.0],
        [0.0, 2.0],
        [1.0, 3.0],
    ])
    y = np.array([0, 1, 1])

    result = evaluate(
        IdentityLogitsModel(),
        X,
        y,
        criterion=MeanSquaredLogitLoss(),
        batch_size=2,
    )

    assert result["accuracy"] == 1.0
    assert result["loss"] == 5 / 3
