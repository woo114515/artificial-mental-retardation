import importlib
import io
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _idx_images_bytes(num_images, num_rows=28, num_cols=28, fill_value=0):
    header = (
        (2051).to_bytes(4, "big")
        + num_images.to_bytes(4, "big")
        + num_rows.to_bytes(4, "big")
        + num_cols.to_bytes(4, "big")
    )
    pixels = bytearray([fill_value]) * (num_images * num_rows * num_cols)
    if pixels:
        pixels[0] = 0
        pixels[1] = 255
    return header + bytes(pixels)


def _idx_labels_bytes(num_labels):
    header = (2049).to_bytes(4, "big") + num_labels.to_bytes(4, "big")
    labels = (np.arange(num_labels) % 10).astype(np.uint8)
    return header + labels.tobytes()


def _fake_gzip_open(filename, mode="rb"):
    name = str(filename)

    if "train-images" in name:
        return io.BytesIO(_idx_images_bytes(60000))

    if "train-labels" in name:
        return io.BytesIO(_idx_labels_bytes(60000))

    if "t10k-images" in name:
        return io.BytesIO(_idx_images_bytes(10000, fill_value=128))

    if "t10k-labels" in name:
        return io.BytesIO(_idx_labels_bytes(10000))

    raise FileNotFoundError(name)


def _import_mnist_with_fake_data(monkeypatch):
    sys.modules.pop("data.mnist", None)
    monkeypatch.setattr("gzip.open", _fake_gzip_open)
    monkeypatch.setattr("numpy.random.permutation", lambda n: np.arange(n))
    return importlib.import_module("data.mnist")


def test_load_mnist_images_reads_idx_gzip(monkeypatch):
    mnist = _import_mnist_with_fake_data(monkeypatch)

    images = mnist.load_mnist_images("train-images-idx3-ubyte.gz")

    assert images.shape == (60000, 28, 28)
    assert images.dtype == np.uint8
    assert images[0, 0, 0] == 0
    assert images[0, 0, 1] == 255


def test_load_mnist_labels_reads_idx_gzip(monkeypatch):
    mnist = _import_mnist_with_fake_data(monkeypatch)

    labels = mnist.load_mnist_labels("train-labels-idx1-ubyte.gz")

    assert labels.shape == (60000,)
    assert labels.dtype == np.uint8
    np.testing.assert_array_equal(labels[:12], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]))


def test_module_exports_expected_split_shapes_and_dtypes(monkeypatch):
    mnist = _import_mnist_with_fake_data(monkeypatch)

    assert mnist.X_train.shape == (55000, 784)
    assert mnist.y_train.shape == (55000,)
    assert mnist.X_val.shape == (5000, 784)
    assert mnist.y_val.shape == (5000,)
    assert mnist.X_test.shape == (10000, 784)
    assert mnist.y_test.shape == (10000,)

    assert mnist.X_train.dtype == np.float32
    assert mnist.X_val.dtype == np.float32
    assert mnist.X_test.dtype == np.float32

    assert mnist.y_train.dtype == np.uint8
    assert mnist.y_val.dtype == np.uint8
    assert mnist.y_test.dtype == np.uint8


def test_module_flattens_normalizes_and_splits_data(monkeypatch):
    mnist = _import_mnist_with_fake_data(monkeypatch)

    assert mnist.X_train.ndim == 2
    assert mnist.X_val.ndim == 2
    assert mnist.X_test.ndim == 2

    assert np.min(mnist.X_train) >= 0.0
    assert np.max(mnist.X_train) <= 1.0
    assert np.min(mnist.X_test) >= 0.0
    assert np.max(mnist.X_test) <= 1.0

    assert mnist.X_train[0, 0] == 0.0
    assert mnist.X_train[0, 1] == 1.0
    assert mnist.y_train[0] == 0
    assert mnist.y_train[-1] == 9
    assert mnist.y_val[0] == 0
    assert mnist.y_test[0] == 0


def test_shuffle_data_keeps_x_y_pairs_aligned(monkeypatch):
    mnist = _import_mnist_with_fake_data(monkeypatch)

    monkeypatch.setattr("numpy.random.permutation", lambda n: np.array([2, 0, 3, 1]))

    X = np.array([
        [10, 100],
        [20, 200],
        [30, 300],
        [40, 400],
    ])
    y = np.array([1, 2, 3, 4])

    X_shuffled, y_shuffled = mnist.shuffle_data(X, y)

    np.testing.assert_array_equal(
        X_shuffled,
        np.array([
            [30, 300],
            [10, 100],
            [40, 400],
            [20, 200],
        ]),
    )
    np.testing.assert_array_equal(y_shuffled, np.array([3, 1, 4, 2]))

