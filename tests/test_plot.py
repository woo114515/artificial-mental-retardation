from utils.plot import plot_history


def test_plot_history_creates_curve_files(tmp_path):
    history = {
        "train_loss": [1.0, 0.8, 0.6],
        "val_loss": [1.1, 0.9, 0.7],
        "train_acc": [0.5, 0.7, 0.8],
        "val_acc": [0.4, 0.6, 0.75],
    }

    plot_history(history, save_dir=tmp_path)

    assert (tmp_path / "loss_curve.png").exists()
    assert (tmp_path / "accuracy_curve.png").exists()
