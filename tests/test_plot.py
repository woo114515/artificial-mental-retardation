from utils.plot import plot_history


def test_plot_history_creates_curve_files(tmp_path):
    history = {
        "train_loss": [1.0, 0.8, 0.6],
        "val_loss": [1.1, 0.9, 0.7],
        "train_acc": [0.5, 0.7, 0.8],
        "val_acc": [0.4, 0.6, 0.75],
    }
    hyperparams = {
        "BATCH_SIZE": 64,
        "HIDDEN_DIMS": [128],
        "LEARNING_RATE": 0.01,
        "RANDOM_SEED": 42,
        "WEIGHT_INIT": "he",
    }

    save_path = plot_history(history, save_dir=tmp_path, hyperparams=hyperparams)

    assert save_path.name == "hidden-128_lr-0.01_bs-64_seed-42_init-he"
    assert (save_path / "loss_curve.png").exists()
    assert (save_path / "accuracy_curve.png").exists()
