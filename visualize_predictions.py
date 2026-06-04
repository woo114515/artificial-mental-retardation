"""
Visualize test-set predictions from a saved checkpoint.
"""

from __future__ import annotations

from config import CHECKPOINT_PATH, RANDOM_SEED
from data.mnist import X_test, y_test
from train import build_model_from_config, get_image_shape, prepare_data
from utils.checkpoint import load_checkpoint
from utils.prediction_visualization import plot_test_predictions


def main():
    image_shape = get_image_shape("mnist")
    input_dim = X_test.shape[1]
    model = build_model_from_config(input_dim=input_dim, image_shape=image_shape)
    metadata = load_checkpoint(model, CHECKPOINT_PATH)

    _, _, X_test_model = prepare_data(X_test, X_test, X_test, image_shape=image_shape)
    output_path = plot_test_predictions(
        model=model,
        X_display=X_test,
        X_model=X_test_model,
        y=y_test,
        save_path="experiments/predictions/test_predictions.png",
        num_samples=16,
        seed=RANDOM_SEED,
        image_shape=(image_shape[1], image_shape[2]),
    )

    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    print(f"Checkpoint metadata: {metadata}")
    print(f"Saved prediction visualization to {output_path}")


if __name__ == "__main__":
    main()
