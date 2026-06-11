"""
Save and load model parameters.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np


def _trainable_params(model):
    return [param for param, _ in model.params_and_grads()]


def _format_value(value) -> str:
    if isinstance(value, list):
        if not value:
            return "none"
        return "-".join(str(item) for item in value)
    return str(value)


def _format_key_for_path(key: str) -> str:
    key_names = {
        "MODEL_TYPE": "model_type",
        "HIDDEN_DIMS": "hidden",
        "LEARNING_RATE": "lr",
        "BATCH_SIZE": "bs",
        "NUM_EPOCHS": "epochs",
        "RANDOM_SEED": "seed",
        "WEIGHT_INIT": "init",
        "OPTIMIZER": "opt",
        "ACTIVATION": "act",
        "CNN_OUT_CHANNELS": "cnn_out_channels",
        "CNN_KERNEL_SIZE": "cnn_kernel_size",
        "CNN_STRIDE": "cnn_stride",
        "CNN_PADDING": "cnn_padding",
        "POOL_KERNEL_SIZE": "pool_kernel_size",
        "POOL_STRIDE": "pool_stride",
        "MOMENTUM": "momentum",
        "BETA1": "beta1",
        "BETA2": "beta2",
        "EPS": "eps",
    }
    return key_names.get(key, key.lower())


def _short_run_name(hyperparams: dict, full_run_name: str, max_stem_length: int) -> str:
    """
    Build a compact run name when the full hyperparameter filename is too long.

    The complete hyperparameters are still stored in checkpoint metadata; the
    filename only needs enough information to identify the run at a glance.
    """
    short_key_names = {
        "DATASET": "ds",
        "MODEL_TYPE": "model",
        "BATCH_SIZE": "bs",
        "HIDDEN_DIMS": "h",
        "LEARNING_RATE": "lr",
        "OPTIMIZER": "opt",
        "ACTIVATION": "act",
        "RANDOM_SEED": "seed",
        "WEIGHT_INIT": "init",
        "CNN_OUT_CHANNELS": "cout",
        "CNN_KERNEL_SIZE": "ck",
        "MOMENTUM": "mom",
        "BETA1": "b1",
        "BETA2": "b2",
    }
    preferred_keys = [
        "DATASET",
        "MODEL_TYPE",
        "BATCH_SIZE",
        "HIDDEN_DIMS",
        "LEARNING_RATE",
        "OPTIMIZER",
        "ACTIVATION",
        "RANDOM_SEED",
        "WEIGHT_INIT",
        "CNN_OUT_CHANNELS",
        "CNN_KERNEL_SIZE",
        "MOMENTUM",
        "BETA1",
        "BETA2",
    ]

    parts = []
    for key in preferred_keys:
        if key in hyperparams:
            parts.append(f"{short_key_names[key]}-{_format_value(hyperparams[key])}")

    digest = hashlib.sha1(full_run_name.encode("utf-8")).hexdigest()[:10]
    short_name = "_".join(parts + [digest])

    if len(short_name) > max_stem_length:
        short_name = f"{short_name[:max_stem_length - 11]}_{digest}"

    return short_name


def checkpoint_path_from_hyperparams(
    hyperparams: dict,
    save_dir: str = "checkpoints",
    suffix: str = ".npz",
    max_filename_length: int = 180,
) -> Path:
    """
    Build a checkpoint path whose filename contains the experiment hyperparameters.
    """
    run_name = "_".join(
        f"{_format_key_for_path(key)}-{_format_value(value)}"
        for key, value in hyperparams.items()
    )

    if len(f"{run_name}{suffix}") > max_filename_length:
        run_name = _short_run_name(
            hyperparams=hyperparams,
            full_run_name=run_name,
            max_stem_length=max_filename_length - len(suffix),
        )

    return Path(save_dir) / f"{run_name}{suffix}"


def save_checkpoint(model, path: str = "checkpoints/latest_model.npz", metadata: dict | None = None) -> Path:
    """
    Save model parameters to a compressed NumPy checkpoint.

    Args:
        model: Model exposing params_and_grads().
        path: Output .npz checkpoint path.
        metadata: Optional JSON-serializable metadata.

    Returns:
        Path to the saved checkpoint.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    params = _trainable_params(model)
    arrays = {f"param_{index}": param.copy() for index, param in enumerate(params)}
    arrays["num_params"] = np.array(len(params), dtype=np.int64)
    arrays["metadata_json"] = np.array(json.dumps(metadata or {}, ensure_ascii=False))

    np.savez_compressed(checkpoint_path, **arrays)
    return checkpoint_path


def load_checkpoint(model, path: str = "checkpoints/latest_model.npz", strict: bool = True) -> dict:
    """
    Load model parameters from a checkpoint.

    Args:
        model: Model exposing params_and_grads().
        path: Input .npz checkpoint path.
        strict: If True, require the same parameter count and shapes.

    Returns:
        Metadata dictionary saved with the checkpoint.
    """
    checkpoint_path = Path(path)
    params = _trainable_params(model)

    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        saved_num_params = int(checkpoint["num_params"])

        if strict and saved_num_params != len(params):
            raise ValueError(
                f"Checkpoint has {saved_num_params} parameters, but model has {len(params)}."
            )

        num_to_load = min(saved_num_params, len(params))
        for index in range(num_to_load):
            saved_param = checkpoint[f"param_{index}"]
            target_param = params[index]

            if strict and saved_param.shape != target_param.shape:
                raise ValueError(
                    f"Shape mismatch for param_{index}: checkpoint shape {saved_param.shape}, "
                    f"model shape {target_param.shape}."
                )

            target_param[...] = saved_param

        metadata = json.loads(str(checkpoint["metadata_json"]))

    return metadata
