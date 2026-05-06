from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.training.config import DEFAULT_TARGET_COLUMN, OUTPUTS_TRAINING_DIR, SEQUENCE_FEATURES
from src.training.data import chronological_split, load_full_prepared_dataset
from src.training.evaluation import compute_metrics, save_run_outputs, summarize_run_metrics
from src.training.sequence import make_sequence_arrays, split_sequence_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM for 48-hour soil moisture forecasting.")
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--output-tag", default=None)
    return parser.parse_args()


def parse_seeds(seeds_text: str) -> list[int]:
    return [int(item.strip()) for item in seeds_text.split(",") if item.strip()]


def resolve_output_dir_name(base_name: str, output_tag: str | None) -> str:
    if output_tag:
        return f"{base_name}__{output_tag}"
    return base_name


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def configure_tensorflow(threads: int):
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    print(f"TensorFlow devices | CPU: {len(cpus)} | GPU: {len(gpus)}")
    if not gpus:
        print("GPU unavailable. Running CPU-only.")


def build_model(input_shape):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def main() -> Path:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("tensorflow is not installed. Install it before running this script.") from exc

    args = parse_args()
    configure_tensorflow(args.threads)
    seeds = parse_seeds(args.seeds)

    full_dataset = load_full_prepared_dataset()
    split_source = full_dataset.dropna(subset=[DEFAULT_TARGET_COLUMN]).copy()
    train_df, val_df, test_df = chronological_split(split_source, train_fraction=0.7, val_fraction=0.15)
    train_end = train_df["timestamp"].max()
    val_end = val_df["timestamp"].max()

    sequence_source = full_dataset.dropna(subset=SEQUENCE_FEATURES + [DEFAULT_TARGET_COLUMN]).copy()
    x_values, y_values, meta = make_sequence_arrays(sequence_source, SEQUENCE_FEATURES, DEFAULT_TARGET_COLUMN, args.lookback)
    train_mask, val_mask, test_mask = split_sequence_meta(meta, train_end=train_end, val_end=val_end)

    x_train, y_train = x_values[train_mask], y_values[train_mask]
    x_val, y_val, meta_val = x_values[val_mask], y_values[val_mask], meta[val_mask].reset_index(drop=True)
    x_test, y_test, meta_test = x_values[test_mask], y_values[test_mask], meta[test_mask].reset_index(drop=True)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    rows = []
    prediction_frames = []
    for seed in seeds:
        set_seed(seed)
        model = build_model((x_train.shape[1], x_train.shape[2]))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
            )
        ]
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
        )

        for split_name, x_split, y_split, meta_split in [
            ("val", x_val, y_val, meta_val),
            ("test", x_test, y_test, meta_test),
        ]:
            predictions = model.predict(x_split, verbose=0).reshape(-1)
            metric_row = {"seed": seed, "split": split_name}
            metric_row.update(compute_metrics(y_split, predictions))
            rows.append(metric_row)

            prediction_frame = meta_split.copy()
            prediction_frame["seed"] = seed
            prediction_frame["split"] = split_name
            prediction_frame["y_true"] = y_split
            prediction_frame["y_pred"] = predictions
            prediction_frames.append(prediction_frame)

    run_metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    test_metrics = run_metrics.loc[run_metrics["split"] == "test", ["seed", "rmse", "mae", "r2"]].rename(columns={"seed": "run_id"})
    summary = summarize_run_metrics(test_metrics)

    output_dir = OUTPUTS_TRAINING_DIR / "lstm" / resolve_output_dir_name("lstm", args.output_tag)
    metadata = {
        "model": "lstm",
        "output_tag": args.output_tag,
        "lookback": args.lookback,
        "sequence_features": SEQUENCE_FEATURES,
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "target_column": DEFAULT_TARGET_COLUMN,
    }
    save_run_outputs(output_dir, run_metrics, predictions, summary, metadata)
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
