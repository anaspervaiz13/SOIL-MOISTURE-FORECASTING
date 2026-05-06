from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd

from training.config import OUTPUTS_TRAINING, SEQUENCE_FEATURES, TARGET_COLUMN
from training.data import build_time_splits, load_full_dataset, make_sequence_arrays
from training.evaluation import regression_metrics, save_run_outputs, summarize_metric_runs
from training.sequence import split_sequence_meta


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM for 48-hour soil moisture forecasting.")
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


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

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled.")
        except Exception as exc:
            print(f"GPU setup warning: {exc}")
    else:
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


def main():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise SystemExit("tensorflow is not installed. Install it before running this script.") from exc

    args = parse_args()
    configure_tensorflow(args.threads)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    df = load_full_dataset()
    split_source = df.dropna(subset=[TARGET_COLUMN]).copy()
    splits = build_time_splits(split_source)

    X, y, meta = make_sequence_arrays(df, SEQUENCE_FEATURES, TARGET_COLUMN, lookback=args.lookback)
    train_mask, val_mask, test_mask = split_sequence_meta(meta, splits.train_end, splits.val_end)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val, meta_val = X[val_mask], y[val_mask], meta[val_mask].reset_index(drop=True)
    X_test, y_test, meta_test = X[test_mask], y[test_mask], meta[test_mask].reset_index(drop=True)

    import tensorflow as tf

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    metrics_rows = []
    prediction_frames = []
    for run_idx, seed in enumerate(seeds, start=1):
        set_seed(seed)
        model = build_model((X_train.shape[1], X_train.shape[2]))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True
            )
        ]
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
        )

        for split_name, X_split, y_split, meta_split in [
            ("val", X_val, y_val, meta_val),
            ("test", X_test, y_test, meta_test),
        ]:
            preds = model.predict(X_split, verbose=0).reshape(-1)
            row = {"run": run_idx, "seed": seed, "split": split_name}
            row.update(regression_metrics(y_split, preds))
            metrics_rows.append(row)

            pred_df = meta_split.copy()
            pred_df["run"] = run_idx
            pred_df["seed"] = seed
            pred_df["split"] = split_name
            pred_df["model"] = "lstm"
            pred_df["y_true"] = y_split
            pred_df["y_pred"] = preds
            prediction_frames.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df = summarize_metric_runs(metrics_df)
    output_dir = OUTPUTS_TRAINING / "lstm"
    metadata = {"model": "lstm", "lookback": args.lookback, "sequence_features": SEQUENCE_FEATURES, "seeds": seeds}
    save_run_outputs(output_dir, metrics_df, predictions_df, summary_df, metadata)
    print(output_dir)


if __name__ == "__main__":
    main()
