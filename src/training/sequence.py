from __future__ import annotations

import numpy as np
import pandas as pd


def _iter_contiguous_hourly_runs(group: pd.DataFrame):
    ordered = group.sort_values("timestamp").reset_index(drop=True)
    run_start = 0

    for index in range(1, len(ordered)):
        gap = ordered.loc[index, "timestamp"] - ordered.loc[index - 1, "timestamp"]
        if gap != pd.Timedelta(hours=1):
            yield ordered.iloc[run_start:index].reset_index(drop=True)
            run_start = index

    if len(ordered) > 0:
        yield ordered.iloc[run_start:].reset_index(drop=True)


def fit_feature_scaler(x_train: np.ndarray) -> dict[str, np.ndarray]:
    mean = x_train.mean(axis=(0, 1))
    std = x_train.std(axis=(0, 1))
    std = np.where(std == 0.0, 1.0, std)
    return {"mean": mean, "std": std}


def apply_feature_scaler(x_values: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (x_values - stats["mean"]) / stats["std"]


def fit_target_scaler(y_train: np.ndarray) -> dict[str, float]:
    mean = float(y_train.mean())
    std = float(y_train.std())
    if std == 0.0:
        std = 1.0
    return {"mean": mean, "std": std}


def apply_target_scaler(y_values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return (y_values - stats["mean"]) / stats["std"]


def inverse_target_scaler(y_values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return (y_values * stats["std"]) + stats["mean"]


def make_sequence_arrays(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lookback: int,
):
    x_sequences = []
    y_values = []
    meta_rows = []

    ordered = frame.sort_values(["station", "timestamp"]).reset_index(drop=True)
    for station, group in ordered.groupby("station", sort=False):
        for run in _iter_contiguous_hourly_runs(group):
            features = run[feature_columns].to_numpy(dtype=float)
            targets = run[target_column].to_numpy(dtype=float)

            for end_index in range(lookback - 1, len(run)):
                start_index = end_index - lookback + 1
                x_sequences.append(features[start_index : end_index + 1])
                y_values.append(targets[end_index])
                meta_rows.append(
                    {
                        "station": station,
                        "timestamp": run.loc[end_index, "timestamp"],
                    }
                )

    x_array = np.asarray(x_sequences, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    meta = pd.DataFrame(meta_rows)
    return x_array, y_array, meta


def split_sequence_meta(meta: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp):
    train_mask = meta["timestamp"] <= train_end
    val_mask = (meta["timestamp"] > train_end) & (meta["timestamp"] <= val_end)
    test_mask = meta["timestamp"] > val_end
    return train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy()
