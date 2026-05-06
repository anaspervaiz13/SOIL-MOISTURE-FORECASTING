from __future__ import annotations

import numpy as np
import pandas as pd


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
        group = group.reset_index(drop=True)
        features = group[feature_columns].to_numpy(dtype=float)
        targets = group[target_column].to_numpy(dtype=float)

        for end_index in range(lookback - 1, len(group)):
            start_index = end_index - lookback + 1
            x_sequences.append(features[start_index : end_index + 1])
            y_values.append(targets[end_index])
            meta_rows.append(
                {
                    "station": station,
                    "timestamp": group.loc[end_index, "timestamp"],
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
