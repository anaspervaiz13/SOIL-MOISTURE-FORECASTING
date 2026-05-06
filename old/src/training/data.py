from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FULL_DATASET_PATH, READY_DATASET_PATH, STATION_COLUMN, TARGET_COLUMN, TIMESTAMP_COLUMN


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp


def load_ready_dataset(path=READY_DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    return df.sort_values([TIMESTAMP_COLUMN, STATION_COLUMN]).reset_index(drop=True)


def load_full_dataset(path=FULL_DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    return df.sort_values([STATION_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)


def build_time_splits(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> DatasetSplits:
    unique_times = np.array(sorted(df[TIMESTAMP_COLUMN].unique()))
    n_times = len(unique_times)
    if n_times < 3:
        raise ValueError("Need at least 3 unique timestamps to build train/val/test splits.")

    train_end_idx = max(1, int(n_times * train_ratio))
    val_end_idx = max(train_end_idx + 1, int(n_times * (train_ratio + val_ratio)))
    train_end_idx = min(train_end_idx, n_times - 2)
    val_end_idx = min(val_end_idx, n_times - 1)

    train_end = pd.Timestamp(unique_times[train_end_idx - 1])
    val_end = pd.Timestamp(unique_times[val_end_idx - 1])

    train = df[df[TIMESTAMP_COLUMN] <= train_end].copy()
    val = df[(df[TIMESTAMP_COLUMN] > train_end) & (df[TIMESTAMP_COLUMN] <= val_end)].copy()
    test = df[df[TIMESTAMP_COLUMN] > val_end].copy()

    return DatasetSplits(train=train, val=val, test=test, train_end=train_end, val_end=val_end)


def extract_xy(df: pd.DataFrame, feature_columns: list[str], target_column: str = TARGET_COLUMN):
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    meta = df[[STATION_COLUMN, TIMESTAMP_COLUMN]].copy()
    return X, y, meta


def make_sequence_arrays(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lookback: int = 168,
):
    X_list = []
    y_list = []
    meta_rows = []

    for station, group in df.groupby(STATION_COLUMN, sort=False):
        group = group.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        values = group[feature_columns + [target_column, TIMESTAMP_COLUMN]].copy()

        for end_idx in range(lookback - 1, len(values)):
            window = values.iloc[end_idx - lookback + 1 : end_idx + 1]
            if window[feature_columns].isna().any().any():
                continue

            target_value = values.iloc[end_idx][target_column]
            if pd.isna(target_value):
                continue

            X_list.append(window[feature_columns].to_numpy(dtype=np.float32))
            y_list.append(np.float32(target_value))
            meta_rows.append(
                {
                    STATION_COLUMN: station,
                    TIMESTAMP_COLUMN: values.iloc[end_idx][TIMESTAMP_COLUMN],
                }
            )

    X = np.stack(X_list) if X_list else np.empty((0, lookback, len(feature_columns)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta
