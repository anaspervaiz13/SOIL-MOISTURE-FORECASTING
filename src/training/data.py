from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_FULL_DATASET_PATH, DEFAULT_READY_DATASET_PATH, STATIC_COLUMNS
from .feature_sets import get_feature_group_columns


def load_prepared_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    path = DEFAULT_READY_DATASET_PATH if dataset_path is None else Path(dataset_path)
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_full_prepared_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    path = DEFAULT_FULL_DATASET_PATH if dataset_path is None else Path(dataset_path)
    return pd.read_csv(path, parse_dates=["timestamp"])


def select_feature_columns(frame: pd.DataFrame, group_name: str, target_column: str) -> list[str]:
    columns = get_feature_group_columns(frame.columns.tolist(), group_name)
    return [column for column in columns if column not in STATIC_COLUMNS and column != target_column]


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave room for a test split.")

    ordered = frame.sort_values(["timestamp", "station"]).reset_index(drop=True)
    unique_timestamps = ordered["timestamp"].drop_duplicates().sort_values().tolist()

    train_end_index = max(1, int(len(unique_timestamps) * train_fraction))
    val_end_index = max(train_end_index + 1, int(len(unique_timestamps) * (train_fraction + val_fraction)))

    train_cutoff = unique_timestamps[train_end_index - 1]
    val_cutoff = unique_timestamps[val_end_index - 1]

    train_df = ordered.loc[ordered["timestamp"] <= train_cutoff].reset_index(drop=True)
    val_df = ordered.loc[(ordered["timestamp"] > train_cutoff) & (ordered["timestamp"] <= val_cutoff)].reset_index(drop=True)
    test_df = ordered.loc[ordered["timestamp"] > val_cutoff].reset_index(drop=True)

    return train_df, val_df, test_df
