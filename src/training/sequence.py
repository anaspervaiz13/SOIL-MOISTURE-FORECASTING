from __future__ import annotations

import pandas as pd

from .config import STATION_COLUMN, TIMESTAMP_COLUMN


def split_sequence_meta(meta: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp):
    train_mask = meta[TIMESTAMP_COLUMN] <= train_end
    val_mask = (meta[TIMESTAMP_COLUMN] > train_end) & (meta[TIMESTAMP_COLUMN] <= val_end)
    test_mask = meta[TIMESTAMP_COLUMN] > val_end
    return train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy()
