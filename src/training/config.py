from __future__ import annotations

from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = WORKSPACE_ROOT / "data" / "processed"
OUTPUTS_TRAINING_DIR = WORKSPACE_ROOT / "outputs" / "training"

DEFAULT_HORIZON_HOURS = 48
DEFAULT_TARGET_COLUMN = f"target_{DEFAULT_HORIZON_HOURS}h"
DEFAULT_READY_DATASET_PATH = DATA_PROCESSED_DIR / f"ismn_forecasting_{DEFAULT_HORIZON_HOURS}h_ready.csv"
DEFAULT_FULL_DATASET_PATH = DATA_PROCESSED_DIR / f"ismn_forecasting_{DEFAULT_HORIZON_HOURS}h_full.csv"
TARGET_SOURCE_COLUMN = "sm_0.05m"
TIMESTAMP_COLUMN = "timestamp"
STATION_COLUMN = "station"

STATIC_COLUMNS = ["station", "timestamp", "latitude", "longitude", "elevation"]
CALENDAR_COLUMNS = ["hour", "dayofweek", "month", "dayofyear", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
BASE_DYNAMIC_COLUMNS = [
    "precipitation",
    "sm_0.05m",
    "sm_0.20m",
    "sm_0.50m",
    "ta_2.00m",
    "ts_0.05m",
    "ts_0.20m",
    "ts_0.50m",
]
SEQUENCE_FEATURES = [
    "sm_0.05m",
    "sm_0.20m",
    "sm_0.50m",
    "ta_2.00m",
    "ts_0.05m",
    "ts_0.20m",
    "ts_0.50m",
    "precipitation",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

SHORT_LAGS = [1, 3, 6, 12, 24]
MEDIUM_LAGS = [48, 72]
LONG_LAGS = [168]
