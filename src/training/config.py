from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_TRAINING = ROOT / "outputs" / "training"

READY_DATASET_PATH = DATA_PROCESSED / "ismn_forecasting_48h_ready.csv"
FULL_DATASET_PATH = DATA_PROCESSED / "ismn_forecasting_48h_full.csv"

TARGET_COLUMN = "target_48h"
TARGET_SOURCE_COLUMN = "sm_0.05m"
TIMESTAMP_COLUMN = "timestamp"
STATION_COLUMN = "station"

CORE_BASE_FEATURES = [
    "sm_0.05m",
    "sm_0.20m",
    "sm_0.50m",
    "ta_2.00m",
    "ts_0.05m",
    "ts_0.20m",
    "ts_0.50m",
    "precipitation",
]

CALENDAR_FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

LAG_HOURS = [1, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [6, 24, 48, 168]

DEPTH_FEATURES = ["sm_0.20m", "sm_0.50m"]
WEATHER_TEMP_FEATURES = ["ta_2.00m", "ts_0.05m", "ts_0.20m", "ts_0.50m", "precipitation"]

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
