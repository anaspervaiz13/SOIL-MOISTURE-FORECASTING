from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "ismn_merged_hourly.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
SUMMARY_DIR = ROOT / "outputs" / "ismn"
LOG_PATH = ROOT / "logs" / "ismn_processing_log.md"

TARGET_COL = "sm_0.05m"
HORIZON_HOURS = 48
LAGS = [1, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [6, 24, 48, 168]
ROLL_MEAN_VARS = ["sm_0.20m", "sm_0.50m", "ta_2.00m", "ts_0.05m", "ts_0.20m", "ts_0.50m"]
DYNAMIC_VARS = [TARGET_COL, "sm_0.20m", "sm_0.50m", "ta_2.00m", "ts_0.05m", "ts_0.20m", "ts_0.50m", "precipitation"]


def append_log(text: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text)


def consolidate_precipitation(df: pd.DataFrame) -> pd.Series:
    precip = df["p_ecotech_rain_gauge"].copy()
    precip = precip.fillna(df["p_ott_pluvio2s_amount"])
    precip = precip.fillna(df["p_vaisala_wxt510"])
    return precip.astype("float32")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["timestamp"].dt.hour
    dayofweek = df["timestamp"].dt.dayofweek
    month = df["timestamp"].dt.month
    dayofyear = df["timestamp"].dt.dayofyear

    df["hour"] = hour.astype("int16")
    df["dayofweek"] = dayofweek.astype("int8")
    df["month"] = month.astype("int8")
    df["dayofyear"] = dayofyear.astype("int16")

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 366).astype("float32")
    df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 366).astype("float32")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading merged dataset...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    print("Consolidating precipitation...")
    df["precipitation"] = consolidate_precipitation(df)

    print("Building 48h target...")
    station_group = df.groupby("station", sort=False)
    df["target_48h"] = station_group[TARGET_COL].shift(-HORIZON_HOURS).astype("float32")

    print("Adding calendar features...")
    df = add_calendar_features(df)

    print("Adding lag features...")
    for var in DYNAMIC_VARS:
        for lag in LAGS:
            df[f"{var}_lag_{lag}h"] = station_group[var].shift(lag).astype("float32")

    print("Adding rolling features...")
    shifted_target = station_group[TARGET_COL].shift(1)
    shifted_precip = station_group["precipitation"].shift(1)
    for window in ROLL_WINDOWS:
        df[f"{TARGET_COL}_roll_mean_{window}h"] = (
            shifted_target.groupby(df["station"], sort=False).rolling(window).mean().reset_index(level=0, drop=True).astype("float32")
        )
        df[f"{TARGET_COL}_roll_std_{window}h"] = (
            shifted_target.groupby(df["station"], sort=False).rolling(window).std().reset_index(level=0, drop=True).astype("float32")
        )
        df[f"precipitation_roll_sum_{window}h"] = (
            shifted_precip.groupby(df["station"], sort=False).rolling(window).sum().reset_index(level=0, drop=True).astype("float32")
        )

    for var in ROLL_MEAN_VARS:
        shifted_var = station_group[var].shift(1)
        for window in [24, 48, 168]:
            df[f"{var}_roll_mean_{window}h"] = (
                shifted_var.groupby(df["station"], sort=False).rolling(window).mean().reset_index(level=0, drop=True).astype("float32")
            )

    drop_precip_cols = ["p_ecotech_rain_gauge", "p_ott_pluvio2s_amount", "p_ott_pluvio2s_volume", "p_vaisala_wxt510"]
    full_df = df.drop(columns=drop_precip_cols)

    core_required = [
        "station",
        "timestamp",
        "latitude",
        "longitude",
        "elevation",
        TARGET_COL,
        "sm_0.20m",
        "sm_0.50m",
        "ta_2.00m",
        "ts_0.05m",
        "ts_0.20m",
        "ts_0.50m",
        "precipitation",
        "target_48h",
    ]
    feature_cols = [col for col in full_df.columns if col not in {"station", "timestamp"}]
    ready_required = core_required + [col for col in feature_cols if "_lag_" in col or "_roll_" in col or col in {"hour", "dayofweek", "month", "dayofyear", "hour_sin", "hour_cos", "doy_sin", "doy_cos"}]
    ready_df = full_df.dropna(subset=ready_required).reset_index(drop=True)

    full_path = OUTPUT_DIR / "ismn_forecasting_48h_full.csv"
    ready_path = OUTPUT_DIR / "ismn_forecasting_48h_ready.csv"
    full_df.to_csv(full_path, index=False)
    ready_df.to_csv(ready_path, index=False)

    station_counts = ready_df["station"].value_counts().to_dict()
    summary = {
        "target_definition": "Predict sm_0.05m 48 hours ahead within each station.",
        "full_rows": int(len(full_df)),
        "full_columns": int(len(full_df.columns)),
        "ready_rows": int(len(ready_df)),
        "ready_columns": int(len(ready_df.columns)),
        "stations": sorted(full_df["station"].unique().tolist()),
        "ready_station_counts": station_counts,
        "time_range_full": {
            "start": str(full_df["timestamp"].min()),
            "end": str(full_df["timestamp"].max()),
        },
        "time_range_ready": {
            "start": str(ready_df["timestamp"].min()) if len(ready_df) else None,
            "end": str(ready_df["timestamp"].max()) if len(ready_df) else None,
        },
        "lag_hours": LAGS,
        "rolling_windows_hours": ROLL_WINDOWS,
    }
    with (SUMMARY_DIR / "final_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    append_log("\n## Step 16: 48-hour forecasting dataset built\n\n")
    append_log("- Read merged hourly ISMN dataset.\n")
    append_log("- Created one consolidated `precipitation` feature using locked sensor priority rule.\n")
    append_log(f"- Created `target_48h` as future `{TARGET_COL}` shifted by `-{HORIZON_HOURS}` hours within each station.\n")
    append_log("- Added calendar, multi-scale lag, and rolling-window features.\n")
    append_log(f"- Saved full engineered dataset to `{full_path.as_posix()}`.\n")
    append_log(f"- Saved model-ready filtered dataset to `{ready_path.as_posix()}`.\n")
    append_log(f"- Full dataset shape: `{full_df.shape[0]:,}` rows x `{full_df.shape[1]}` columns.\n")
    append_log(f"- Ready dataset shape: `{ready_df.shape[0]:,}` rows x `{ready_df.shape[1]}` columns.\n")

    print(f"Saved full dataset: {full_path}")
    print(f"Saved ready dataset: {ready_path}")
    print(f"Full shape: {full_df.shape}")
    print(f"Ready shape: {ready_df.shape}")
    print(f"Ready station counts: {station_counts}")


if __name__ == "__main__":
    main()
