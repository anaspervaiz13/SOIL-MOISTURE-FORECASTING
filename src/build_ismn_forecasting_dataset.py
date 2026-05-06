from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MERGED_INPUT_PATH = WORKSPACE_ROOT / "data" / "processed" / "ismn_merged_hourly.csv"
PROCESSED_DATA_DIR = WORKSPACE_ROOT / "data" / "processed"
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs" / "ismn"
FULL_OUTPUT_PATH = PROCESSED_DATA_DIR / "ismn_forecasting_48h_full.csv"
READY_OUTPUT_PATH = PROCESSED_DATA_DIR / "ismn_forecasting_48h_ready.csv"
SUMMARY_OUTPUT_PATH = OUTPUTS_DIR / "final_dataset_summary.json"

STATIC_COLUMNS = ["station", "timestamp", "latitude", "longitude", "elevation"]
BASE_FEATURE_COLUMNS = [
    "precipitation",
    "sm_0.05m",
    "sm_0.20m",
    "sm_0.50m",
    "ta_2.00m",
    "ts_0.05m",
    "ts_0.20m",
    "ts_0.50m",
]
LAG_SOURCE_COLUMNS = BASE_FEATURE_COLUMNS
LAG_HOURS = [1, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [6, 24, 48, 168]
ROLLING_MEAN_WINDOWS = [24, 48, 168]
FORECAST_HORIZONS = [24, 48, 72]


def make_forecast_target_name(horizon_hours: int) -> str:
    return f"target_{horizon_hours}h"


def consolidate_precipitation(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["p_ecotech_rain_gauge"]
        .combine_first(frame["p_ott_pluvio2s_amount"])
        .combine_first(frame["p_vaisala_wxt510"])
    )


def expand_station_hourly_grid(frame: pd.DataFrame) -> pd.DataFrame:
    expanded_groups = []

    for station, group in frame.groupby("station", sort=False):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        full_timestamps = pd.date_range(
            start=ordered["timestamp"].min(),
            end=ordered["timestamp"].max(),
            freq="h",
        )

        expanded = ordered.set_index("timestamp").reindex(full_timestamps).reset_index()
        expanded = expanded.rename(columns={"index": "timestamp"})
        expanded["station"] = station

        for column in ["latitude", "longitude", "elevation"]:
            if column in expanded.columns:
                expanded[column] = expanded[column].ffill().bfill()

        expanded_groups.append(expanded)

    return pd.concat(expanded_groups, ignore_index=True)


def create_target_48h(frame: pd.DataFrame, horizon_hours: int = 48) -> pd.DataFrame:
    result = frame.sort_values(["station", "timestamp"]).copy()
    target_column = make_forecast_target_name(horizon_hours)
    result[target_column] = result.groupby("station")["sm_0.05m"].shift(-horizon_hours)
    return result


def build_model_ready_dataset(frame: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    if target_column is None:
        target_columns = [column for column in frame.columns if column.startswith("target_")]
        if not target_columns:
            return frame.dropna().reset_index(drop=True)
        target_column = target_columns[0]

    required_columns = [column for column in frame.columns if not column.startswith("target_")]
    required_columns.append(target_column)
    return frame.dropna(subset=required_columns).reset_index(drop=True)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = pd.to_datetime(result["timestamp"])
    result["hour"] = timestamp.dt.hour
    result["dayofweek"] = timestamp.dt.dayofweek
    result["month"] = timestamp.dt.month
    result["dayofyear"] = timestamp.dt.dayofyear
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["doy_sin"] = np.sin(2 * np.pi * result["dayofyear"] / 365.25)
    result["doy_cos"] = np.cos(2 * np.pi * result["dayofyear"] / 365.25)
    return result


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in LAG_SOURCE_COLUMNS:
        for lag in LAG_HOURS:
            result[f"{column}_lag_{lag}h"] = result.groupby("station")[column].shift(lag)
    return result


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    station_groups = result.groupby("station", group_keys=False)

    for window in ROLLING_WINDOWS:
        shifted_surface = station_groups["sm_0.05m"].shift(1)
        shifted_precipitation = station_groups["precipitation"].shift(1)
        result[f"sm_0.05m_roll_mean_{window}h"] = shifted_surface.groupby(result["station"]).rolling(window).mean().reset_index(level=0, drop=True)
        result[f"sm_0.05m_roll_std_{window}h"] = shifted_surface.groupby(result["station"]).rolling(window).std().reset_index(level=0, drop=True)
        result[f"precipitation_roll_sum_{window}h"] = shifted_precipitation.groupby(result["station"]).rolling(window).sum().reset_index(level=0, drop=True)

    for column in ["sm_0.20m", "sm_0.50m", "ta_2.00m", "ts_0.05m", "ts_0.20m", "ts_0.50m"]:
        shifted = station_groups[column].shift(1)
        for window in ROLLING_MEAN_WINDOWS:
            result[f"{column}_roll_mean_{window}h"] = shifted.groupby(result["station"]).rolling(window).mean().reset_index(level=0, drop=True)

    return result


def build_dataset_summary(full_frame: pd.DataFrame, ready_frame: pd.DataFrame, target_column: str) -> dict[str, object]:
    return {
        "target_column": target_column,
        "full_shape": {"rows": int(len(full_frame)), "columns": int(len(full_frame.columns))},
        "ready_shape": {"rows": int(len(ready_frame)), "columns": int(len(ready_frame.columns))},
        "ready_stations": ready_frame["station"].value_counts().sort_index().to_dict(),
        "ready_time_start": None if ready_frame.empty else ready_frame["timestamp"].min().isoformat(),
        "ready_time_end": None if ready_frame.empty else ready_frame["timestamp"].max().isoformat(),
        "missingness": full_frame.isna().mean().round(6).to_dict(),
    }


def save_summary(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_base_feature_dataset() -> pd.DataFrame:
    merged = pd.read_csv(MERGED_INPUT_PATH, parse_dates=["timestamp"])
    merged = merged.sort_values(["station", "timestamp"]).reset_index(drop=True)

    dataset = merged.drop(
        columns=[
            "p_ecotech_rain_gauge",
            "p_ott_pluvio2s_amount",
            "p_ott_pluvio2s_volume",
            "p_vaisala_wxt510",
        ]
    ).copy()
    dataset["precipitation"] = consolidate_precipitation(merged)
    dataset = expand_station_hourly_grid(dataset)
    dataset = add_calendar_features(dataset)
    dataset = add_lag_features(dataset)
    dataset = add_rolling_features(dataset)
    return dataset


def build_and_save_forecasting_dataset(horizon_hours: int = 48) -> dict[str, object]:
    target_column = make_forecast_target_name(horizon_hours)
    dataset = create_target_48h(build_base_feature_dataset(), horizon_hours=horizon_hours)
    ready = build_model_ready_dataset(dataset, target_column=target_column)

    full_output_path = PROCESSED_DATA_DIR / f"ismn_forecasting_{horizon_hours}h_full.csv"
    ready_output_path = PROCESSED_DATA_DIR / f"ismn_forecasting_{horizon_hours}h_ready.csv"
    summary_output_path = OUTPUTS_DIR / f"final_dataset_summary_{horizon_hours}h.json"

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(full_output_path, index=False)
    ready.to_csv(ready_output_path, index=False)

    summary = build_dataset_summary(dataset, ready, target_column=target_column)
    save_summary(summary, summary_output_path)

    if horizon_hours == 48:
        dataset.to_csv(FULL_OUTPUT_PATH, index=False)
        ready.to_csv(READY_OUTPUT_PATH, index=False)
        save_summary(summary, SUMMARY_OUTPUT_PATH)

    return summary


def main() -> None:
    for horizon_hours in FORECAST_HORIZONS:
        summary = build_and_save_forecasting_dataset(horizon_hours=horizon_hours)
        print(
            f"Saved {horizon_hours}h forecasting datasets: "
            f"full={summary['full_shape']['rows']}x{summary['full_shape']['columns']}, "
            f"ready={summary['ready_shape']['rows']}x{summary['ready_shape']['columns']}"
        )


if __name__ == "__main__":
    main()
