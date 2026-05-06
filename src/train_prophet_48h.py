from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.training.config import (
    DEFAULT_FULL_DATASET_PATH,
    DEFAULT_READY_DATASET_PATH,
    DEFAULT_TARGET_COLUMN,
    OUTPUTS_TRAINING_DIR,
    STATION_COLUMN,
    TARGET_SOURCE_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.training.data import chronological_split, load_full_prepared_dataset, load_prepared_dataset
from src.training.evaluation import compute_metrics, save_run_outputs, summarize_run_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prophet baseline for 48-hour soil moisture forecasting.")
    parser.add_argument("--changepoint-prior-scale", type=float, default=0.05)
    parser.add_argument("--seasonality-prior-scale", type=float, default=10.0)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_output_dir_name(base_name: str, output_tag: str | None) -> str:
    if output_tag:
        return f"{base_name}__{output_tag}"
    return base_name


def prepare_prophet_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame[[TIMESTAMP_COLUMN, TARGET_SOURCE_COLUMN]].dropna().copy()
    prepared = prepared.rename(columns={TIMESTAMP_COLUMN: "ds", TARGET_SOURCE_COLUMN: "y"})
    return prepared


def fit_station_prophet(train_frame: pd.DataFrame, forecast_end, changepoint_prior_scale: float, seasonality_prior_scale: float):
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise SystemExit("prophet is not installed. Install it before running Prophet.") from exc

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
    )
    model.fit(train_frame)
    future = pd.DataFrame({"ds": pd.date_range(train_frame["ds"].max() + pd.Timedelta(hours=1), forecast_end, freq="h")})
    forecast = model.predict(future)
    return pd.Series(forecast["yhat"].to_numpy(), index=pd.to_datetime(forecast["ds"]))


def align_forecast_to_target_rows(eval_frame: pd.DataFrame, forecast_series: pd.Series, horizon_hours: int = 48):
    target_times = pd.to_datetime(eval_frame[TIMESTAMP_COLUMN]) + pd.Timedelta(hours=horizon_hours)
    return forecast_series.reindex(target_times).to_numpy(dtype=float)


def main() -> Path:
    args = parse_args()
    split_source = load_prepared_dataset(DEFAULT_READY_DATASET_PATH)
    train_df, val_df, _ = chronological_split(split_source, train_fraction=0.7, val_fraction=0.15)
    train_end = train_df[TIMESTAMP_COLUMN].max()
    val_end = val_df[TIMESTAMP_COLUMN].max()
    full_dataset = load_full_prepared_dataset(DEFAULT_FULL_DATASET_PATH)

    rows = []
    prediction_frames = []
    skipped_stations = []
    skipped_reasons = {}
    for station, station_frame in full_dataset.groupby(STATION_COLUMN, sort=False):
        station_frame = station_frame.sort_values(TIMESTAMP_COLUMN)
        train = station_frame[station_frame[TIMESTAMP_COLUMN] <= train_end].copy()
        train_val = station_frame[station_frame[TIMESTAMP_COLUMN] <= val_end].copy()
        val_eval = station_frame[
            (station_frame[TIMESTAMP_COLUMN] > train_end)
            & (station_frame[TIMESTAMP_COLUMN] <= val_end)
            & station_frame[DEFAULT_TARGET_COLUMN].notna()
        ].copy()
        test_eval = station_frame[
            (station_frame[TIMESTAMP_COLUMN] > val_end)
            & station_frame[DEFAULT_TARGET_COLUMN].notna()
        ].copy()

        if len(train) < 50:
            skipped_stations.append(station)
            skipped_reasons[station] = "train_history_too_short"
            if args.debug:
                print(f"[skip] {station}: train_history_too_short")
            continue
        if len(val_eval) == 0:
            skipped_stations.append(station)
            skipped_reasons[station] = "val_eval_empty"
            if args.debug:
                print(f"[skip] {station}: val_eval_empty")
            continue
        if len(test_eval) == 0:
            skipped_stations.append(station)
            skipped_reasons[station] = "test_eval_empty"
            if args.debug:
                print(f"[skip] {station}: test_eval_empty")
            continue

        train_fit = prepare_prophet_training_frame(train)
        train_val_fit = prepare_prophet_training_frame(train_val)

        try:
            val_forecast = fit_station_prophet(
                train_fit,
                pd.to_datetime(val_eval[TIMESTAMP_COLUMN]).max() + pd.Timedelta(hours=48),
                args.changepoint_prior_scale,
                args.seasonality_prior_scale,
            )
            test_forecast = fit_station_prophet(
                train_val_fit,
                pd.to_datetime(test_eval[TIMESTAMP_COLUMN]).max() + pd.Timedelta(hours=48),
                args.changepoint_prior_scale,
                args.seasonality_prior_scale,
            )
        except Exception as exc:
            skipped_stations.append(station)
            skipped_reasons[station] = f"fit_failed:{type(exc).__name__}"
            if args.debug:
                print(f"[skip] {station}: fit_failed:{type(exc).__name__}")
            continue

        val_pred = align_forecast_to_target_rows(val_eval, val_forecast, horizon_hours=48)
        test_pred = align_forecast_to_target_rows(test_eval, test_forecast, horizon_hours=48)
        if not pd.notna(val_pred).all():
            skipped_stations.append(station)
            skipped_reasons[station] = "val_forecast_alignment_failed"
            if args.debug:
                print(f"[skip] {station}: val_forecast_alignment_failed")
            continue
        if not pd.notna(test_pred).all():
            skipped_stations.append(station)
            skipped_reasons[station] = "test_forecast_alignment_failed"
            if args.debug:
                print(f"[skip] {station}: test_forecast_alignment_failed")
            continue

        for split_name, eval_frame, predictions in [("val", val_eval, val_pred), ("test", test_eval, test_pred)]:
            metric_row = {"station": station, "split": split_name}
            metric_row.update(compute_metrics(eval_frame[DEFAULT_TARGET_COLUMN], predictions))
            rows.append(metric_row)

            prediction_frame = eval_frame[[STATION_COLUMN, TIMESTAMP_COLUMN]].copy()
            prediction_frame["split"] = split_name
            prediction_frame["y_true"] = eval_frame[DEFAULT_TARGET_COLUMN].to_numpy()
            prediction_frame["y_pred"] = predictions
            prediction_frame["station_name"] = station
            prediction_frames.append(prediction_frame)

    if not rows:
        raise SystemExit("Prophet produced no valid station forecasts.")

    run_metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    test_metrics = run_metrics.loc[run_metrics["split"] == "test", ["station", "rmse", "mae", "r2"]].rename(columns={"station": "run_id"})
    summary = summarize_run_metrics(test_metrics)

    output_dir = OUTPUTS_TRAINING_DIR / "prophet" / resolve_output_dir_name("prophet", args.output_tag)
    metadata = {
        "model": "prophet",
        "output_tag": args.output_tag,
        "changepoint_prior_scale": args.changepoint_prior_scale,
        "seasonality_prior_scale": args.seasonality_prior_scale,
        "target_column": DEFAULT_TARGET_COLUMN,
        "target_source_column": TARGET_SOURCE_COLUMN,
        "skipped_stations": skipped_stations,
        "skipped_reasons": skipped_reasons,
    }
    save_run_outputs(output_dir, run_metrics, predictions, summary, metadata)
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
