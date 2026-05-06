from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

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
    parser = argparse.ArgumentParser(description="Train ARIMA baseline for 48-hour soil moisture forecasting.")
    parser.add_argument("--order", default="3,0,1")
    parser.add_argument("--seasonal-order", default="0,0,0,0")
    parser.add_argument("--max-history-hours", type=int, default=24 * 180)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def parse_order(order_text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in order_text.split(",") if item.strip())


def resolve_output_dir_name(base_name: str, output_tag: str | None) -> str:
    if output_tag:
        return f"{base_name}__{output_tag}"
    return base_name


def prepare_source_series(frame: pd.DataFrame) -> pd.Series:
    series = frame.sort_values(TIMESTAMP_COLUMN).set_index(TIMESTAMP_COLUMN)[TARGET_SOURCE_COLUMN].copy()
    series.index = pd.DatetimeIndex(series.index)
    return series.asfreq("h").astype(float)


def prepare_training_source(frame: pd.DataFrame) -> pd.Series:
    return prepare_source_series(frame).interpolate(method="time", limit_direction="both")


def limit_history_series(series: pd.Series, max_history_hours: int | None) -> pd.Series:
    if max_history_hours is None or max_history_hours <= 0:
        return series
    return series.tail(max_history_hours)


def align_forecast_to_target_rows(eval_frame: pd.DataFrame, forecast_series: pd.Series, horizon_hours: int = 48) -> np.ndarray:
    target_times = pd.to_datetime(eval_frame[TIMESTAMP_COLUMN]) + pd.Timedelta(hours=horizon_hours)
    return forecast_series.reindex(target_times).to_numpy(dtype=float)


def safe_forecast(train_series: pd.Series, steps: int, order, seasonal_order):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise SystemExit("statsmodels is not installed. Install it before running ARIMA.") from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                train_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            predictions = result.forecast(steps=steps)
    except Exception:
        return None

    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape[0] != steps or not np.isfinite(predictions).all():
        return None
    return predictions


def forecast_block(train_series: pd.Series, eval_frame: pd.DataFrame, order, seasonal_order, horizon_hours: int = 48):
    if eval_frame.empty:
        return None

    forecast_end = pd.to_datetime(eval_frame[TIMESTAMP_COLUMN]).max() + pd.Timedelta(hours=horizon_hours)
    train_end = train_series.index.max()
    steps = int((forecast_end - train_end) / pd.Timedelta(hours=1))
    if steps <= 0:
        return None

    predictions = safe_forecast(train_series, steps, order, seasonal_order)
    if predictions is None:
        return None

    forecast_index = pd.date_range(train_end + pd.Timedelta(hours=1), periods=steps, freq="h")
    forecast_series = pd.Series(predictions, index=forecast_index)
    aligned = align_forecast_to_target_rows(eval_frame, forecast_series, horizon_hours=horizon_hours)
    if not np.isfinite(aligned).all():
        return None
    return aligned


def fit_and_forecast_station(
    station_frame: pd.DataFrame,
    train_end,
    val_end,
    order,
    seasonal_order,
    max_history_hours: int,
):
    train_hist = station_frame[station_frame[TIMESTAMP_COLUMN] <= train_end].copy()
    train_val_hist = station_frame[station_frame[TIMESTAMP_COLUMN] <= val_end].copy()
    val_eval = station_frame[
        (station_frame[TIMESTAMP_COLUMN] > train_end)
        & (station_frame[TIMESTAMP_COLUMN] <= val_end)
        & station_frame[DEFAULT_TARGET_COLUMN].notna()
    ].copy()
    test_eval = station_frame[
        (station_frame[TIMESTAMP_COLUMN] > val_end)
        & station_frame[DEFAULT_TARGET_COLUMN].notna()
    ].copy()

    if len(train_hist) < 50:
        return None, "train_history_too_short"
    if len(val_eval) == 0:
        return None, "val_eval_empty"
    if len(test_eval) == 0:
        return None, "test_eval_empty"

    train_series = limit_history_series(prepare_training_source(train_hist), max_history_hours=max_history_hours)
    val_pred = forecast_block(train_series, val_eval, order, seasonal_order)
    if val_pred is None:
        return None, "val_forecast_failed"

    train_val_series = limit_history_series(prepare_training_source(train_val_hist), max_history_hours=max_history_hours)
    test_pred = forecast_block(train_val_series, test_eval, order, seasonal_order)
    if test_pred is None:
        return None, "test_forecast_failed"

    return {"val": (val_eval, val_pred), "test": (test_eval, test_pred)}, "ok"


def main() -> Path:
    args = parse_args()
    order = parse_order(args.order)
    seasonal_order = parse_order(args.seasonal_order)

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
        result, reason = fit_and_forecast_station(
            station_frame,
            train_end,
            val_end,
            order,
            seasonal_order,
            max_history_hours=args.max_history_hours,
        )
        if result is None:
            skipped_stations.append(station)
            skipped_reasons[station] = reason
            if args.debug:
                print(f"[skip] {station}: {reason}")
            continue

        for split_name in ["val", "test"]:
            eval_frame, predictions = result[split_name]
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
        if args.debug and skipped_reasons:
            print(f"skip reasons: {skipped_reasons}")
        raise SystemExit("ARIMA produced no valid station forecasts. Try a different order or smaller max-history-hours.")

    run_metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    test_metrics = run_metrics.loc[run_metrics["split"] == "test", ["station", "rmse", "mae", "r2"]].rename(columns={"station": "run_id"})
    summary = summarize_run_metrics(test_metrics)

    output_dir = OUTPUTS_TRAINING_DIR / "arima" / resolve_output_dir_name("arima", args.output_tag)
    metadata = {
        "model": "arima",
        "output_tag": args.output_tag,
        "order": order,
        "seasonal_order": seasonal_order,
        "max_history_hours": args.max_history_hours,
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
