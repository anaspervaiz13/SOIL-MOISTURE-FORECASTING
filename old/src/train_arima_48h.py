from __future__ import annotations

import argparse
import warnings

import numpy as np

import pandas as pd

try:
    from training.config import OUTPUTS_TRAINING, STATION_COLUMN, TARGET_COLUMN, TARGET_SOURCE_COLUMN, TIMESTAMP_COLUMN
    from training.data import build_time_splits, load_full_dataset, load_ready_dataset
    from training.evaluation import aggregate_run_metrics, regression_metrics, save_run_outputs, summarize_metric_runs
except ModuleNotFoundError:
    from src.training.config import OUTPUTS_TRAINING, STATION_COLUMN, TARGET_COLUMN, TARGET_SOURCE_COLUMN, TIMESTAMP_COLUMN
    from src.training.data import build_time_splits, load_full_dataset, load_ready_dataset
    from src.training.evaluation import aggregate_run_metrics, regression_metrics, save_run_outputs, summarize_metric_runs


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARIMA baseline for 48-hour soil moisture forecasting.")
    parser.add_argument("--order", default="3,0,1")
    parser.add_argument("--seasonal-order", default="0,0,0,0")
    return parser.parse_args()


def prepare_source_series(df: pd.DataFrame) -> pd.Series:
    series = df.sort_values(TIMESTAMP_COLUMN).set_index(TIMESTAMP_COLUMN)[TARGET_SOURCE_COLUMN].copy()
    series.index = pd.DatetimeIndex(series.index)
    series = series.asfreq("h")
    return series.astype(float)


def align_forecast_to_target_rows(eval_df: pd.DataFrame, forecast_series: pd.Series, horizon_hours: int = 48) -> np.ndarray:
    target_times = pd.to_datetime(eval_df[TIMESTAMP_COLUMN]) + pd.Timedelta(hours=horizon_hours)
    aligned = forecast_series.reindex(target_times)
    return aligned.to_numpy(dtype=float)


def prepare_training_source(df: pd.DataFrame) -> pd.Series:
    series = prepare_source_series(df)
    # Interpolate within the historical series to stabilize the classical baseline on sensor gaps.
    return series.interpolate(method="time", limit_direction="both")


def safe_forecast(train_series: pd.Series, steps: int, order, seasonal_order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

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
        preds = result.forecast(steps=steps)

    preds = np.asarray(preds, dtype=float)
    if preds.shape[0] != steps or not np.isfinite(preds).all():
        return None
    return preds


def forecast_block(train_series: pd.Series, eval_df: pd.DataFrame, order, seasonal_order):
    if eval_df.empty:
        return None
    forecast_end = pd.to_datetime(eval_df[TIMESTAMP_COLUMN]).max() + pd.Timedelta(hours=48)
    train_end = train_series.index.max()
    steps = int((forecast_end - train_end) / pd.Timedelta(hours=1))
    if steps <= 0:
        return None

    preds = safe_forecast(train_series, steps, order, seasonal_order)
    if preds is None:
        return None

    forecast_index = pd.date_range(train_end + pd.Timedelta(hours=1), periods=steps, freq="h")
    forecast_series = pd.Series(preds, index=forecast_index)
    aligned = align_forecast_to_target_rows(eval_df, forecast_series, horizon_hours=48)
    if not np.isfinite(aligned).all():
        return None
    return aligned


def fit_and_forecast_station(station_df: pd.DataFrame, train_end, val_end, order, seasonal_order):
    train_hist = station_df[station_df[TIMESTAMP_COLUMN] <= train_end].copy()
    train_val_hist = station_df[station_df[TIMESTAMP_COLUMN] <= val_end].copy()
    val_eval = station_df[(station_df[TIMESTAMP_COLUMN] > train_end) & (station_df[TIMESTAMP_COLUMN] <= val_end) & station_df[TARGET_COLUMN].notna()].copy()
    test_eval = station_df[(station_df[TIMESTAMP_COLUMN] > val_end) & station_df[TARGET_COLUMN].notna()].copy()

    if len(train_hist) < 50 or len(val_eval) == 0 or len(test_eval) == 0:
        return None

    train_series = prepare_training_source(train_hist)
    val_pred = forecast_block(train_series, val_eval, order, seasonal_order)
    if val_pred is None:
        return None

    train_val_series = prepare_training_source(train_val_hist)
    test_pred = forecast_block(train_val_series, test_eval, order, seasonal_order)
    if test_pred is None:
        return None

    return {"val": (val_eval, val_pred), "test": (test_eval, test_pred)}


def main():
    args = parse_args()
    order = tuple(int(x.strip()) for x in args.order.split(","))
    seasonal_order = tuple(int(x.strip()) for x in args.seasonal_order.split(","))

    split_df = load_ready_dataset()
    splits = build_time_splits(split_df)
    df = load_full_dataset()

    metrics_rows = []
    prediction_frames = []
    skipped_stations = []
    for station, station_df in df.groupby(STATION_COLUMN, sort=False):
        result = fit_and_forecast_station(station_df, splits.train_end, splits.val_end, order, seasonal_order)
        if result is None:
            skipped_stations.append(station)
            continue

        for split_name in ["val", "test"]:
            split_df, preds = result[split_name]
            row = {"run": 1, "seed": None, "split": split_name, "station": station}
            row.update(regression_metrics(split_df[TARGET_COLUMN], preds))
            metrics_rows.append(row)

            pred_df = split_df[[STATION_COLUMN, TIMESTAMP_COLUMN]].copy()
            pred_df["run"] = 1
            pred_df["seed"] = None
            pred_df["split"] = split_name
            pred_df["model"] = "arima"
            pred_df["y_true"] = split_df[TARGET_COLUMN].to_numpy()
            pred_df["y_pred"] = preds
            prediction_frames.append(pred_df)

    if not metrics_rows:
        raise SystemExit("ARIMA produced no valid station forecasts. Try a different order.")

    metrics_df = aggregate_run_metrics(pd.DataFrame(metrics_rows))
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df = summarize_metric_runs(metrics_df)
    output_dir = OUTPUTS_TRAINING / "arima"
    metadata = {"model": "arima", "order": order, "seasonal_order": seasonal_order, "skipped_stations": skipped_stations}
    save_run_outputs(output_dir, metrics_df, predictions_df, summary_df, metadata)
    print(output_dir)


if __name__ == "__main__":
    main()
