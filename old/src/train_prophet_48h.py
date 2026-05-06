from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description="Train Prophet baseline for 48-hour soil moisture forecasting.")
    parser.add_argument("--changepoint-prior-scale", type=float, default=0.05)
    parser.add_argument("--seasonality-prior-scale", type=float, default=10.0)
    return parser.parse_args()


def fit_station_prophet(train_df, forecast_end, args):
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise SystemExit("prophet is not installed. Install it before running this script.") from exc

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=args.changepoint_prior_scale,
        seasonality_prior_scale=args.seasonality_prior_scale,
    )
    model.fit(train_df.rename(columns={TIMESTAMP_COLUMN: "ds", TARGET_SOURCE_COLUMN: "y"})[["ds", "y"]])
    future = pd.DataFrame({"ds": pd.date_range(train_df[TIMESTAMP_COLUMN].max() + pd.Timedelta(hours=1), forecast_end, freq="h")})
    forecast = model.predict(future)
    return pd.Series(forecast["yhat"].to_numpy(), index=pd.to_datetime(forecast["ds"]))


def align_forecast_to_target_rows(eval_df: pd.DataFrame, forecast_series: pd.Series, horizon_hours: int = 48):
    target_times = pd.to_datetime(eval_df[TIMESTAMP_COLUMN]) + pd.Timedelta(hours=horizon_hours)
    aligned = forecast_series.reindex(target_times)
    return aligned.to_numpy(dtype=float)


def main():
    args = parse_args()
    split_df = load_ready_dataset()
    splits = build_time_splits(split_df)
    df = load_full_dataset()

    metrics_rows = []
    prediction_frames = []

    for station, station_df in df.groupby(STATION_COLUMN, sort=False):
        station_df = station_df.sort_values(TIMESTAMP_COLUMN)
        train = station_df[station_df[TIMESTAMP_COLUMN] <= splits.train_end].copy()
        train_val = station_df[station_df[TIMESTAMP_COLUMN] <= splits.val_end].copy()
        val = station_df[(station_df[TIMESTAMP_COLUMN] > splits.train_end) & (station_df[TIMESTAMP_COLUMN] <= splits.val_end) & station_df[TARGET_COLUMN].notna()].copy()
        test = station_df[(station_df[TIMESTAMP_COLUMN] > splits.val_end) & station_df[TARGET_COLUMN].notna()].copy()
        if len(train) < 50 or len(val) == 0 or len(test) == 0:
            continue

        train_fit = train[[TIMESTAMP_COLUMN, TARGET_SOURCE_COLUMN]].dropna().copy()
        train_val_fit = train_val[[TIMESTAMP_COLUMN, TARGET_SOURCE_COLUMN]].dropna().copy()
        val_forecast = fit_station_prophet(train_fit, pd.to_datetime(val[TIMESTAMP_COLUMN]).max() + pd.Timedelta(hours=48), args)
        test_forecast = fit_station_prophet(train_val_fit, pd.to_datetime(test[TIMESTAMP_COLUMN]).max() + pd.Timedelta(hours=48), args)
        val_pred = align_forecast_to_target_rows(val, val_forecast, horizon_hours=48)
        test_pred = align_forecast_to_target_rows(test, test_forecast, horizon_hours=48)
        if not pd.notna(val_pred).all() or not pd.notna(test_pred).all():
            continue

        for split_name, split_df, preds in [("val", val, val_pred), ("test", test, test_pred)]:
            row = {"run": 1, "seed": None, "split": split_name, "station": station}
            row.update(regression_metrics(split_df[TARGET_COLUMN], preds))
            metrics_rows.append(row)

            pred_df = split_df[[STATION_COLUMN, TIMESTAMP_COLUMN]].copy()
            pred_df["run"] = 1
            pred_df["seed"] = None
            pred_df["split"] = split_name
            pred_df["model"] = "prophet"
            pred_df["y_true"] = split_df[TARGET_COLUMN].to_numpy()
            pred_df["y_pred"] = preds
            prediction_frames.append(pred_df)

    if not metrics_rows:
        raise SystemExit("Prophet produced no valid station forecasts.")

    metrics_df = aggregate_run_metrics(pd.DataFrame(metrics_rows))
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df = summarize_metric_runs(metrics_df)
    output_dir = OUTPUTS_TRAINING / "prophet"
    metadata = {"model": "prophet", "changepoint_prior_scale": args.changepoint_prior_scale, "seasonality_prior_scale": args.seasonality_prior_scale}
    save_run_outputs(output_dir, metrics_df, predictions_df, summary_df, metadata)
    print(output_dir)


if __name__ == "__main__":
    main()
