from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from training.config import OUTPUTS_TRAINING, TARGET_COLUMN
from training.data import build_time_splits, extract_xy, load_ready_dataset
from training.evaluation import regression_metrics, save_run_outputs, summarize_metric_runs
from training.feature_sets import get_feature_group


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost for 48-hour soil moisture forecasting.")
    parser.add_argument("--feature-group", default="weather_depth_temporal")
    parser.add_argument("--seeds", default="42,52,62,72,82")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    return parser.parse_args()


def main():
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("xgboost is not installed. Install it before running this script.") from exc

    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    df = load_ready_dataset()
    splits = build_time_splits(df)
    features = get_feature_group(args.feature_group)

    X_train, y_train, _ = extract_xy(splits.train, features, TARGET_COLUMN)
    X_val, y_val, meta_val = extract_xy(splits.val, features, TARGET_COLUMN)
    X_test, y_test, meta_test = extract_xy(splits.test, features, TARGET_COLUMN)

    metrics_rows = []
    prediction_frames = []
    for run_idx, seed in enumerate(seeds, start=1):
        model = XGBRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        for split_name, X_split, y_split, meta_split in [
            ("val", X_val, y_val, meta_val),
            ("test", X_test, y_test, meta_test),
        ]:
            preds = model.predict(X_split)
            row = {"run": run_idx, "seed": seed, "split": split_name}
            row.update(regression_metrics(y_split, preds))
            metrics_rows.append(row)

            pred_df = meta_split.copy()
            pred_df["run"] = run_idx
            pred_df["seed"] = seed
            pred_df["split"] = split_name
            pred_df["model"] = "xgboost"
            pred_df["y_true"] = y_split.to_numpy()
            pred_df["y_pred"] = preds
            prediction_frames.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df = summarize_metric_runs(metrics_df)
    output_dir = OUTPUTS_TRAINING / "xgboost"
    metadata = {
        "model": "xgboost",
        "feature_group": args.feature_group,
        "features": features,
        "seeds": seeds,
    }
    save_run_outputs(output_dir, metrics_df, predictions_df, summary_df, metadata)
    print(output_dir)


if __name__ == "__main__":
    main()
