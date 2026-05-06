from __future__ import annotations

import argparse

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.config import OUTPUTS_TRAINING, TARGET_COLUMN
from training.data import build_time_splits, extract_xy, load_ready_dataset
from training.evaluation import regression_metrics, save_run_outputs, summarize_metric_runs
from training.feature_sets import get_feature_group


def parse_args():
    parser = argparse.ArgumentParser(description="Train KNN for 48-hour soil moisture forecasting.")
    parser.add_argument("--feature-group", default="weather_depth_temporal")
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--weights", default="distance")
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_ready_dataset()
    splits = build_time_splits(df)
    features = get_feature_group(args.feature_group)

    X_train, y_train, _ = extract_xy(splits.train, features, TARGET_COLUMN)
    X_val, y_val, meta_val = extract_xy(splits.val, features, TARGET_COLUMN)
    X_test, y_test, meta_test = extract_xy(splits.test, features, TARGET_COLUMN)

    metrics_rows = []
    prediction_frames = []
    for run_idx in range(1, args.runs + 1):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=args.neighbors, weights=args.weights)),
            ]
        )
        model.fit(X_train, y_train)

        for split_name, X_split, y_split, meta_split in [
            ("val", X_val, y_val, meta_val),
            ("test", X_test, y_test, meta_test),
        ]:
            preds = model.predict(X_split)
            row = {"run": run_idx, "seed": None, "split": split_name}
            row.update(regression_metrics(y_split, preds))
            metrics_rows.append(row)

            pred_df = meta_split.copy()
            pred_df["run"] = run_idx
            pred_df["seed"] = None
            pred_df["split"] = split_name
            pred_df["model"] = "knn"
            pred_df["y_true"] = y_split.to_numpy()
            pred_df["y_pred"] = preds
            prediction_frames.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df = summarize_metric_runs(metrics_df)
    output_dir = OUTPUTS_TRAINING / "knn"
    metadata = {"model": "knn", "feature_group": args.feature_group, "features": features}
    save_run_outputs(output_dir, metrics_df, predictions_df, summary_df, metadata)
    print(output_dir)


if __name__ == "__main__":
    main()
