from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.training.config import DEFAULT_TARGET_COLUMN, OUTPUTS_TRAINING_DIR
from src.training.data import chronological_split, load_prepared_dataset, select_feature_columns
from src.training.evaluation import compute_metrics, save_run_outputs, summarize_run_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KNN ablation models for 48-hour soil moisture forecasting.")
    parser.add_argument(
        "--feature-group",
        default="full_multiscale",
        choices=["baseline_limited", "short_lags", "short_plus_medium", "full_multiscale"],
    )
    parser.add_argument("--neighbors", default="9")
    parser.add_argument("--weights", default="distance", choices=["uniform", "distance"])
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--output-tag", default=None)
    return parser.parse_args()


def parse_neighbor_values(values_text: str) -> list[int]:
    return [int(item.strip()) for item in values_text.split(",") if item.strip()]


def resolve_output_dir_name(feature_group: str, output_tag: str | None) -> str:
    if output_tag:
        return f"{feature_group}__{output_tag}"
    return feature_group


def build_xy(frame: pd.DataFrame, feature_columns: list[str], target_column: str):
    x_values = frame[feature_columns]
    y_values = frame[target_column]
    metadata = frame[["station", "timestamp"]].copy()
    return x_values, y_values, metadata


def main() -> Path:
    args = parse_args()
    neighbor_values = parse_neighbor_values(args.neighbors)
    dataset = load_prepared_dataset()
    feature_columns = select_feature_columns(dataset, args.feature_group, DEFAULT_TARGET_COLUMN)
    train_df, val_df, test_df = chronological_split(
        dataset,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    x_train, y_train, _ = build_xy(train_df, feature_columns, DEFAULT_TARGET_COLUMN)
    x_val, y_val, meta_val = build_xy(val_df, feature_columns, DEFAULT_TARGET_COLUMN)
    x_test, y_test, meta_test = build_xy(test_df, feature_columns, DEFAULT_TARGET_COLUMN)

    rows = []
    prediction_frames = []
    for neighbor_count in neighbor_values:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=neighbor_count, weights=args.weights)),
            ]
        )
        model.fit(x_train, y_train)

        for split_name, x_split, y_split, meta_split in [
            ("val", x_val, y_val, meta_val),
            ("test", x_test, y_test, meta_test),
        ]:
            predictions = model.predict(x_split)
            metric_row = {"neighbor_count": neighbor_count, "split": split_name}
            metric_row.update(compute_metrics(y_split, predictions))
            rows.append(metric_row)

            prediction_frame = meta_split.copy()
            prediction_frame["neighbor_count"] = neighbor_count
            prediction_frame["split"] = split_name
            prediction_frame["y_true"] = y_split.to_numpy()
            prediction_frame["y_pred"] = predictions
            prediction_frames.append(prediction_frame)

    run_metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    test_metrics = run_metrics.loc[run_metrics["split"] == "test", ["neighbor_count", "rmse", "mae", "r2"]].rename(
        columns={"neighbor_count": "run_id"}
    )
    summary = summarize_run_metrics(test_metrics)

    output_dir = OUTPUTS_TRAINING_DIR / "knn" / resolve_output_dir_name(args.feature_group, args.output_tag)
    metadata = {
        "model": "knn",
        "feature_group": args.feature_group,
        "output_tag": args.output_tag,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "neighbors": neighbor_values,
        "weights": args.weights,
        "target_column": DEFAULT_TARGET_COLUMN,
    }
    save_run_outputs(output_dir, run_metrics, predictions, summary, metadata)
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
