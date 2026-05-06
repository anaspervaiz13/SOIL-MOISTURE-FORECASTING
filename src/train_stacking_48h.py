from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LinearRegression

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.training.config import OUTPUTS_TRAINING_DIR
from src.training.evaluation import compute_metrics, save_run_outputs, summarize_run_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validation-fit stacking ensemble for 48-hour forecasting.")
    parser.add_argument("--xgboost-experiment", default="full_multiscale__strong")
    parser.add_argument("--knn-experiment", default="baseline_limited__light")
    parser.add_argument("--prophet-experiment", default="prophet__safe")
    parser.add_argument("--output-tag", default="aligned")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_xgboost_mean_predictions(experiment_dir: Path) -> pd.DataFrame:
    predictions = pd.read_csv(experiment_dir / "predictions.csv")
    grouped = (
        predictions.groupby(["station", "timestamp", "split", "y_true"], as_index=False)["y_pred"]
        .mean()
        .rename(columns={"y_pred": "xgboost_pred"})
    )
    return grouped


def select_best_knn_neighbor(experiment_dir: Path) -> int:
    run_metrics = pd.read_csv(experiment_dir / "run_metrics.csv")
    val_rows = run_metrics.loc[run_metrics["split"] == "val"].copy()
    best_row = val_rows.sort_values(["rmse", "neighbor_count"]).iloc[0]
    return int(best_row["neighbor_count"])


def load_best_knn_predictions(experiment_dir: Path) -> tuple[pd.DataFrame, int]:
    best_neighbor = select_best_knn_neighbor(experiment_dir)
    predictions = pd.read_csv(experiment_dir / "predictions.csv")
    filtered = predictions.loc[predictions["neighbor_count"] == best_neighbor].copy()
    filtered = filtered[["station", "timestamp", "split", "y_true", "y_pred"]].rename(columns={"y_pred": "knn_pred"})
    return filtered, best_neighbor


def load_prophet_predictions(experiment_dir: Path) -> pd.DataFrame:
    predictions = pd.read_csv(experiment_dir / "predictions.csv")
    columns = ["station", "timestamp", "split", "y_true", "y_pred"]
    filtered = predictions[columns].copy().rename(columns={"y_pred": "prophet_pred"})
    return filtered


def build_aligned_stack_frame(
    xgboost_predictions: pd.DataFrame,
    knn_predictions: pd.DataFrame,
    prophet_predictions: pd.DataFrame,
) -> pd.DataFrame:
    merged = xgboost_predictions.merge(
        knn_predictions,
        on=["station", "timestamp", "split", "y_true"],
        how="inner",
    )
    merged = merged.merge(
        prophet_predictions,
        on=["station", "timestamp", "split", "y_true"],
        how="inner",
    )
    return merged.sort_values(["split", "station", "timestamp"]).reset_index(drop=True)


def main() -> Path:
    args = parse_args()
    xgboost_dir = OUTPUTS_TRAINING_DIR / "xgboost" / args.xgboost_experiment
    knn_dir = OUTPUTS_TRAINING_DIR / "knn" / args.knn_experiment
    prophet_dir = OUTPUTS_TRAINING_DIR / "prophet" / args.prophet_experiment

    xgb_predictions = load_xgboost_mean_predictions(xgboost_dir)
    knn_predictions, best_neighbor = load_best_knn_predictions(knn_dir)
    prophet_predictions = load_prophet_predictions(prophet_dir)
    stacked = build_aligned_stack_frame(xgb_predictions, knn_predictions, prophet_predictions)

    feature_columns = ["xgboost_pred", "knn_pred", "prophet_pred"]
    train_meta = stacked.loc[stacked["split"] == "val"].copy()
    test_meta = stacked.loc[stacked["split"] == "test"].copy()

    model = LinearRegression()
    model.fit(train_meta[feature_columns], train_meta["y_true"])

    run_rows = []
    prediction_frames = []
    for split_name, frame in [("val", train_meta), ("test", test_meta)]:
        predictions = model.predict(frame[feature_columns])
        metric_row = {"run_id": "stack_linear", "split": split_name}
        metric_row.update(compute_metrics(frame["y_true"], predictions))
        run_rows.append(metric_row)

        prediction_frame = frame[["station", "timestamp", "split", "y_true"] + feature_columns].copy()
        prediction_frame["y_pred"] = predictions
        prediction_frames.append(prediction_frame)

    run_metrics = pd.DataFrame(run_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_run_metrics(run_metrics.loc[run_metrics["split"] == "test", ["run_id", "rmse", "mae", "r2"]])

    output_dir = OUTPUTS_TRAINING_DIR / "stacking" / f"stack_linear__{args.output_tag}"
    metadata = {
        "model": "stacking",
        "ensemble_type": "linear_regression",
        "members": {
            "xgboost": args.xgboost_experiment,
            "knn": args.knn_experiment,
            "prophet": args.prophet_experiment,
        },
        "selected_knn_neighbor_count": best_neighbor,
        "aligned_row_count": int(len(stacked)),
        "aligned_val_row_count": int(len(train_meta)),
        "aligned_test_row_count": int(len(test_meta)),
        "feature_columns": feature_columns,
        "coefficients": dict(zip(feature_columns, model.coef_.tolist())),
        "intercept": float(model.intercept_),
    }
    save_run_outputs(output_dir, run_metrics, predictions, summary, metadata)
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
