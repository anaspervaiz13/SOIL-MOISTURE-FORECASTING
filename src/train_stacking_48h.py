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
    parser = argparse.ArgumentParser(description="Build a second-generation stacking ensemble for 48-hour forecasting.")
    parser.add_argument("--xgboost-experiment", default="full_multiscale")
    parser.add_argument("--catboost-experiment", default=None)
    parser.add_argument("--lstm-experiment", default=None)
    parser.add_argument("--meta-learner", default="xgboost", choices=["linear", "xgboost"])
    parser.add_argument("--output-tag", default="gen2")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_seed_mean_predictions(experiment_dir: Path, prediction_column: str) -> pd.DataFrame:
    predictions = pd.read_csv(experiment_dir / "predictions.csv")
    grouped = (
        predictions.groupby(["station", "timestamp", "split", "y_true"], as_index=False)["y_pred"]
        .mean()
        .rename(columns={"y_pred": prediction_column})
    )
    return grouped


def select_best_knn_neighbor(experiment_dir: Path) -> int:
    run_metrics = pd.read_csv(experiment_dir / "run_metrics.csv")
    val_rows = run_metrics.loc[run_metrics["split"] == "val"].copy()
    best_row = val_rows.sort_values(["rmse", "neighbor_count"]).iloc[0]
    return int(best_row["neighbor_count"])


def build_aligned_stack_frame(prediction_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not prediction_frames:
        return pd.DataFrame()

    merged = prediction_frames[0].copy()
    for frame in prediction_frames[1:]:
        merged = merged.merge(
            frame,
            on=["station", "timestamp", "split", "y_true"],
            how="inner",
        )
    return merged.sort_values(["split", "station", "timestamp"]).reset_index(drop=True)


def build_feature_columns(include_catboost: bool, include_lstm: bool) -> list[str]:
    columns = ["xgboost_pred"]
    if include_catboost:
        columns.append("catboost_pred")
    if include_lstm:
        columns.append("lstm_pred")
    return columns


def fit_meta_learner(meta_learner: str, train_meta: pd.DataFrame, feature_columns: list[str]):
    if meta_learner == "linear":
        model = LinearRegression()
        model.fit(train_meta[feature_columns], train_meta["y_true"])
        return model

    if meta_learner == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise SystemExit("xgboost is not installed. Install it before running stacking with the xgboost meta-learner.") from exc

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_meta[feature_columns], train_meta["y_true"])
        return model

    raise ValueError(f"Unknown meta learner: {meta_learner}")


def main() -> Path:
    args = parse_args()
    xgboost_dir = OUTPUTS_TRAINING_DIR / "xgboost" / args.xgboost_experiment

    prediction_frames = [load_seed_mean_predictions(xgboost_dir, "xgboost_pred")]
    members = {"xgboost": args.xgboost_experiment}

    if args.catboost_experiment:
        catboost_dir = OUTPUTS_TRAINING_DIR / "catboost" / args.catboost_experiment
        prediction_frames.append(load_seed_mean_predictions(catboost_dir, "catboost_pred"))
        members["catboost"] = args.catboost_experiment

    if args.lstm_experiment:
        lstm_dir = OUTPUTS_TRAINING_DIR / "lstm" / args.lstm_experiment
        prediction_frames.append(load_seed_mean_predictions(lstm_dir, "lstm_pred"))
        members["lstm"] = args.lstm_experiment

    stacked = build_aligned_stack_frame(prediction_frames)

    feature_columns = build_feature_columns(
        include_catboost=bool(args.catboost_experiment),
        include_lstm=bool(args.lstm_experiment),
    )
    train_meta = stacked.loc[stacked["split"] == "val"].copy()
    test_meta = stacked.loc[stacked["split"] == "test"].copy()

    model = fit_meta_learner(args.meta_learner, train_meta, feature_columns)

    run_rows = []
    prediction_frames = []
    for split_name, frame in [("val", train_meta), ("test", test_meta)]:
        predictions = model.predict(frame[feature_columns])
        metric_row = {"run_id": f"stack_{args.meta_learner}", "split": split_name}
        metric_row.update(compute_metrics(frame["y_true"], predictions))
        run_rows.append(metric_row)

        prediction_frame = frame[["station", "timestamp", "split", "y_true"] + feature_columns].copy()
        prediction_frame["y_pred"] = predictions
        prediction_frames.append(prediction_frame)

    run_metrics = pd.DataFrame(run_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_run_metrics(run_metrics.loc[run_metrics["split"] == "test", ["run_id", "rmse", "mae", "r2"]])

    output_dir = OUTPUTS_TRAINING_DIR / "stacking" / f"stack_{args.meta_learner}__{args.output_tag}"
    metadata = {
        "model": "stacking",
        "ensemble_type": args.meta_learner,
        "members": members,
        "aligned_row_count": int(len(stacked)),
        "aligned_val_row_count": int(len(train_meta)),
        "aligned_test_row_count": int(len(test_meta)),
        "feature_columns": feature_columns,
    }
    if args.meta_learner == "linear":
        metadata["coefficients"] = dict(zip(feature_columns, model.coef_.tolist()))
        metadata["intercept"] = float(model.intercept_)
    save_run_outputs(output_dir, run_metrics, predictions, summary, metadata)
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
