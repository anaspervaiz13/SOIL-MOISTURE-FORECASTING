from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge

from training.config import OUTPUTS_TRAINING
from training.evaluation import regression_metrics, save_run_outputs, summarize_metric_runs


DEFAULT_MODELS = ["xgboost", "knn", "arima", "prophet", "lstm", "transformer"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train stacking meta-learner from saved base-model predictions.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def load_predictions(model_name: str) -> pd.DataFrame:
    path = OUTPUTS_TRAINING / model_name / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions for model '{model_name}': {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    agg = (
        df.groupby(["station", "timestamp", "split", "model"], as_index=False)
        .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
    )
    return agg


def main():
    args = parse_args()
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    frames = []
    for model_name in model_names:
        model_df = load_predictions(model_name)
        model_df = model_df.rename(columns={"y_pred": f"pred_{model_name}"})
        frames.append(model_df[["station", "timestamp", "split", "y_true", f"pred_{model_name}"]])

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["station", "timestamp", "split", "y_true"], how="inner")

    val_df = merged[merged["split"] == "val"].copy()
    test_df = merged[merged["split"] == "test"].copy()
    if val_df.empty or test_df.empty:
        raise SystemExit("Stacking alignment produced no overlapping validation/test rows. Choose a different base-model set.")
    feature_cols = [col for col in merged.columns if col.startswith("pred_")]

    model = Ridge(alpha=args.alpha)
    model.fit(val_df[feature_cols], val_df["y_true"])
    test_pred = model.predict(test_df[feature_cols])

    metrics_df = pd.DataFrame(
        [
            {"run": 1, "seed": None, "split": "test", **regression_metrics(test_df["y_true"], test_pred)},
        ]
    )
    summary_df = summarize_metric_runs(metrics_df)
    predictions_df = test_df[["station", "timestamp", "split", "y_true"]].copy()
    predictions_df["run"] = 1
    predictions_df["seed"] = None
    predictions_df["model"] = "stacking"
    predictions_df["y_pred"] = test_pred

    output_dir = OUTPUTS_TRAINING / "stacking"
    metadata = {
        "model": "stacking",
        "base_models": model_names,
        "alpha": args.alpha,
        "aligned_rows": {
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "stations_covered": sorted(test_df["station"].dropna().unique().tolist()),
    }
    save_run_outputs(output_dir, metrics_df, predictions_df, summary_df, metadata)
    print(output_dir)


if __name__ == "__main__":
    main()
