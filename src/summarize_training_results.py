from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.training.config import OUTPUTS_TRAINING_DIR


def summarize_xgboost_feature_groups(root: Path) -> pd.DataFrame:
    rows = []
    for group_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summary_file = group_dir / "summary_metrics.csv"
        if not summary_file.exists():
            continue

        summary = pd.read_csv(summary_file)
        row = {"feature_group": group_dir.name}
        for _, metric_row in summary.iterrows():
            metric_name = metric_row["metric"]
            row[f"{metric_name}_mean"] = metric_row["mean"]
            row[f"{metric_name}_std"] = metric_row["std"]
            row[f"{metric_name}_ci95"] = metric_row["ci95"]
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["feature_group", "rmse_mean", "mae_mean", "r2_mean"])

    return pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)


def _load_run_summary(experiment_dir: Path) -> dict | None:
    summary_file = experiment_dir / "summary_metrics.csv"
    metadata_file = experiment_dir / "run_metadata.json"
    if not summary_file.exists():
        return None

    row = {
        "experiment": experiment_dir.name,
        "model": experiment_dir.parent.name,
    }

    summary = pd.read_csv(summary_file)
    for _, metric_row in summary.iterrows():
        metric_name = metric_row["metric"]
        row[f"{metric_name}_mean"] = metric_row["mean"]
        row[f"{metric_name}_std"] = metric_row["std"]
        row[f"{metric_name}_ci95"] = metric_row["ci95"]

    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        for key in ["feature_group", "output_tag", "model", "target_column", "target_source_column"]:
            if key in metadata:
                row[key] = metadata[key]

    predictions_file = experiment_dir / "predictions.csv"
    if predictions_file.exists():
        predictions = pd.read_csv(predictions_file)
        if "split" in predictions.columns:
            predictions = predictions.loc[predictions["split"] == "test"].copy()
        row["test_prediction_rows"] = int(len(predictions))
        if {"station", "timestamp"}.issubset(predictions.columns):
            row["test_unique_keys"] = int(len(predictions[["station", "timestamp"]].drop_duplicates()))
        if "station" in predictions.columns:
            stations = sorted(predictions["station"].dropna().astype(str).unique().tolist())
            row["test_station_count"] = int(len(stations))
            row["test_stations"] = ",".join(stations)
        variant_columns = [column for column in ["seed", "neighbor_count"] if column in predictions.columns]
        if variant_columns:
            row["test_variant_count"] = int(len(predictions[variant_columns].drop_duplicates()))
            row["coverage_basis"] = f"all_saved_test_predictions_across_{'_and_'.join(variant_columns)}"
        else:
            row["test_variant_count"] = 1
            row["coverage_basis"] = "single_saved_test_prediction_set"

    return row


def summarize_all_models(root: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for experiment_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            row = _load_run_summary(experiment_dir)
            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "experiment",
                "feature_group",
                "output_tag",
                "rmse_mean",
                "mae_mean",
                "r2_mean",
            ]
        )

    ordered_columns = [
        "model",
        "experiment",
        "feature_group",
        "output_tag",
        "target_column",
        "target_source_column",
        "test_prediction_rows",
        "test_unique_keys",
        "test_station_count",
        "test_stations",
        "test_variant_count",
        "coverage_basis",
        "rmse_mean",
        "rmse_std",
        "rmse_ci95",
        "mae_mean",
        "mae_std",
        "mae_ci95",
        "r2_mean",
        "r2_std",
        "r2_ci95",
    ]
    summary = pd.DataFrame(rows)
    existing_ordered_columns = [column for column in ordered_columns if column in summary.columns]
    remaining_columns = [column for column in summary.columns if column not in existing_ordered_columns]
    summary = summary[existing_ordered_columns + remaining_columns]
    return summary.sort_values(["model", "experiment"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize training outputs across saved experiment folders.")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    if args.all_models:
        summary = summarize_all_models(OUTPUTS_TRAINING_DIR)
        output_path = Path(args.output) if args.output else OUTPUTS_TRAINING_DIR / "all_model_comparison_summary.csv"
    else:
        model_root = OUTPUTS_TRAINING_DIR / args.model
        summary = summarize_xgboost_feature_groups(model_root)
        output_path = Path(args.output) if args.output else model_root / "model_comparison_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()
