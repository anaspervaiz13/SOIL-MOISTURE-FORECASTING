from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def aggregate_run_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df.groupby(["run", "seed", "split"], as_index=False, dropna=False)[["RMSE", "MAE", "R2"]].mean()


def summarize_metric_runs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, split_df in metrics_df.groupby("split"):
        summary = {"split": split, "runs": int(len(split_df))}
        for metric in ["RMSE", "MAE", "R2"]:
            values = split_df[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci95 = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_ci95"] = ci95
        rows.append(summary)
    return pd.DataFrame(rows)


def save_run_outputs(output_dir: Path, metrics_df: pd.DataFrame, predictions_df: pd.DataFrame, summary_df: pd.DataFrame, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "run_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
