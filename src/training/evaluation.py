from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "r2": float(r2_score(y_true_array, y_pred_array)),
    }


def summarize_run_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in run_metrics.columns if column != "run_id"]
    rows = []

    for metric in metric_columns:
        values = run_metrics[metric].astype(float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci95 = float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": std,
                "ci95": ci95,
            }
        )

    return pd.DataFrame(rows)


def save_run_outputs(
    output_dir: Path,
    run_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics.to_csv(output_dir / "run_metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary_metrics.to_csv(output_dir / "summary_metrics.csv", index=False)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
