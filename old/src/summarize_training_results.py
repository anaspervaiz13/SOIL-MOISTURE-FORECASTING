from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.config import OUTPUTS_TRAINING


def get_prediction_coverage(model_dir: Path) -> dict[tuple[str, str], tuple[int, int]]:
    predictions_path = model_dir / "predictions.csv"
    if not predictions_path.exists() or predictions_path.stat().st_size == 0:
        return {}

    df = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    if df.empty:
        return {}

    grouped = (
        df.groupby("split", dropna=False)
        .agg(
            evaluation_rows=("timestamp", lambda s: int(pd.Series(list(zip(df.loc[s.index, "station"], s))).drop_duplicates().shape[0])),
            stations_covered=("station", lambda s: int(pd.Series(s).dropna().nunique())),
        )
        .reset_index()
    )
    return {
        (row["split"], model_dir.name): (int(row["evaluation_rows"]), int(row["stations_covered"]))
        for _, row in grouped.iterrows()
    }


def main():
    rows = []
    coverage_by_model_split = {}
    for model_dir in [p for p in OUTPUTS_TRAINING.iterdir() if p.is_dir()]:
        coverage_by_model_split.update(get_prediction_coverage(model_dir))

    for summary_path in OUTPUTS_TRAINING.glob("*/summary_metrics.csv"):
        model_name = summary_path.parent.name
        if summary_path.stat().st_size == 0:
            print(f"Skipping empty summary file: {summary_path}")
            continue

        try:
            summary_df = pd.read_csv(summary_path)
        except pd.errors.EmptyDataError:
            print(f"Skipping unreadable summary file: {summary_path}")
            continue

        if summary_df.empty:
            print(f"Skipping empty summary table: {summary_path}")
            continue

        for _, row in summary_df.iterrows():
            coverage = coverage_by_model_split.get((row["split"], model_name), (None, None))
            rows.append(
                {
                    "model": model_name,
                    "split": row["split"],
                    "evaluation_rows": coverage[0],
                    "stations_covered": coverage[1],
                    "RMSE_mean": row["RMSE_mean"],
                    "RMSE_std": row["RMSE_std"],
                    "RMSE_ci95": row["RMSE_ci95"],
                    "MAE_mean": row["MAE_mean"],
                    "MAE_std": row["MAE_std"],
                    "MAE_ci95": row["MAE_ci95"],
                    "R2_mean": row["R2_mean"],
                    "R2_std": row["R2_std"],
                    "R2_ci95": row["R2_ci95"],
                }
            )

    if not rows:
        raise SystemExit("No summary_metrics.csv files found under outputs/training.")

    out_df = pd.DataFrame(rows).sort_values(["split", "RMSE_mean"])
    output_path = OUTPUTS_TRAINING / "model_comparison_summary.csv"
    out_df.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
