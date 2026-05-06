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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export novelty-focused result tables and summary text.")
    parser.add_argument("--xgboost-ablation-tag", default="strong")
    parser.add_argument("--xgboost-best-experiment", default="full_multiscale")
    parser.add_argument("--knn-experiment", default="baseline_limited__light")
    parser.add_argument("--lstm-experiment", default="lstm__cpu40")
    parser.add_argument("--prophet-experiment", default="prophet__safe")
    parser.add_argument("--arima-experiment", default="arima__safe180")
    parser.add_argument("--stacking-experiment", default="stack_linear__aligned")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _read_summary_metrics(summary_path: Path) -> dict[str, float]:
    summary = pd.read_csv(summary_path)
    metrics: dict[str, float] = {}
    for _, row in summary.iterrows():
        metric = row["metric"]
        metrics[f"{metric}_mean"] = float(row["mean"])
        metrics[f"{metric}_std"] = float(row["std"])
        metrics[f"{metric}_ci95"] = float(row["ci95"])
    return metrics


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_coverage(predictions_path: Path, filters: dict[str, object] | None = None) -> dict[str, object]:
    predictions = pd.read_csv(predictions_path)
    if "split" in predictions.columns:
        predictions = predictions.loc[predictions["split"] == "test"].copy()
    if filters:
        for column, value in filters.items():
            predictions = predictions.loc[predictions[column] == value].copy()

    unique_keys = 0
    if {"station", "timestamp"}.issubset(predictions.columns):
        unique_keys = int(len(predictions[["station", "timestamp"]].drop_duplicates()))

    stations = sorted(predictions["station"].dropna().astype(str).unique().tolist()) if "station" in predictions.columns else []
    return {
        "test_prediction_rows": int(len(predictions)),
        "test_unique_keys": unique_keys,
        "test_station_count": int(len(stations)),
        "test_stations": ",".join(stations),
    }


def build_xgboost_ablation_table(root: Path, output_tag: str) -> pd.DataFrame:
    rows = []
    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.endswith(f"__{output_tag}")):
        metadata = _read_json(experiment_dir / "run_metadata.json")
        metrics = _read_summary_metrics(experiment_dir / "summary_metrics.csv")
        rows.append(
            {
                "feature_group": metadata["feature_group"],
                "experiment": experiment_dir.name,
                **metrics,
            }
        )

    summary = pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)
    best_rmse = float(summary["rmse_mean"].min())
    baseline_rmse = float(summary.loc[summary["feature_group"] == "baseline_limited", "rmse_mean"].iloc[0])
    summary["rmse_gain_vs_baseline"] = baseline_rmse - summary["rmse_mean"]
    summary["relative_rmse_gain_pct_vs_baseline"] = (summary["rmse_gain_vs_baseline"] / baseline_rmse) * 100.0
    summary["rmse_gap_from_best"] = summary["rmse_mean"] - best_rmse
    return summary


def build_benchmark_table(outputs_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    xgboost_dir = outputs_root / "xgboost" / args.xgboost_best_experiment
    knn_dir = outputs_root / "knn" / args.knn_experiment
    lstm_dir = outputs_root / "lstm" / args.lstm_experiment
    prophet_dir = outputs_root / "prophet" / args.prophet_experiment
    arima_dir = outputs_root / "arima" / args.arima_experiment
    stacking_dir = outputs_root / "stacking" / args.stacking_experiment

    xgboost_metrics = _read_summary_metrics(xgboost_dir / "summary_metrics.csv")
    knn_run_metrics = pd.read_csv(knn_dir / "run_metrics.csv")
    best_knn_neighbor = int(
        knn_run_metrics.loc[knn_run_metrics["split"] == "val"].sort_values(["rmse", "neighbor_count"]).iloc[0]["neighbor_count"]
    )
    best_knn_row = knn_run_metrics.loc[
        (knn_run_metrics["split"] == "test") & (knn_run_metrics["neighbor_count"] == best_knn_neighbor)
    ].iloc[0]
    lstm_metrics = _read_summary_metrics(lstm_dir / "summary_metrics.csv")
    prophet_metrics = _read_summary_metrics(prophet_dir / "summary_metrics.csv")
    arima_metrics = _read_summary_metrics(arima_dir / "summary_metrics.csv")
    stacking_metrics = _read_summary_metrics(stacking_dir / "summary_metrics.csv")
    stacking_metadata = _read_json(stacking_dir / "run_metadata.json")
    xgboost_coverage = _prediction_coverage(xgboost_dir / "predictions.csv")
    knn_coverage = _prediction_coverage(knn_dir / "predictions.csv", filters={"neighbor_count": best_knn_neighbor})
    lstm_coverage = _prediction_coverage(lstm_dir / "predictions.csv")
    prophet_coverage = _prediction_coverage(prophet_dir / "predictions.csv")
    arima_coverage = _prediction_coverage(arima_dir / "predictions.csv")
    stacking_coverage = _prediction_coverage(stacking_dir / "predictions.csv")

    rows = [
        {
            "model": "XGBoost",
            "configuration": "full_multiscale repeated-seed run",
            "experiment": args.xgboost_best_experiment,
            "selection_basis": "preselected repeated-seed experiment",
            **xgboost_metrics,
            **xgboost_coverage,
        },
        {
            "model": "KNN",
            "configuration": f"baseline_limited, neighbor_count={int(best_knn_row['neighbor_count'])}",
            "experiment": args.knn_experiment,
            "selection_basis": "neighbor_count chosen on validation RMSE; test row reported",
            "rmse_mean": float(best_knn_row["rmse"]),
            "rmse_std": 0.0,
            "rmse_ci95": 0.0,
            "mae_mean": float(best_knn_row["mae"]),
            "mae_std": 0.0,
            "mae_ci95": 0.0,
            "r2_mean": float(best_knn_row["r2"]),
            "r2_std": 0.0,
            "r2_ci95": 0.0,
            **knn_coverage,
        },
        {
            "model": "LSTM",
            "configuration": "lookback=168, cpu40 repeated-seed run",
            "experiment": args.lstm_experiment,
            "selection_basis": "preselected repeated-seed experiment",
            **lstm_metrics,
            **lstm_coverage,
        },
        {
            "model": "Prophet",
            "configuration": "safe classical baseline",
            "experiment": args.prophet_experiment,
            "selection_basis": "single configured run",
            **prophet_metrics,
            **prophet_coverage,
        },
        {
            "model": "ARIMA",
            "configuration": "safe180 classical baseline",
            "experiment": args.arima_experiment,
            "selection_basis": "single configured run",
            **arima_metrics,
            **arima_coverage,
        },
        {
            "model": "Stacking",
            "configuration": f"aligned linear stack, knn={stacking_metadata['selected_knn_neighbor_count']}",
            "experiment": args.stacking_experiment,
            "selection_basis": "validation-fit aligned ensemble",
            **stacking_metrics,
            **stacking_coverage,
        },
    ]

    summary = pd.DataFrame(rows).sort_values(["model", "experiment"]).reset_index(drop=True)
    return summary


def build_markdown_summary(ablation: pd.DataFrame, benchmark: pd.DataFrame) -> str:
    best_ablation = ablation.iloc[0]
    baseline_ablation = ablation.loc[ablation["feature_group"] == "baseline_limited"].iloc[0]
    xgboost_row = benchmark.loc[benchmark["model"] == "XGBoost"].iloc[0]
    stacking_row = benchmark.loc[benchmark["model"] == "Stacking"].iloc[0]

    lines = [
        "# Novelty Results Summary",
        "",
        "## Main Result",
        "",
        f"- Best lag-design result: `{best_ablation['feature_group']}` with `RMSE {best_ablation['rmse_mean']:.6f}`, `MAE {best_ablation['mae_mean']:.6f}`, `R2 {best_ablation['r2_mean']:.6f}`.",
        f"- Baseline lag result: `{baseline_ablation['feature_group']}` with `RMSE {baseline_ablation['rmse_mean']:.6f}`.",
        f"- Absolute RMSE gain of best lag design over baseline: `{best_ablation['rmse_gain_vs_baseline']:.6f}`.",
        f"- Relative RMSE gain of best lag design over baseline: `{best_ablation['relative_rmse_gain_pct_vs_baseline']:.3f}%`.",
        "",
        "## Benchmark Context",
        "",
        f"- XGBoost reference result: `RMSE {xgboost_row['rmse_mean']:.6f}` on `{int(xgboost_row['test_unique_keys'])}` unique test keys.",
        f"- Stacking reference result: `RMSE {stacking_row['rmse_mean']:.6f}` on `{int(stacking_row['test_unique_keys'])}` aligned unique test keys.",
        "- Benchmark rows are descriptive and include coverage columns because the current models do not all share the same effective test population.",
        "",
        "## Interpretation",
        "",
        "- The strongest evidence for novelty is the XGBoost lag-design ablation, not the stacking result.",
        "- Multi-scale lag design shows a small but measurable benefit for the strongest tree-based model over simpler lag-group alternatives.",
        "- The aligned stacking experiment is supplementary and mostly tracks XGBoost rather than adding a strong new ensemble effect.",
        "- Cross-model rankings should be stated carefully until a common evaluation subset is exported for every model.",
        "- This supports the project direction: the main contribution is temporal lag design for medium-horizon soil moisture forecasting, not stacking.",
        "",
    ]
    return "\n".join(lines)


def main() -> Path:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_TRAINING_DIR / "novelty"
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation = build_xgboost_ablation_table(OUTPUTS_TRAINING_DIR / "xgboost", args.xgboost_ablation_tag)
    benchmark = build_benchmark_table(OUTPUTS_TRAINING_DIR, args)
    markdown = build_markdown_summary(ablation, benchmark)

    ablation.to_csv(output_dir / "xgboost_ablation_table.csv", index=False)
    benchmark.to_csv(output_dir / "benchmark_table.csv", index=False)
    (output_dir / "novelty_results_summary.md").write_text(markdown, encoding="utf-8")

    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
