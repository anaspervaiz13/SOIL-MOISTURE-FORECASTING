import json
from pathlib import Path

import unittest

import pandas as pd

from src.training.evaluation import compute_metrics, save_run_outputs, summarize_run_metrics

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"


class TestEvaluationUtils(unittest.TestCase):
    def test_compute_metrics_returns_expected_keys(self):
        metrics = compute_metrics(
            y_true=[1.0, 2.0, 3.0],
            y_pred=[1.0, 2.5, 2.5],
        )

        self.assertEqual(set(metrics.keys()), {"rmse", "mae", "r2"})
        self.assertGreater(metrics["rmse"], 0.0)
        self.assertGreater(metrics["mae"], 0.0)

    def test_summarize_run_metrics_builds_mean_std_and_ci(self):
        run_metrics = pd.DataFrame(
            [
                {"run_id": 1, "rmse": 0.30, "mae": 0.20, "r2": 0.80},
                {"run_id": 2, "rmse": 0.32, "mae": 0.21, "r2": 0.78},
                {"run_id": 3, "rmse": 0.31, "mae": 0.19, "r2": 0.81},
            ]
        )

        summary = summarize_run_metrics(run_metrics)

        self.assertIn("metric", summary.columns)
        self.assertIn("mean", summary.columns)
        self.assertIn("std", summary.columns)
        self.assertIn("ci95", summary.columns)
        rmse_row = summary.loc[summary["metric"] == "rmse"].iloc[0]
        self.assertAlmostEqual(rmse_row["mean"], 0.31)
        self.assertGreater(rmse_row["ci95"], 0.0)

    def test_save_run_outputs_writes_expected_artifacts(self):
        output_dir = RUNTIME_DIR / "eval_outputs"
        if output_dir.exists():
            for child in output_dir.iterdir():
                child.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        run_metrics = pd.DataFrame([{"run_id": 1, "split": "test", "rmse": 0.3, "mae": 0.2, "r2": 0.8}])
        predictions = pd.DataFrame([{"station": "A", "timestamp": "2020-01-01 00:00:00", "y_true": 1.0, "y_pred": 1.1}])
        summary = pd.DataFrame([{"metric": "rmse", "mean": 0.3, "std": 0.0, "ci95": 0.0}])
        metadata = {"model": "xgboost", "feature_group": "short_lags"}

        save_run_outputs(output_dir, run_metrics, predictions, summary, metadata)

        self.assertTrue((output_dir / "run_metrics.csv").exists())
        self.assertTrue((output_dir / "predictions.csv").exists())
        self.assertTrue((output_dir / "summary_metrics.csv").exists())
        self.assertTrue((output_dir / "run_metadata.json").exists())
        saved_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_metadata["model"], "xgboost")


if __name__ == "__main__":
    unittest.main()
