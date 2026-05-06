import unittest
import json
from pathlib import Path

import pandas as pd

from src.summarize_training_results import summarize_all_models, summarize_xgboost_feature_groups
from src.train_xgboost_48h import resolve_output_dir_name


class TestSummarizeTrainingResults(unittest.TestCase):
    def test_resolve_output_dir_name_appends_output_tag(self):
        self.assertEqual(resolve_output_dir_name("full_multiscale", None), "full_multiscale")
        self.assertEqual(resolve_output_dir_name("full_multiscale", "strong"), "full_multiscale__strong")

    def test_summarize_xgboost_feature_groups_collects_rows(self):
        root = Path("C:/Users/HP/Desktop/UNI/final work/tests/runtime/summaries")
        group_a = root / "baseline_limited"
        group_b = root / "full_multiscale"
        for folder in [group_a, group_b]:
            folder.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"metric": "rmse", "mean": 0.30, "std": 0.01, "ci95": 0.01},
                {"metric": "mae", "mean": 0.20, "std": 0.01, "ci95": 0.01},
                {"metric": "r2", "mean": 0.80, "std": 0.01, "ci95": 0.01},
            ]
        ).to_csv(group_a / "summary_metrics.csv", index=False)
        pd.DataFrame(
            [
                {"metric": "rmse", "mean": 0.28, "std": 0.01, "ci95": 0.01},
                {"metric": "mae", "mean": 0.18, "std": 0.01, "ci95": 0.01},
                {"metric": "r2", "mean": 0.85, "std": 0.01, "ci95": 0.01},
            ]
        ).to_csv(group_b / "summary_metrics.csv", index=False)

        summary = summarize_xgboost_feature_groups(root)

        self.assertEqual(summary.iloc[0]["feature_group"], "full_multiscale")
        self.assertAlmostEqual(summary.iloc[0]["rmse_mean"], 0.28)
        self.assertEqual(len(summary), 2)

    def test_summarize_all_models_collects_cross_model_rows(self):
        root = Path("C:/Users/HP/Desktop/UNI/final work/tests/runtime/all_model_summaries")
        xgb_run = root / "xgboost" / "full_multiscale__strong"
        prophet_run = root / "prophet" / "prophet__safe"
        for folder in [xgb_run, prophet_run]:
            folder.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"metric": "rmse", "mean": 0.029, "std": 0.001, "ci95": 0.001},
                {"metric": "mae", "mean": 0.018, "std": 0.001, "ci95": 0.001},
                {"metric": "r2", "mean": 0.85, "std": 0.01, "ci95": 0.01},
            ]
        ).to_csv(xgb_run / "summary_metrics.csv", index=False)
        (xgb_run / "run_metadata.json").write_text(
            json.dumps(
                {
                    "model": "xgboost",
                    "feature_group": "full_multiscale",
                    "output_tag": "strong",
                    "target_column": "target_48h",
                    "target_source_column": "sm_0.05m",
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 42, "split": "test", "y_true": 0.1, "y_pred": 0.1},
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 52, "split": "test", "y_true": 0.1, "y_pred": 0.1},
            ]
        ).to_csv(xgb_run / "predictions.csv", index=False)

        pd.DataFrame(
            [
                {"metric": "rmse", "mean": 0.069, "std": 0.020, "ci95": 0.023},
                {"metric": "mae", "mean": 0.056, "std": 0.018, "ci95": 0.020},
                {"metric": "r2", "mean": 0.08, "std": 0.30, "ci95": 0.34},
            ]
        ).to_csv(prophet_run / "summary_metrics.csv", index=False)
        (prophet_run / "run_metadata.json").write_text(
            json.dumps(
                {
                    "model": "prophet",
                    "output_tag": "safe",
                    "target_column": "target_48h",
                    "target_source_column": "sm_0.05m",
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"station": "A", "timestamp": "2021-01-01 01:00:00", "split": "test", "y_true": 0.2, "y_pred": 0.2},
            ]
        ).to_csv(prophet_run / "predictions.csv", index=False)

        summary = summarize_all_models(root)

        self.assertEqual(len(summary), 2)
        xgb_row = summary.loc[summary["model"] == "xgboost"].iloc[0]
        prophet_row = summary.loc[summary["model"] == "prophet"].iloc[0]
        self.assertEqual(xgb_row["feature_group"], "full_multiscale")
        self.assertEqual(int(xgb_row["test_prediction_rows"]), 2)
        self.assertEqual(int(xgb_row["test_unique_keys"]), 1)
        self.assertEqual(int(xgb_row["test_variant_count"]), 2)
        self.assertEqual(xgb_row["coverage_basis"], "all_saved_test_predictions_across_seed")
        self.assertAlmostEqual(prophet_row["rmse_mean"], 0.069)
        self.assertEqual(int(prophet_row["test_variant_count"]), 1)


if __name__ == "__main__":
    unittest.main()
