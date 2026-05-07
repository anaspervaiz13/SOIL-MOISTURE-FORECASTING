import json
import unittest
from pathlib import Path

import pandas as pd

from src.export_novelty_results import (
    build_benchmark_table,
    build_common_subset_benchmark_table,
    build_markdown_summary,
    build_xgboost_ablation_table,
)

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"


def write_summary(path: Path, rmse: float, mae: float, r2: float) -> None:
    pd.DataFrame(
        [
            {"metric": "rmse", "mean": rmse, "std": 0.001, "ci95": 0.001},
            {"metric": "mae", "mean": mae, "std": 0.001, "ci95": 0.001},
            {"metric": "r2", "mean": r2, "std": 0.001, "ci95": 0.001},
        ]
    ).to_csv(path / "summary_metrics.csv", index=False)


class ArgsStub:
    xgboost_best_experiment = "full_multiscale"
    catboost_experiment = "baseline_limited__confirm"
    lightgbm_experiment = "full_multiscale__full"
    gru_experiment = "gru__repaired40"
    knn_experiment = "baseline_limited__light"
    lstm_experiment = "lstm__repaired40"
    prophet_experiment = "prophet__safe"
    arima_experiment = "arima__safe180"
    stacking_experiment = "stack_linear__xgb_lstm_linear"


class TestExportNoveltyResults(unittest.TestCase):
    def test_build_xgboost_ablation_table_adds_gain_columns(self):
        root = RUNTIME_DIR / "novelty_xgb"
        experiments = {
            "baseline_limited__strong": ("baseline_limited", 0.0300),
            "short_lags__strong": ("short_lags", 0.0298),
            "short_plus_medium__strong": ("short_plus_medium", 0.0297),
            "full_multiscale__strong": ("full_multiscale", 0.0291),
        }
        for name, (feature_group, rmse) in experiments.items():
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            write_summary(folder, rmse=rmse, mae=0.02, r2=0.85)
            (folder / "run_metadata.json").write_text(
                json.dumps({"feature_group": feature_group}),
                encoding="utf-8",
            )

        table = build_xgboost_ablation_table(root, "strong")

        self.assertEqual(table.iloc[0]["feature_group"], "full_multiscale")
        self.assertAlmostEqual(table.iloc[0]["rmse_gain_vs_baseline"], 0.0009)
        self.assertGreater(table.iloc[0]["relative_rmse_gain_pct_vs_baseline"], 0.0)

    def test_build_benchmark_table_uses_best_knn_test_row(self):
        root = RUNTIME_DIR / "novelty_benchmark"
        xgb = root / "xgboost" / "full_multiscale"
        catboost = root / "catboost" / "baseline_limited__confirm"
        lightgbm = root / "lightgbm" / "full_multiscale__full"
        knn = root / "knn" / "baseline_limited__light"
        lstm = root / "lstm" / "lstm__repaired40"
        gru = root / "gru" / "gru__repaired40"
        prophet = root / "prophet" / "prophet__safe"
        arima = root / "arima" / "arima__safe180"
        stacking = root / "stacking" / "stack_linear__xgb_lstm_linear"
        for folder in [xgb, catboost, lightgbm, knn, lstm, gru, prophet, arima, stacking]:
            folder.mkdir(parents=True, exist_ok=True)

        write_summary(xgb, rmse=0.0284, mae=0.0172, r2=0.86)
        write_summary(catboost, rmse=0.0286, mae=0.0178, r2=0.859)
        write_summary(lightgbm, rmse=0.0291, mae=0.0189, r2=0.854)
        pd.DataFrame(
            [
                {"neighbor_count": 5, "split": "val", "rmse": 0.0380, "mae": 0.0270, "r2": 0.75},
                {"neighbor_count": 13, "split": "val", "rmse": 0.0370, "mae": 0.0260, "r2": 0.77},
                {"neighbor_count": 5, "split": "test", "rmse": 0.0400, "mae": 0.0280, "r2": 0.72},
                {"neighbor_count": 13, "split": "test", "rmse": 0.0366, "mae": 0.0256, "r2": 0.77},
            ]
        ).to_csv(knn / "run_metrics.csv", index=False)
        pd.DataFrame(
            [
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 1, "split": "test", "y_true": 0.1, "y_pred": 0.1},
                {"station": "A", "timestamp": "2021-01-01 01:00:00", "seed": 2, "split": "test", "y_true": 0.2, "y_pred": 0.2},
            ]
        ).to_csv(xgb / "predictions.csv", index=False)
        pd.DataFrame(
            [
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "neighbor_count": 5, "split": "test", "y_true": 0.1, "y_pred": 0.1},
                {"station": "A", "timestamp": "2021-01-01 01:00:00", "neighbor_count": 5, "split": "test", "y_true": 0.2, "y_pred": 0.2},
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "neighbor_count": 13, "split": "test", "y_true": 0.1, "y_pred": 0.1},
            ]
        ).to_csv(knn / "predictions.csv", index=False)
        pd.DataFrame(
            [{"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 42, "split": "test", "y_true": 0.1, "y_pred": 0.1}]
        ).to_csv(lstm / "predictions.csv", index=False)
        pd.DataFrame(
            [{"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 42, "split": "test", "y_true": 0.1, "y_pred": 0.1}]
        ).to_csv(gru / "predictions.csv", index=False)
        pd.DataFrame(
            [{"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 1, "split": "test", "y_true": 0.1, "y_pred": 0.1}]
        ).to_csv(catboost / "predictions.csv", index=False)
        pd.DataFrame(
            [{"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 1, "split": "test", "y_true": 0.1, "y_pred": 0.1}]
        ).to_csv(lightgbm / "predictions.csv", index=False)
        pd.DataFrame(
            [{"station": "A", "timestamp": "2021-01-01 00:00:00", "split": "test", "y_true": 0.1, "y_pred": 0.1}]
        ).to_csv(prophet / "predictions.csv", index=False)
        pd.DataFrame(
            [{"station": "A", "timestamp": "2021-01-01 00:00:00", "split": "test", "y_true": 0.1, "y_pred": 0.1}]
        ).to_csv(arima / "predictions.csv", index=False)
        pd.DataFrame(
            [{"station": "A", "timestamp": "2021-01-01 00:00:00", "split": "test", "y_true": 0.1, "y_pred": 0.1}]
        ).to_csv(stacking / "predictions.csv", index=False)
        write_summary(lstm, rmse=0.0268, mae=0.0185, r2=0.888)
        write_summary(gru, rmse=0.0277, mae=0.0190, r2=0.881)
        write_summary(prophet, rmse=0.0688, mae=0.0557, r2=0.08)
        write_summary(arima, rmse=0.2709, mae=0.2564, r2=-12.2)
        write_summary(stacking, rmse=0.0288, mae=0.0174, r2=0.857)
        (stacking / "run_metadata.json").write_text(
            json.dumps({"ensemble_type": "linear"}),
            encoding="utf-8",
        )

        table = build_benchmark_table(root, ArgsStub())

        knn_row = table.loc[table["model"] == "KNN"].iloc[0]
        self.assertAlmostEqual(knn_row["rmse_mean"], 0.0366)
        self.assertIn("neighbor_count=13", knn_row["configuration"])
        self.assertIn("validation", knn_row["selection_basis"])
        self.assertEqual(int(knn_row["test_unique_keys"]), 1)
        self.assertEqual(int(knn_row["test_prediction_rows"]), 1)

    def test_build_markdown_summary_mentions_project_direction(self):
        ablation = pd.DataFrame(
            [
                {
                    "feature_group": "full_multiscale",
                    "rmse_mean": 0.0291,
                    "mae_mean": 0.0186,
                    "r2_mean": 0.8545,
                    "rmse_gain_vs_baseline": 0.0005,
                    "relative_rmse_gain_pct_vs_baseline": 1.7,
                },
                {
                    "feature_group": "baseline_limited",
                    "rmse_mean": 0.0296,
                    "mae_mean": 0.0183,
                    "r2_mean": 0.8499,
                    "rmse_gain_vs_baseline": 0.0,
                    "relative_rmse_gain_pct_vs_baseline": 0.0,
                },
            ]
        )
        benchmark = pd.DataFrame(
            [
                {"model": "XGBoost", "rmse_mean": 0.0284, "test_unique_keys": 9557},
                {"model": "Stacking", "rmse_mean": 0.0288, "test_unique_keys": 9557},
            ]
        )

        markdown = build_markdown_summary(ablation, benchmark)

        self.assertIn("temporal lag design", markdown)
        self.assertIn("full_multiscale", markdown)
        self.assertIn("same effective test population", markdown)

    def test_build_common_subset_benchmark_table_aligns_selected_models(self):
        root = RUNTIME_DIR / "common_subset_benchmark"
        xgb = root / "xgboost" / "full_multiscale"
        cat = root / "catboost" / "baseline_limited__confirm"
        lgbm = root / "lightgbm" / "full_multiscale__full"
        lstm = root / "lstm" / "lstm__repaired40"
        gru = root / "gru" / "gru__repaired40"
        stack = root / "stacking" / "stack_linear__xgb_lstm_linear"
        for folder in [xgb, cat, lgbm, lstm, gru, stack]:
            folder.mkdir(parents=True, exist_ok=True)

        def write_preds(folder: Path, rows: list[dict]) -> None:
            pd.DataFrame(rows).to_csv(folder / "predictions.csv", index=False)

        shared_rows = [
            {"station": "A", "timestamp": "2021-01-01 00:00:00", "split": "test", "y_true": 0.10, "y_pred": 0.11},
            {"station": "A", "timestamp": "2021-01-01 01:00:00", "split": "test", "y_true": 0.20, "y_pred": 0.19},
        ]
        xgb_rows = [
            {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 1, "split": "test", "y_true": 0.10, "y_pred": 0.11},
            {"station": "A", "timestamp": "2021-01-01 01:00:00", "seed": 1, "split": "test", "y_true": 0.20, "y_pred": 0.19},
            {"station": "A", "timestamp": "2021-01-01 02:00:00", "seed": 1, "split": "test", "y_true": 0.30, "y_pred": 0.31},
        ]
        write_preds(xgb, xgb_rows)
        write_preds(cat, shared_rows)
        write_preds(lgbm, shared_rows)
        write_preds(
            lstm,
            [
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 42, "split": "test", "y_true": 0.10, "y_pred": 0.10},
                {"station": "A", "timestamp": "2021-01-01 01:00:00", "seed": 42, "split": "test", "y_true": 0.20, "y_pred": 0.18},
            ],
        )
        write_preds(gru, shared_rows)
        write_preds(stack, shared_rows)

        table = build_common_subset_benchmark_table(root, ArgsStub())

        self.assertEqual(len(table), 6)
        self.assertTrue((table["common_test_unique_keys"] == 2).all())
        self.assertEqual(set(table["model"]), {"XGBoost", "CatBoost", "LightGBM", "LSTM", "GRU", "Stacking"})


if __name__ == "__main__":
    unittest.main()
