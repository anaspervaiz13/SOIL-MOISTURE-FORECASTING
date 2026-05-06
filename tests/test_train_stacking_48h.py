import json
import unittest
from pathlib import Path

import pandas as pd

from src.train_stacking_48h import (
    build_aligned_stack_frame,
    load_best_knn_predictions,
    load_prophet_predictions,
    load_xgboost_mean_predictions,
    select_best_knn_neighbor,
)


class TestTrainStacking48h(unittest.TestCase):
    def test_select_best_knn_neighbor_uses_validation_rmse(self):
        root = Path("C:/Users/HP/Desktop/UNI/final work/tests/runtime/stacking_knn")
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"neighbor_count": 5, "split": "val", "rmse": 0.041},
                {"neighbor_count": 13, "split": "val", "rmse": 0.037},
                {"neighbor_count": 9, "split": "val", "rmse": 0.039},
            ]
        ).to_csv(root / "run_metrics.csv", index=False)

        self.assertEqual(select_best_knn_neighbor(root), 13)

    def test_prediction_loaders_align_expected_columns(self):
        root = Path("C:/Users/HP/Desktop/UNI/final work/tests/runtime/stacking_models")
        xgb_root = root / "xgboost"
        knn_root = root / "knn"
        prophet_root = root / "prophet"
        for folder in [xgb_root, knn_root, prophet_root]:
            folder.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 42, "split": "val", "y_true": 0.1, "y_pred": 0.11},
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 52, "split": "val", "y_true": 0.1, "y_pred": 0.13},
            ]
        ).to_csv(xgb_root / "predictions.csv", index=False)

        pd.DataFrame(
            [
                {"neighbor_count": 5, "split": "val", "rmse": 0.03},
                {"neighbor_count": 9, "split": "val", "rmse": 0.04},
            ]
        ).to_csv(knn_root / "run_metrics.csv", index=False)
        pd.DataFrame(
            [
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "neighbor_count": 5, "split": "val", "y_true": 0.1, "y_pred": 0.12},
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "neighbor_count": 9, "split": "val", "y_true": 0.1, "y_pred": 0.20},
            ]
        ).to_csv(knn_root / "predictions.csv", index=False)

        pd.DataFrame(
            [
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "split": "val", "y_true": 0.1, "y_pred": 0.09},
            ]
        ).to_csv(prophet_root / "predictions.csv", index=False)
        (prophet_root / "run_metadata.json").write_text(json.dumps({"model": "prophet"}), encoding="utf-8")

        xgb_predictions = load_xgboost_mean_predictions(xgb_root)
        knn_predictions, best_neighbor = load_best_knn_predictions(knn_root)
        prophet_predictions = load_prophet_predictions(prophet_root)
        aligned = build_aligned_stack_frame(xgb_predictions, knn_predictions, prophet_predictions)

        self.assertEqual(best_neighbor, 5)
        self.assertAlmostEqual(aligned.iloc[0]["xgboost_pred"], 0.12)
        self.assertAlmostEqual(aligned.iloc[0]["knn_pred"], 0.12)
        self.assertAlmostEqual(aligned.iloc[0]["prophet_pred"], 0.09)


if __name__ == "__main__":
    unittest.main()
