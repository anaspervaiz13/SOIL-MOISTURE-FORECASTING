import json
import unittest
from pathlib import Path

import pandas as pd

from src.train_stacking_48h import (
    build_aligned_stack_frame,
    build_feature_columns,
    fit_meta_learner,
    load_seed_mean_predictions,
    select_best_knn_neighbor,
)

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"


class TestTrainStacking48h(unittest.TestCase):
    def test_select_best_knn_neighbor_uses_validation_rmse(self):
        root = RUNTIME_DIR / "stacking_knn"
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
        root = RUNTIME_DIR / "stacking_models"
        xgb_root = root / "xgboost"
        cat_root = root / "catboost"
        lstm_root = root / "lstm"
        for folder in [xgb_root, cat_root, lstm_root]:
            folder.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 42, "split": "val", "y_true": 0.1, "y_pred": 0.11},
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 52, "split": "val", "y_true": 0.1, "y_pred": 0.13},
            ]
        ).to_csv(xgb_root / "predictions.csv", index=False)

        pd.DataFrame(
            [
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 42, "split": "val", "y_true": 0.1, "y_pred": 0.12},
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 52, "split": "val", "y_true": 0.1, "y_pred": 0.14},
            ]
        ).to_csv(cat_root / "predictions.csv", index=False)
        pd.DataFrame(
            [
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 42, "split": "val", "y_true": 0.1, "y_pred": 0.09},
                {"station": "A", "timestamp": "2021-01-01 00:00:00", "seed": 52, "split": "val", "y_true": 0.1, "y_pred": 0.11},
            ]
        ).to_csv(lstm_root / "predictions.csv", index=False)

        xgb_predictions = load_seed_mean_predictions(xgb_root, "xgboost_pred")
        cat_predictions = load_seed_mean_predictions(cat_root, "catboost_pred")
        lstm_predictions = load_seed_mean_predictions(lstm_root, "lstm_pred")
        aligned = build_aligned_stack_frame([xgb_predictions, cat_predictions, lstm_predictions])

        self.assertAlmostEqual(aligned.iloc[0]["xgboost_pred"], 0.12)
        self.assertAlmostEqual(aligned.iloc[0]["catboost_pred"], 0.13)
        self.assertAlmostEqual(aligned.iloc[0]["lstm_pred"], 0.10)

    def test_fit_meta_learner_supports_linear_and_xgboost(self):
        frame = pd.DataFrame(
            [
                {"xgboost_pred": 0.10, "catboost_pred": 0.11, "lstm_pred": 0.09, "y_true": 0.10},
                {"xgboost_pred": 0.20, "catboost_pred": 0.19, "lstm_pred": 0.22, "y_true": 0.21},
                {"xgboost_pred": 0.30, "catboost_pred": 0.29, "lstm_pred": 0.31, "y_true": 0.30},
            ]
        )
        feature_columns = ["xgboost_pred", "catboost_pred", "lstm_pred"]

        linear_model = fit_meta_learner("linear", frame, feature_columns)
        self.assertTrue(hasattr(linear_model, "predict"))

        xgb_model = fit_meta_learner("xgboost", frame, feature_columns)
        self.assertTrue(hasattr(xgb_model, "predict"))

    def test_build_feature_columns_matches_selected_members(self):
        self.assertEqual(
            build_feature_columns(include_catboost=False, include_lstm=True),
            ["xgboost_pred", "lstm_pred"],
        )
        self.assertEqual(
            build_feature_columns(include_catboost=True, include_lstm=True),
            ["xgboost_pred", "catboost_pred", "lstm_pred"],
        )


if __name__ == "__main__":
    unittest.main()
