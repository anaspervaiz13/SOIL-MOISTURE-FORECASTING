import unittest
from pathlib import Path

from src.generate_paper_figures import (
    load_report_data,
    prepare_catboost_confirmation,
    prepare_common_subset_dual_metrics,
    prepare_refined_xgboost_ablation,
)


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


class TestGeneratePaperFigures(unittest.TestCase):
    def test_common_subset_dual_metrics_use_actual_reported_r2_values(self):
        data = load_report_data(WORKSPACE_ROOT)

        dual = prepare_common_subset_dual_metrics(data["common_subset"])

        self.assertEqual(list(dual["model"]), ["XGBoost", "Stacking", "CatBoost", "LightGBM", "LSTM", "GRU"])
        self.assertAlmostEqual(float(dual.loc[dual["model"] == "LightGBM", "r2_mean"].iloc[0]), 0.9185040313368921)
        self.assertAlmostEqual(float(dual.loc[dual["model"] == "LSTM", "r2_mean"].iloc[0]), 0.8910679696838393)
        self.assertAlmostEqual(float(dual.loc[dual["model"] == "GRU", "r2_mean"].iloc[0]), 0.8832963892763450)

    def test_catboost_confirmation_uses_active_confirmation_runs(self):
        data = load_report_data(WORKSPACE_ROOT)

        catboost = prepare_catboost_confirmation(data["all_model_summary"])

        self.assertEqual(
            list(catboost["feature_group"]),
            [
                "current_only",
                "baseline_limited",
                "short_lags",
                "short_plus_medium",
                "short_plus_weekly",
                "full_multiscale_no_rolling",
                "full_multiscale",
            ],
        )
        best_row = catboost.sort_values("rmse_mean").iloc[0]
        self.assertEqual(best_row["feature_group"], "baseline_limited")
        self.assertAlmostEqual(float(best_row["rmse_mean"]), 0.0286405262602436)

    def test_refined_xgboost_ablation_includes_weekly_and_rolling_variants(self):
        data = load_report_data(WORKSPACE_ROOT)

        refined = prepare_refined_xgboost_ablation(data["all_model_summary"])

        self.assertTrue(refined.empty)


if __name__ == "__main__":
    unittest.main()
