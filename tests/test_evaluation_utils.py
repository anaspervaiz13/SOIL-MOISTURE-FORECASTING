import unittest

import pandas as pd

from src.training.evaluation import aggregate_run_metrics


class EvaluationUtilsTests(unittest.TestCase):
    def test_aggregate_run_metrics_keeps_deterministic_rows_with_null_seed(self):
        df = pd.DataFrame(
            [
                {"run": 1, "seed": None, "split": "val", "station": "A", "RMSE": 1.0, "MAE": 0.5, "R2": 0.8},
                {"run": 1, "seed": None, "split": "val", "station": "B", "RMSE": 3.0, "MAE": 1.5, "R2": 0.4},
            ]
        )

        out = aggregate_run_metrics(df)

        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["split"], "val")
        self.assertAlmostEqual(out.iloc[0]["RMSE"], 2.0)
        self.assertAlmostEqual(out.iloc[0]["MAE"], 1.0)
        self.assertAlmostEqual(out.iloc[0]["R2"], 0.6)


if __name__ == "__main__":
    unittest.main()
