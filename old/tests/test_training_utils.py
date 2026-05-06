import unittest

import pandas as pd

from src.training.data import build_time_splits
from src.training.feature_sets import get_feature_group


class TrainingUtilsTests(unittest.TestCase):
    def test_build_time_splits_preserves_time_order(self):
        timestamps = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "station": ["A"] * 10,
                "timestamp": timestamps,
                "target_48h": range(10),
            }
        )

        splits = build_time_splits(df, train_ratio=0.6, val_ratio=0.2)

        self.assertLess(splits.train["timestamp"].max(), splits.val["timestamp"].min())
        self.assertLess(splits.val["timestamp"].max(), splits.test["timestamp"].min())
        self.assertEqual(len(splits.train), 6)
        self.assertEqual(len(splits.val), 2)
        self.assertEqual(len(splits.test), 2)

    def test_get_feature_group_returns_expected_core_columns(self):
        features = get_feature_group("weather_depth_temporal")

        self.assertIn("sm_0.20m", features)
        self.assertIn("sm_0.50m", features)
        self.assertIn("precipitation", features)
        self.assertIn("hour_sin", features)
        self.assertIn("sm_0.05m_lag_168h", features)


if __name__ == "__main__":
    unittest.main()
