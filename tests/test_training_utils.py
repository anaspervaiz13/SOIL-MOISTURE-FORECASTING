import unittest

import pandas as pd

from src.training.data import chronological_split, select_feature_columns
from src.training.feature_sets import FEATURE_GROUPS, get_feature_group_columns


class TestTrainingUtils(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            [
                {
                    "station": "A",
                    "timestamp": pd.Timestamp("2020-01-01 00:00:00"),
                    "latitude": 50.0,
                    "longitude": 6.0,
                    "elevation": 100.0,
                    "sm_0.05m": 0.10,
                    "sm_0.20m": 0.20,
                    "sm_0.50m": 0.30,
                    "ta_2.00m": 10.0,
                    "ts_0.05m": 11.0,
                    "ts_0.20m": 12.0,
                    "ts_0.50m": 13.0,
                    "precipitation": 0.0,
                    "target_48h": 0.50,
                    "hour": 0,
                    "dayofweek": 2,
                    "month": 1,
                    "dayofyear": 1,
                    "hour_sin": 0.0,
                    "hour_cos": 1.0,
                    "doy_sin": 0.0,
                    "doy_cos": 1.0,
                    "sm_0.05m_lag_1h": 0.09,
                    "sm_0.05m_lag_24h": 0.08,
                    "sm_0.05m_lag_48h": 0.07,
                    "sm_0.05m_lag_72h": 0.06,
                    "sm_0.05m_lag_168h": 0.05,
                    "precipitation_lag_1h": 0.0,
                    "precipitation_lag_24h": 0.0,
                    "precipitation_lag_48h": 0.1,
                    "precipitation_lag_72h": 0.2,
                    "precipitation_lag_168h": 0.3,
                    "sm_0.05m_roll_mean_24h": 0.15,
                    "precipitation_roll_sum_24h": 1.0,
                },
                {
                    "station": "A",
                    "timestamp": pd.Timestamp("2020-01-01 01:00:00"),
                    "latitude": 50.0,
                    "longitude": 6.0,
                    "elevation": 100.0,
                    "sm_0.05m": 0.11,
                    "sm_0.20m": 0.21,
                    "sm_0.50m": 0.31,
                    "ta_2.00m": 10.1,
                    "ts_0.05m": 11.1,
                    "ts_0.20m": 12.1,
                    "ts_0.50m": 13.1,
                    "precipitation": 0.0,
                    "target_48h": 0.51,
                    "hour": 1,
                    "dayofweek": 2,
                    "month": 1,
                    "dayofyear": 1,
                    "hour_sin": 0.2,
                    "hour_cos": 0.9,
                    "doy_sin": 0.0,
                    "doy_cos": 1.0,
                    "sm_0.05m_lag_1h": 0.10,
                    "sm_0.05m_lag_24h": 0.08,
                    "sm_0.05m_lag_48h": 0.07,
                    "sm_0.05m_lag_72h": 0.06,
                    "sm_0.05m_lag_168h": 0.05,
                    "precipitation_lag_1h": 0.0,
                    "precipitation_lag_24h": 0.0,
                    "precipitation_lag_48h": 0.1,
                    "precipitation_lag_72h": 0.2,
                    "precipitation_lag_168h": 0.3,
                    "sm_0.05m_roll_mean_24h": 0.16,
                    "precipitation_roll_sum_24h": 1.1,
                },
                {
                    "station": "B",
                    "timestamp": pd.Timestamp("2020-01-01 02:00:00"),
                    "latitude": 51.0,
                    "longitude": 7.0,
                    "elevation": 110.0,
                    "sm_0.05m": 0.12,
                    "sm_0.20m": 0.22,
                    "sm_0.50m": 0.32,
                    "ta_2.00m": 10.2,
                    "ts_0.05m": 11.2,
                    "ts_0.20m": 12.2,
                    "ts_0.50m": 13.2,
                    "precipitation": 0.1,
                    "target_48h": 0.52,
                    "hour": 2,
                    "dayofweek": 2,
                    "month": 1,
                    "dayofyear": 1,
                    "hour_sin": 0.4,
                    "hour_cos": 0.8,
                    "doy_sin": 0.0,
                    "doy_cos": 1.0,
                    "sm_0.05m_lag_1h": 0.11,
                    "sm_0.05m_lag_24h": 0.09,
                    "sm_0.05m_lag_48h": 0.08,
                    "sm_0.05m_lag_72h": 0.07,
                    "sm_0.05m_lag_168h": 0.06,
                    "precipitation_lag_1h": 0.0,
                    "precipitation_lag_24h": 0.0,
                    "precipitation_lag_48h": 0.1,
                    "precipitation_lag_72h": 0.2,
                    "precipitation_lag_168h": 0.3,
                    "sm_0.05m_roll_mean_24h": 0.17,
                    "precipitation_roll_sum_24h": 1.2,
                },
                {
                    "station": "B",
                    "timestamp": pd.Timestamp("2020-01-01 03:00:00"),
                    "latitude": 51.0,
                    "longitude": 7.0,
                    "elevation": 110.0,
                    "sm_0.05m": 0.13,
                    "sm_0.20m": 0.23,
                    "sm_0.50m": 0.33,
                    "ta_2.00m": 10.3,
                    "ts_0.05m": 11.3,
                    "ts_0.20m": 12.3,
                    "ts_0.50m": 13.3,
                    "precipitation": 0.0,
                    "target_48h": 0.53,
                    "hour": 3,
                    "dayofweek": 2,
                    "month": 1,
                    "dayofyear": 1,
                    "hour_sin": 0.6,
                    "hour_cos": 0.7,
                    "doy_sin": 0.0,
                    "doy_cos": 1.0,
                    "sm_0.05m_lag_1h": 0.12,
                    "sm_0.05m_lag_24h": 0.10,
                    "sm_0.05m_lag_48h": 0.09,
                    "sm_0.05m_lag_72h": 0.08,
                    "sm_0.05m_lag_168h": 0.07,
                    "precipitation_lag_1h": 0.1,
                    "precipitation_lag_24h": 0.0,
                    "precipitation_lag_48h": 0.1,
                    "precipitation_lag_72h": 0.2,
                    "precipitation_lag_168h": 0.3,
                    "sm_0.05m_roll_mean_24h": 0.18,
                    "precipitation_roll_sum_24h": 1.3,
                },
            ]
        )

    def test_feature_groups_include_required_ablation_names(self):
        for name in [
            "current_only",
            "baseline_limited",
            "short_lags",
            "short_plus_medium",
            "short_plus_weekly",
            "full_multiscale_no_rolling",
            "full_multiscale",
        ]:
            self.assertIn(name, FEATURE_GROUPS)

    def test_get_feature_group_columns_expands_patterns(self):
        columns = get_feature_group_columns(self.frame.columns.tolist(), "short_plus_medium")

        self.assertIn("sm_0.05m_lag_1h", columns)
        self.assertIn("sm_0.05m_lag_48h", columns)
        self.assertIn("precipitation_lag_72h", columns)
        self.assertNotIn("sm_0.05m_lag_168h", columns)

    def test_weekly_ablation_group_keeps_weekly_but_not_medium_lags(self):
        columns = get_feature_group_columns(self.frame.columns.tolist(), "short_plus_weekly")

        self.assertIn("sm_0.05m_lag_168h", columns)
        self.assertNotIn("sm_0.05m_lag_48h", columns)
        self.assertNotIn("sm_0.05m_lag_72h", columns)

    def test_full_multiscale_no_rolling_excludes_rolling_features(self):
        columns = get_feature_group_columns(self.frame.columns.tolist(), "full_multiscale_no_rolling")

        self.assertIn("sm_0.05m_lag_168h", columns)
        self.assertNotIn("sm_0.05m_roll_mean_24h", columns)
        self.assertNotIn("precipitation_roll_sum_24h", columns)

    def test_select_feature_columns_excludes_target_and_static_columns(self):
        columns = select_feature_columns(self.frame, "baseline_limited", target_column="target_48h")

        self.assertNotIn("target_48h", columns)
        self.assertNotIn("station", columns)
        self.assertIn("sm_0.05m", columns)

    def test_chronological_split_uses_time_order(self):
        train_df, val_df, test_df = chronological_split(
            self.frame,
            train_fraction=0.5,
            val_fraction=0.25,
        )

        self.assertEqual(train_df["timestamp"].max(), pd.Timestamp("2020-01-01 01:00:00"))
        self.assertEqual(val_df["timestamp"].min(), pd.Timestamp("2020-01-01 02:00:00"))
        self.assertEqual(test_df["timestamp"].min(), pd.Timestamp("2020-01-01 03:00:00"))


if __name__ == "__main__":
    unittest.main()
