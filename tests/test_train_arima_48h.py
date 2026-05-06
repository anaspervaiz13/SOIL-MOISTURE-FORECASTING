import unittest

import numpy as np
import pandas as pd

from src.train_arima_48h import (
    align_forecast_to_target_rows,
    limit_history_series,
    parse_order,
    prepare_source_series,
)


class TestTrainArima48h(unittest.TestCase):
    def test_parse_order_splits_comma_separated_values(self):
        self.assertEqual(parse_order("3,0,1"), (3, 0, 1))

    def test_prepare_source_series_uses_observed_source_column(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
                "sm_0.05m": [0.2, 0.3],
                "target_48h": [0.8, 0.9],
            }
        )

        series = prepare_source_series(frame)

        self.assertTrue(np.array_equal(series.to_numpy(), np.array([0.2, 0.3])))

    def test_align_forecast_to_target_rows_uses_future_target_time(self):
        eval_frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
            }
        )
        forecast_series = pd.Series(
            [0.5, 0.6],
            index=pd.to_datetime(["2020-01-03 00:00:00", "2020-01-03 01:00:00"]),
        )

        aligned = align_forecast_to_target_rows(eval_frame, forecast_series, horizon_hours=48)

        self.assertTrue(np.array_equal(aligned, np.array([0.5, 0.6])))

    def test_limit_history_series_keeps_recent_window_only(self):
        series = pd.Series(
            [1.0, 2.0, 3.0, 4.0],
            index=pd.date_range("2020-01-01 00:00:00", periods=4, freq="h"),
        )

        limited = limit_history_series(series, max_history_hours=2)

        self.assertEqual(limited.index.min(), pd.Timestamp("2020-01-01 02:00:00"))
        self.assertEqual(len(limited), 2)


if __name__ == "__main__":
    unittest.main()
