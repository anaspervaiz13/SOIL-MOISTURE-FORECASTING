import unittest

import numpy as np
import pandas as pd

from src.train_arima_48h import align_forecast_to_target_rows, prepare_source_series


class ArimaUtilsTests(unittest.TestCase):
    def test_prepare_source_series_uses_observed_source_not_future_target(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01 00:00:00", periods=4, freq="h"),
                "sm_0.05m": [0.1, 0.2, 0.3, 0.4],
                "target_48h": [10.0, 20.0, 30.0, 40.0],
            }
        )

        series = prepare_source_series(df)

        self.assertIsInstance(series.index, pd.DatetimeIndex)
        self.assertEqual(series.index.freqstr.lower(), "h")
        self.assertTrue(np.allclose(series.to_numpy(), [0.1, 0.2, 0.3, 0.4]))

    def test_align_forecast_to_target_rows_uses_timestamp_plus_horizon(self):
        eval_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
                "target_48h": [10.0, 11.0],
            }
        )
        forecast_index = pd.to_datetime(["2024-01-03 00:00:00", "2024-01-03 01:00:00"])
        forecast_series = pd.Series([1.5, 2.5], index=forecast_index)

        aligned = align_forecast_to_target_rows(eval_df, forecast_series, horizon_hours=48)

        self.assertTrue(np.allclose(aligned, [1.5, 2.5]))


if __name__ == "__main__":
    unittest.main()
