import unittest

import numpy as np
import pandas as pd

from src.train_prophet_48h import align_forecast_to_target_rows, prepare_prophet_training_frame, resolve_output_dir_name


class TestTrainProphet48h(unittest.TestCase):
    def test_resolve_output_dir_name_appends_tag_when_present(self):
        self.assertEqual(resolve_output_dir_name("prophet", None), "prophet")
        self.assertEqual(resolve_output_dir_name("prophet", "safe"), "prophet__safe")

    def test_prepare_prophet_training_frame_uses_observed_source_column(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
                "sm_0.05m": [0.2, 0.3],
                "target_48h": [0.8, 0.9],
            }
        )

        prepared = prepare_prophet_training_frame(frame)

        self.assertEqual(prepared["y"].tolist(), [0.2, 0.3])
        self.assertEqual(prepared.columns.tolist(), ["ds", "y"])

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


if __name__ == "__main__":
    unittest.main()
