import unittest

import numpy as np
import pandas as pd

from src.training.sequence import (
    apply_feature_scaler,
    apply_target_scaler,
    fit_feature_scaler,
    fit_target_scaler,
    inverse_target_scaler,
    make_sequence_arrays,
    split_sequence_meta,
)


class TestSequenceUtils(unittest.TestCase):
    def test_make_sequence_arrays_builds_windows_without_crossing_station_boundaries(self):
        frame = pd.DataFrame(
            [
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 00:00:00"), "f1": 1.0, "target_48h": 10.0},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 01:00:00"), "f1": 2.0, "target_48h": 11.0},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 02:00:00"), "f1": 3.0, "target_48h": 12.0},
                {"station": "B", "timestamp": pd.Timestamp("2020-01-01 00:00:00"), "f1": 4.0, "target_48h": 13.0},
                {"station": "B", "timestamp": pd.Timestamp("2020-01-01 01:00:00"), "f1": 5.0, "target_48h": 14.0},
                {"station": "B", "timestamp": pd.Timestamp("2020-01-01 02:00:00"), "f1": 6.0, "target_48h": 15.0},
            ]
        )

        x_values, y_values, meta = make_sequence_arrays(frame, ["f1"], "target_48h", lookback=2)

        self.assertEqual(x_values.shape, (4, 2, 1))
        self.assertTrue(np.array_equal(y_values, np.array([11.0, 12.0, 14.0, 15.0])))
        self.assertEqual(meta["station"].tolist(), ["A", "A", "B", "B"])

    def test_make_sequence_arrays_skips_windows_that_cross_timestamp_gaps(self):
        frame = pd.DataFrame(
            [
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 00:00:00"), "f1": 1.0, "target_48h": 10.0},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 01:00:00"), "f1": 2.0, "target_48h": 11.0},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 03:00:00"), "f1": 3.0, "target_48h": 12.0},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 04:00:00"), "f1": 4.0, "target_48h": 13.0},
            ]
        )

        x_values, y_values, meta = make_sequence_arrays(frame, ["f1"], "target_48h", lookback=2)

        self.assertEqual(x_values.shape, (2, 2, 1))
        self.assertTrue(np.array_equal(y_values, np.array([11.0, 13.0])))
        self.assertEqual(
            meta["timestamp"].tolist(),
            [pd.Timestamp("2020-01-01 01:00:00"), pd.Timestamp("2020-01-01 04:00:00")],
        )

    def test_split_sequence_meta_uses_timestamp_boundaries(self):
        meta = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00"]
                )
            }
        )

        train_mask, val_mask, test_mask = split_sequence_meta(
            meta,
            train_end=pd.Timestamp("2020-01-01 00:00:00"),
            val_end=pd.Timestamp("2020-01-01 01:00:00"),
        )

        self.assertTrue(np.array_equal(train_mask, np.array([True, False, False])))
        self.assertTrue(np.array_equal(val_mask, np.array([False, True, False])))
        self.assertTrue(np.array_equal(test_mask, np.array([False, False, True])))

    def test_feature_scaler_uses_train_statistics(self):
        train_x = np.array(
            [
                [[1.0, 10.0], [3.0, 14.0]],
                [[5.0, 18.0], [7.0, 22.0]],
            ]
        )
        stats = fit_feature_scaler(train_x)
        scaled = apply_feature_scaler(train_x, stats)

        self.assertTrue(np.allclose(scaled.mean(axis=(0, 1)), np.zeros(2)))
        self.assertTrue(np.allclose(scaled.std(axis=(0, 1)), np.ones(2)))

    def test_target_scaler_round_trips_values(self):
        y_train = np.array([10.0, 14.0, 18.0])
        stats = fit_target_scaler(y_train)
        scaled = apply_target_scaler(np.array([10.0, 18.0]), stats)
        restored = inverse_target_scaler(scaled, stats)

        self.assertTrue(np.allclose(restored, np.array([10.0, 18.0])))


if __name__ == "__main__":
    unittest.main()
