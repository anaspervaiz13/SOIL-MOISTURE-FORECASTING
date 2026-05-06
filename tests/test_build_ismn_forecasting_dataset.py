import unittest

import pandas as pd

from src.build_ismn_forecasting_dataset import (
    build_model_ready_dataset,
    consolidate_precipitation,
    create_target_48h,
    expand_station_hourly_grid,
    make_forecast_target_name,
)


class TestBuildIsmnForecastingDataset(unittest.TestCase):
    def test_make_forecast_target_name_uses_hour_horizon(self):
        self.assertEqual(make_forecast_target_name(24), "target_24h")
        self.assertEqual(make_forecast_target_name(48), "target_48h")
        self.assertEqual(make_forecast_target_name(72), "target_72h")

    def test_expand_station_hourly_grid_inserts_missing_hours(self):
        frame = pd.DataFrame(
            [
                {
                    "station": "A",
                    "timestamp": pd.Timestamp("2020-01-01 00:00:00"),
                    "latitude": 50.0,
                    "longitude": 6.0,
                    "elevation": 100.0,
                    "sm_0.05m": 0.10,
                    "precipitation": 0.0,
                },
                {
                    "station": "A",
                    "timestamp": pd.Timestamp("2020-01-01 02:00:00"),
                    "latitude": 50.0,
                    "longitude": 6.0,
                    "elevation": 100.0,
                    "sm_0.05m": 0.30,
                    "precipitation": 0.2,
                },
            ]
        )

        expanded = expand_station_hourly_grid(frame)

        self.assertEqual(expanded["timestamp"].tolist(), [
            pd.Timestamp("2020-01-01 00:00:00"),
            pd.Timestamp("2020-01-01 01:00:00"),
            pd.Timestamp("2020-01-01 02:00:00"),
        ])
        self.assertEqual(expanded["station"].tolist(), ["A", "A", "A"])
        self.assertTrue(pd.isna(expanded.loc[1, "sm_0.05m"]))
        self.assertEqual(expanded.loc[0, "latitude"], 50.0)
        self.assertEqual(expanded.loc[2, "elevation"], 100.0)

    def test_consolidate_precipitation_uses_priority_rule(self):
        frame = pd.DataFrame(
            [
                {
                    "p_ecotech_rain_gauge": 0.1,
                    "p_ott_pluvio2s_amount": 0.2,
                    "p_vaisala_wxt510": 0.3,
                },
                {
                    "p_ecotech_rain_gauge": None,
                    "p_ott_pluvio2s_amount": 0.4,
                    "p_vaisala_wxt510": 0.5,
                },
                {
                    "p_ecotech_rain_gauge": None,
                    "p_ott_pluvio2s_amount": None,
                    "p_vaisala_wxt510": 0.6,
                },
            ]
        )

        consolidated = consolidate_precipitation(frame)

        self.assertEqual(consolidated.tolist(), [0.1, 0.4, 0.6])

    def test_create_target_48h_shifts_future_surface_moisture_within_station(self):
        frame = pd.DataFrame(
            [
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 00:00:00"), "sm_0.05m": 0.10},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 01:00:00"), "sm_0.05m": 0.20},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 02:00:00"), "sm_0.05m": 0.30},
                {"station": "B", "timestamp": pd.Timestamp("2020-01-01 00:00:00"), "sm_0.05m": 0.40},
                {"station": "B", "timestamp": pd.Timestamp("2020-01-01 01:00:00"), "sm_0.05m": 0.50},
                {"station": "B", "timestamp": pd.Timestamp("2020-01-01 02:00:00"), "sm_0.05m": 0.60},
            ]
        )

        with_target = create_target_48h(frame, horizon_hours=1)

        self.assertEqual(with_target["target_1h"].iloc[0], 0.20)
        self.assertEqual(with_target["target_1h"].iloc[1], 0.30)
        self.assertTrue(pd.isna(with_target["target_1h"].iloc[2]))
        self.assertEqual(with_target["target_1h"].iloc[3], 0.50)
        self.assertEqual(with_target["target_1h"].iloc[4], 0.60)
        self.assertTrue(pd.isna(with_target["target_1h"].iloc[5]))

    def test_create_target_48h_can_build_other_horizons(self):
        frame = pd.DataFrame(
            [
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 00:00:00"), "sm_0.05m": 0.10},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 01:00:00"), "sm_0.05m": 0.20},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 02:00:00"), "sm_0.05m": 0.30},
            ]
        )

        with_target = create_target_48h(frame, horizon_hours=2)

        self.assertEqual(with_target["target_2h"].iloc[0], 0.30)
        self.assertTrue(pd.isna(with_target["target_2h"].iloc[1]))
        self.assertTrue(pd.isna(with_target["target_2h"].iloc[2]))

    def test_target_shift_respects_missing_hours_after_hourly_expansion(self):
        frame = pd.DataFrame(
            [
                {
                    "station": "A",
                    "timestamp": pd.Timestamp("2020-01-01 00:00:00"),
                    "latitude": 50.0,
                    "longitude": 6.0,
                    "elevation": 100.0,
                    "sm_0.05m": 0.10,
                    "precipitation": 0.0,
                },
                {
                    "station": "A",
                    "timestamp": pd.Timestamp("2020-01-01 02:00:00"),
                    "latitude": 50.0,
                    "longitude": 6.0,
                    "elevation": 100.0,
                    "sm_0.05m": 0.30,
                    "precipitation": 0.2,
                },
            ]
        )

        expanded = expand_station_hourly_grid(frame)
        with_target = create_target_48h(expanded, horizon_hours=1)

        self.assertTrue(pd.isna(with_target["target_1h"].iloc[0]))
        self.assertEqual(with_target["target_1h"].iloc[1], 0.30)
        self.assertTrue(pd.isna(with_target["target_1h"].iloc[2]))

    def test_build_model_ready_dataset_drops_rows_with_missing_features_or_target(self):
        frame = pd.DataFrame(
            [
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 00:00:00"), "target_48h": 0.2, "sm_0.05m": 0.1},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 01:00:00"), "target_48h": None, "sm_0.05m": 0.2},
                {"station": "A", "timestamp": pd.Timestamp("2020-01-01 02:00:00"), "target_48h": 0.4, "sm_0.05m": None},
            ]
        )

        ready = build_model_ready_dataset(frame)

        self.assertEqual(len(ready), 1)
        self.assertEqual(ready.iloc[0]["target_48h"], 0.2)


if __name__ == "__main__":
    unittest.main()
