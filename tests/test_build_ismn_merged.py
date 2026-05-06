import unittest
from pathlib import Path

import pandas as pd

from src.build_ismn_merged import (
    aggregate_replicates,
    build_merged_hourly,
    parse_file_metadata,
    read_stm_file,
    summarize_merged_dataset,
)

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"


class TestBuildIsmnMerged(unittest.TestCase):
    def test_parse_file_metadata_builds_expected_feature_names(self):
        rainfall_path = Path(
            "data/original dataset/TERENO/Merzenhausen/"
            "TERENO_TERENO_Merzenhausen_p_-1.000000_-1.000000_OTT-Pluvio2S-amount_1_3_20091208_20260505.stm"
        )
        temperature_path = Path(
            "data/original dataset/TERENO/Gevenich/"
            "TERENO_TERENO_Gevenich_ta_-2.000000_-2.000000_Vaisala-WXT520_1_1_20091208_20260505.stm"
        )
        moisture_path = Path(
            "data/original dataset/TERENO/Selhausen/"
            "TERENO_TERENO_Selhausen_sm_0.050000_0.050000_Stevens-Hydraprobe-Digital_1_3_20091208_20260505.stm"
        )

        rainfall_meta = parse_file_metadata(rainfall_path)
        temperature_meta = parse_file_metadata(temperature_path)
        moisture_meta = parse_file_metadata(moisture_path)

        self.assertEqual(rainfall_meta["station"], "Merzenhausen")
        self.assertEqual(rainfall_meta["feature_name"], "p_ott_pluvio2s_amount")
        self.assertEqual(temperature_meta["feature_name"], "ta_2.00m")
        self.assertEqual(moisture_meta["feature_name"], "sm_0.05m")

    def test_read_stm_file_filters_non_g_quality_rows(self):
        stm_contents = "\n".join(
            [
                "TERENO TERENO TestStation 50.0 6.0 100.0 0.0500 0.0500 'Test-Sensor'",
                "2020/01/01 00:00 0.25 G 4_2002",
                "2020/01/01 01:00 0.35 B 4_2002",
                "2020/01/01 02:00 0.45 G 4_2002",
            ]
        )

        file_path = RUNTIME_DIR / "TERENO_TERENO_TestStation_sm_0.050000_0.050000_Test-Sensor_1_1_20091208_20260505.stm"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            file_path.write_text(stm_contents, encoding="utf-8")
            frame = read_stm_file(file_path)
        finally:
            if file_path.exists():
                file_path.unlink()

        self.assertEqual(frame["quality_flag"].tolist(), ["G", "G"])
        self.assertEqual(frame["value"].tolist(), [0.25, 0.45])
        self.assertEqual(frame["feature_name"].unique().tolist(), ["sm_0.05m"])

    def test_build_merged_hourly_aggregates_replicates_by_median(self):
        long_frame = pd.DataFrame(
            [
                {
                    "station": "TestStation",
                    "timestamp": pd.Timestamp("2020-01-01 00:00:00"),
                    "latitude": 50.0,
                    "longitude": 6.0,
                    "elevation": 100.0,
                    "feature_name": "sm_0.05m",
                    "value": 0.20,
                },
                {
                    "station": "TestStation",
                    "timestamp": pd.Timestamp("2020-01-01 00:00:00"),
                    "latitude": 50.0,
                    "longitude": 6.0,
                    "elevation": 100.0,
                    "feature_name": "sm_0.05m",
                    "value": 0.30,
                },
                {
                    "station": "TestStation",
                    "timestamp": pd.Timestamp("2020-01-01 00:00:00"),
                    "latitude": 50.0,
                    "longitude": 6.0,
                    "elevation": 100.0,
                    "feature_name": "ta_2.00m",
                    "value": 25.0,
                },
            ]
        )

        aggregated = aggregate_replicates(long_frame)
        merged = build_merged_hourly(aggregated)

        self.assertEqual(len(aggregated), 2)
        self.assertAlmostEqual(merged.loc[0, "sm_0.05m"], 0.25)
        self.assertAlmostEqual(merged.loc[0, "ta_2.00m"], 25.0)
        self.assertEqual(
            list(merged.columns),
            [
                "station",
                "timestamp",
                "latitude",
                "longitude",
                "elevation",
                "sm_0.05m",
                "ta_2.00m",
            ],
        )

    def test_summarize_merged_dataset_reports_core_counts(self):
        merged = pd.DataFrame(
            [
                {
                    "station": "A",
                    "timestamp": pd.Timestamp("2020-01-01 00:00:00"),
                    "latitude": 50.0,
                    "longitude": 6.0,
                    "elevation": 100.0,
                    "sm_0.05m": 0.20,
                    "ta_2.00m": 20.0,
                },
                {
                    "station": "B",
                    "timestamp": pd.Timestamp("2020-01-01 01:00:00"),
                    "latitude": 51.0,
                    "longitude": 7.0,
                    "elevation": 105.0,
                    "sm_0.05m": None,
                    "ta_2.00m": 21.0,
                },
            ]
        )

        summary = summarize_merged_dataset(merged)

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["columns"], 7)
        self.assertEqual(summary["stations"], ["A", "B"])
        self.assertEqual(summary["time_start"], "2020-01-01T00:00:00")
        self.assertEqual(summary["time_end"], "2020-01-01T01:00:00")
        self.assertEqual(summary["missingness"]["sm_0.05m"], 0.5)


if __name__ == "__main__":
    unittest.main()
