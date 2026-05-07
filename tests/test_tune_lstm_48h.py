import unittest

from src.tune_lstm_48h import build_fast_tuning_runs, format_tuning_output_tag


class TestTuneLstm48h(unittest.TestCase):
    def test_build_fast_tuning_runs_returns_small_distinct_sweep(self):
        runs = build_fast_tuning_runs()

        self.assertEqual(len(runs), 5)
        self.assertEqual(runs[0]["lookback"], 120)
        self.assertIn(0.0005, [run["learning_rate"] for run in runs])
        self.assertIn(128, [run["batch_size"] for run in runs])

    def test_format_tuning_output_tag_encodes_key_choices(self):
        run = {
            "name": "lb120_lr5e4",
            "lookback": 120,
            "batch_size": 256,
            "dropout": 0.2,
            "learning_rate": 0.0005,
            "lstm_units": [64, 32],
        }

        self.assertEqual(format_tuning_output_tag(run), "tune_lb120_lr5e4")


if __name__ == "__main__":
    unittest.main()
