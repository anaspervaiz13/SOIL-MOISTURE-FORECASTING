import unittest

from src.train_lstm_48h import default_model_config, parse_seeds, resolve_output_dir_name


class TestTrainLstm48h(unittest.TestCase):
    def test_parse_seeds_splits_comma_separated_values(self):
        self.assertEqual(parse_seeds("42, 52,62"), [42, 52, 62])

    def test_resolve_output_dir_name_appends_tag_when_present(self):
        self.assertEqual(resolve_output_dir_name("lstm", None), "lstm")
        self.assertEqual(resolve_output_dir_name("lstm", "cpu40"), "lstm__cpu40")

    def test_default_model_config_matches_repaired_baseline(self):
        config = default_model_config()

        self.assertEqual(config["lstm_units"], [64, 32])
        self.assertEqual(config["dropout"], 0.2)
        self.assertEqual(config["learning_rate"], 0.001)


if __name__ == "__main__":
    unittest.main()
