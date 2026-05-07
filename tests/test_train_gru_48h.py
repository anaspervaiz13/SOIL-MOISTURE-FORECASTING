import unittest

from src.train_gru_48h import parse_args
from src.train_lstm_48h import parse_lstm_units, parse_seeds, resolve_output_dir_name


class TestTrainGru48h(unittest.TestCase):
    def test_parse_seeds_splits_comma_separated_values(self):
        self.assertEqual(parse_seeds("42, 52,62"), [42, 52, 62])

    def test_parse_lstm_units_supports_gru_units_text(self):
        self.assertEqual(parse_lstm_units("64,32"), [64, 32])

    def test_resolve_output_dir_name_appends_tag_when_present(self):
        self.assertEqual(resolve_output_dir_name("gru", None), "gru")
        self.assertEqual(resolve_output_dir_name("gru", "repaired40"), "gru__repaired40")


if __name__ == "__main__":
    unittest.main()
