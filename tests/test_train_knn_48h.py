import unittest

from src.train_knn_48h import parse_neighbor_values, resolve_output_dir_name


class TestTrainKnn48h(unittest.TestCase):
    def test_parse_neighbor_values_splits_comma_separated_values(self):
        self.assertEqual(parse_neighbor_values("5, 9,13"), [5, 9, 13])

    def test_resolve_output_dir_name_appends_tag_when_present(self):
        self.assertEqual(resolve_output_dir_name("short_lags", None), "short_lags")
        self.assertEqual(resolve_output_dir_name("short_lags", "light"), "short_lags__light")


if __name__ == "__main__":
    unittest.main()
