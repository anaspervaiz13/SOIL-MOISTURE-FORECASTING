import unittest

from src.train_lightgbm_48h import parse_seeds, resolve_output_dir_name


class TestTrainLightgbm48h(unittest.TestCase):
    def test_parse_seeds_splits_comma_separated_values(self):
        self.assertEqual(parse_seeds("42, 52,62"), [42, 52, 62])

    def test_resolve_output_dir_name_appends_tag_when_present(self):
        self.assertEqual(resolve_output_dir_name("full_multiscale", None), "full_multiscale")
        self.assertEqual(resolve_output_dir_name("full_multiscale", "full"), "full_multiscale__full")


if __name__ == "__main__":
    unittest.main()
