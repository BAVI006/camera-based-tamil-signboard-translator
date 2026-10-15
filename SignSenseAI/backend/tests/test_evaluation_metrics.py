import unittest

from backend.scripts.evaluate_ocr import error_rates, edit_distance


class EvaluationMetricTests(unittest.TestCase):
    def test_edit_distance_uses_character_operations(self):
        self.assertEqual(edit_distance("cat", "cut"), 1)

    def test_cer_and_wer_are_zero_for_identical_text(self):
        self.assertEqual(error_rates("தமிழ் text", "தமிழ் text"), (0.0, 0.0))

    def test_word_error_rate_counts_word_changes(self):
        _, wer = error_rates("one two three", "one four three")
        self.assertAlmostEqual(wer, 1 / 3)


if __name__ == "__main__":
    unittest.main()
