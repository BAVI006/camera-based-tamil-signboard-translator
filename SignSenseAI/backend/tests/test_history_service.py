import tempfile
import unittest
from pathlib import Path

from backend.services.history_service import HistoryService


class HistoryServiceTests(unittest.TestCase):
    def test_saves_lists_and_clears_translation_results_without_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            service = HistoryService(Path(directory) / "history.sqlite3")
            result = {
                "tamil_text": "மருத்துவமனை",
                "english_translation": None,
            }

            saved = service.save(r"C:\private\sign.png", result)
            items = service.list_recent()

            self.assertEqual(saved["file_name"], "sign.png")
            self.assertEqual(items[0]["result"], result)
            self.assertTrue((Path(directory) / "history.sqlite3").is_file())
            service.clear()
            self.assertEqual(service.list_recent(), [])

    def test_rejects_history_limits_outside_supported_range(self):
        with tempfile.TemporaryDirectory() as directory:
            service = HistoryService(Path(directory) / "history.sqlite3")
            with self.assertRaises(ValueError):
                service.list_recent(0)


if __name__ == "__main__":
    unittest.main()
