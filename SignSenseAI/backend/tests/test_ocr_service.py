import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import pytesseract

from backend.services.ocr_service import (
    OCRConfigurationError,
    OCRProcessingError,
    OCRWord,
    TesseractOCREngine,
)


def sample_tsv():
    return {
        "text": ["மருத்துவமனை", "Hospital", ""],
        "conf": ["78.0", "92.0", "-1"],
        "left": ["10", "100", "0"],
        "top": ["15", "15", "0"],
        "width": ["80", "60", "0"],
        "height": ["20", "20", "0"],
    }


def detailed_test_image():
    texture = (np.indices((64, 128)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return Image.fromarray(texture).convert("RGB")


class OCRServiceTests(unittest.TestCase):
    def make_engine(self):
        engine = TesseractOCREngine(executable="tesseract-test")
        engine._available_languages = {"tam", "eng"}
        return engine

    def test_returns_real_tesseract_confidence_and_word_boxes(self):
        engine = self.make_engine()
        image = detailed_test_image()

        with (
            patch("backend.services.ocr_service.pytesseract.image_to_string",
                 return_value="மருத்துவமனை Hospital"),
            patch("backend.services.ocr_service.pytesseract.image_to_data",
                 return_value=sample_tsv()),
        ):
            result = engine.recognize(image, language="tam+eng")

        self.assertEqual(result.text, "மருத்துவமனை Hospital")
        self.assertEqual(result.confidence, 85.0)
        self.assertEqual(len(result.words), 2)
        self.assertEqual(result.words[0].left, 10)
        self.assertEqual(result.words[0].confidence, 78.0)
        self.assertFalse(result.needs_review)

    def test_checks_requested_tamil_language_data_before_ocr(self):
        engine = self.make_engine()
        engine._available_languages = {"eng"}

        with self.assertRaisesRegex(OCRConfigurationError, "tam"):
            engine.recognize(Image.new("RGB", (20, 20), "white"))

    def test_empty_result_has_no_fabricated_confidence_and_needs_review(self):
        engine = self.make_engine()
        data = {
            "text": [""],
            "conf": ["-1"],
            "left": ["0"],
            "top": ["0"],
            "width": ["0"],
            "height": ["0"],
        }
        detailed_image = detailed_test_image()

        with (
            patch("backend.services.ocr_service.pytesseract.image_to_string",
                 return_value=""),
            patch("backend.services.ocr_service.pytesseract.image_to_data",
                 return_value=data),
        ):
            result = engine.recognize(detailed_image)

        self.assertEqual(result.text, "")
        self.assertIsNone(result.confidence)
        self.assertTrue(result.needs_review)
        self.assertIn("No readable text was detected.", result.warnings)

    def test_discards_punctuation_noise_but_preserves_tamil_text(self):
        engine = self.make_engine()
        data = {
            "text": ["!", "குப்பை", "போடாதே", "|"],
            "conf": ["20.0", "90.0", "88.0", "20.0"],
            "left": ["0", "10", "90", "0"],
            "top": ["0", "10", "10", "0"],
            "width": ["2", "60", "70", "2"],
            "height": ["2", "20", "20", "2"],
        }
        with (
            patch(
                "backend.services.ocr_service.pytesseract.image_to_string",
                return_value="! குப்பை\nபோடாதே |",
            ),
            patch(
                "backend.services.ocr_service.pytesseract.image_to_data",
                return_value=data,
            ),
        ):
            result = engine.recognize(detailed_test_image())

        self.assertEqual(result.text, "குப்பை\nபோடாதே")
        self.assertEqual([word.text for word in result.words], ["குப்பை", "போடாதே"])
        self.assertEqual(result.confidence, 89.0)
        self.assertFalse(result.needs_review)

    def test_retries_sparse_page_segmentation_when_default_ocr_is_unreliable(self):
        engine = self.make_engine()
        low_confidence_data = {
            "text": ["ர்", "ன"],
            "conf": ["59.0", "59.0"],
            "left": ["0", "20"],
            "top": ["0", "0"],
            "width": ["10", "10"],
            "height": ["10", "10"],
        }
        good_data = {
            "text": ["குப்பை", "போடாதே"],
            "conf": ["95.0", "94.0"],
            "left": ["50", "130"],
            "top": ["40", "40"],
            "width": ["70", "90"],
            "height": ["24", "24"],
        }
        with (
            patch(
                "backend.services.ocr_service.pytesseract.image_to_string",
                side_effect=["ர் ன", "குப்பை\nபோடாதே"],
            ) as image_to_string_mock,
            patch(
                "backend.services.ocr_service.pytesseract.image_to_data",
                side_effect=[low_confidence_data, good_data],
            ),
        ):
            result = engine.recognize(detailed_test_image(), language="tam+eng")

        self.assertEqual(result.text, "குப்பை\nபோடாதே")
        self.assertEqual(result.confidence, 94.5)
        self.assertEqual([word.text for word in result.words], ["குப்பை", "போடாதே"])
        self.assertEqual(image_to_string_mock.call_count, 2)
        self.assertIn("--psm 6", image_to_string_mock.call_args_list[0].kwargs["config"])
        self.assertIn("--psm 11", image_to_string_mock.call_args_list[1].kwargs["config"])

    def test_uses_high_confidence_text_from_large_colored_sign_region(self):
        engine = self.make_engine()
        image = Image.new("RGB", (500, 500), "white")
        ImageDraw.Draw(image).rectangle((60, 120, 440, 300), fill="yellow")

        def recognize_candidate(candidate_image, language, psm):
            if candidate_image.width == image.width:
                return "வி", (
                    OCRWord("வி", 100, 150, 20, 20, 59.0),
                ), 59.0
            return "குப்பை\nபோடாதே", (
                OCRWord("குப்பை", 20, 20, 80, 24, 95.0),
                OCRWord("போடாதே", 20, 60, 90, 24, 94.0),
            ), 94.5

        with patch.object(engine, "_read_candidate", side_effect=recognize_candidate):
            result = engine.recognize(image, language="tam+eng")

        self.assertEqual(result.text, "குப்பை\nபோடாதே")
        self.assertEqual(result.confidence, 94.5)
        self.assertEqual(result.words[0].left, 70)
        self.assertEqual(result.words[0].top, 130)

    def test_uniform_image_is_flagged_as_low_detail(self):
        engine = self.make_engine()
        with (
            patch("backend.services.ocr_service.pytesseract.image_to_string",
                 return_value="மருத்துவமனை"),
            patch("backend.services.ocr_service.pytesseract.image_to_data",
                 return_value=sample_tsv()),
        ):
            result = engine.recognize(Image.new("RGB", (80, 80), "white"))

        self.assertTrue(result.needs_review)
        self.assertTrue(any("blurry or low-detail" in warning for warning in result.warnings))

    def test_invalid_language_choice_is_rejected(self):
        engine = self.make_engine()
        with self.assertRaisesRegex(ValueError, "'tam' or 'tam\\+eng'"):
            engine.recognize(Image.new("RGB", (20, 20), "white"), language="eng")

    def test_tesseract_timeout_is_reported_as_processing_error(self):
        engine = self.make_engine()
        detailed_image = detailed_test_image()
        with (
            patch("backend.services.ocr_service.pytesseract.image_to_string",
                 side_effect=RuntimeError("Tesseract process timeout")),
        ):
            with self.assertRaisesRegex(OCRProcessingError, "timeout"):
                engine.recognize(detailed_image)

    def test_unavailable_tesseract_binary_is_reported(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("backend.services.ocr_service.shutil.which", return_value=None),
            patch("backend.services.ocr_service.Path.is_file", return_value=False),
        ):
            with self.assertRaisesRegex(OCRConfigurationError, "TESSERACT_CMD"):
                TesseractOCREngine()

    def test_tesseract_executable_can_be_configured_by_environment(self):
        with patch.dict("os.environ", {"TESSERACT_CMD": r"C:\OCR\tesseract.exe"}):
            engine = TesseractOCREngine()

        self.assertEqual(engine.executable, r"C:\OCR\tesseract.exe")
        self.assertEqual(pytesseract.pytesseract.tesseract_cmd, r"C:\OCR\tesseract.exe")


if __name__ == "__main__":
    unittest.main()
