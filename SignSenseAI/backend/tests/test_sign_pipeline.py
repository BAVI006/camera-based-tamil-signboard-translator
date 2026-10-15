from io import BytesIO
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.ocr_service import OCRResult, OCRWord
from backend.services.sign_pipeline import PipelineOptions, SignPipeline
from backend.services.translation_service import TranslationResult


def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (120, 80), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.word = OCRWord("மருத்துவமனை", 5, 5, 60, 15, 88.0)
        self.original_result = OCRResult(
            text="மருத்துவமனை",
            confidence=88.0,
            words=(self.word,),
            needs_review=False,
            warnings=(),
        )

    def test_pipeline_returns_required_fields_and_uses_model_translation(self):
        engine = Mock()
        engine.recognize.return_value = OCRResult(
            text="மருத்துவமனை அருகில்",
            confidence=88.0,
            words=(self.word,),
            needs_review=False,
            warnings=(),
        )
        translator = Mock()
        translator.translate.return_value = TranslationResult("Near the hospital")
        pipeline = SignPipeline(engine, translator)

        result = pipeline.analyze(image_bytes())

        self.assertEqual(result["tamil_text"], "மருத்துவமனை அருகில்")
        self.assertIsNotNone(result["transliteration"])
        self.assertEqual(result["english_translation"], "Near the hospital")
        self.assertEqual(result["category"], "healthcare")
        self.assertEqual(result["ocr_model"], "Tesseract OCR (tam+eng)")
        self.assertEqual(result["translation_model"], "configured-translator")
        self.assertEqual(
            result["context_model"],
            "Rule-based classifier + reviewed phrase dictionary",
        )
        self.assertEqual(result["confidence"], 88.0)
        self.assertIsInstance(result["processing_time_ms"], int)
        self.assertFalse(result["needs_review"])
        self.assertEqual(engine.recognize.call_count, 2)

    def test_pipeline_uses_exact_verified_phrase_when_model_is_unconfigured(self):
        engine = Mock()
        engine.recognize.return_value = self.original_result
        translator = Mock()
        from backend.services.translation_service import TranslationUnavailable
        translator.translate.side_effect = TranslationUnavailable("not configured")

        result = SignPipeline(engine, translator).analyze(image_bytes())

        self.assertEqual(result["english_translation"], "Hospital")
        self.assertEqual(result["translation_model"], "Reviewed phrase dictionary")
        self.assertTrue(any("verified phrase dictionary" in warning for warning in result["warnings"]))
        translator.translate.assert_not_called()

    def test_pipeline_does_not_invent_unverified_translation(self):
        engine = Mock()
        engine.recognize.return_value = OCRResult(
            text="அறியாத சொற்றொடர்",
            confidence=88.0,
            words=(),
            needs_review=False,
            warnings=(),
        )
        translator = Mock()
        from backend.services.translation_service import TranslationUnavailable
        translator.translate.side_effect = TranslationUnavailable("not configured")

        result = SignPipeline(engine, translator).analyze(image_bytes())

        self.assertIsNone(result["english_translation"])
        self.assertIsNone(result["translation_model"])
        self.assertTrue(result["needs_review"])
        self.assertTrue(any("not configured" in warning for warning in result["warnings"]))

    def test_pipeline_returns_verified_translation_for_litter_sign(self):
        engine = Mock()
        engine.recognize.return_value = OCRResult(
            text="! குப்பை\nபோடாதே |",
            confidence=82.0,
            words=(),
            needs_review=False,
            warnings=(),
        )
        translator = Mock()
        from backend.services.translation_service import TranslationUnavailable
        translator.translate.side_effect = TranslationUnavailable("not configured")

        result = SignPipeline(engine, translator).analyze(image_bytes())

        self.assertEqual(result["tamil_text"], "! குப்பை\nபோடாதே |")
        self.assertEqual(result["transliteration"], "Kuppai podaathae")
        self.assertEqual(result["english_translation"], "Do not litter.")
        self.assertEqual(result["translation_model"], "Reviewed phrase dictionary")
        self.assertEqual(result["category"], "warnings_prohibitions")
        self.assertTrue(any("verified phrase dictionary" in warning for warning in result["warnings"]))

    def test_verified_phrase_is_not_overridden_by_inaccurate_model_output(self):
        engine = Mock()
        engine.recognize.return_value = OCRResult(
            text="குப்பை போடாதே",
            confidence=95.0,
            words=(),
            needs_review=False,
            warnings=(),
        )
        translator = Mock()
        translator.translate.return_value = TranslationResult("What is it?")

        result = SignPipeline(engine, translator).analyze(image_bytes())

        self.assertEqual(result["english_translation"], "Do not litter.")
        self.assertEqual(result["translation_model"], "Reviewed phrase dictionary")
        translator.translate.assert_not_called()

    def test_fertility_center_ocr_variant_uses_reviewed_translation_and_context(self):
        engine = Mock()
        engine.recognize.return_value = OCRResult(
            text="ஐஸ்வர்யா\nக த்தமித்தல் மையம்\nகனவு",
            confidence=82.0,
            words=(),
            needs_review=False,
            warnings=(),
        )
        translator = Mock()
        translator.translate.return_value = TranslationResult("The Center for Gathering")

        result = SignPipeline(engine, translator).analyze(image_bytes())

        self.assertEqual(result["tamil_text"], "ஐஸ்வர்யா\nக த்தமித்தல் மையம்\nகனவு")
        self.assertIn("Karuththarithal maiyam", result["transliteration"])
        self.assertEqual(result["english_translation"], "Fertility Centre")
        self.assertEqual(result["translation_model"], "Reviewed phrase dictionary")
        self.assertEqual(result["category"], "healthcare")
        self.assertIn("fertility centre", result["context"])
        self.assertTrue(
            any("reviewed spelling variant" in warning for warning in result["warnings"])
        )
        translator.translate.assert_not_called()

    def test_quiet_sign_uses_reviewed_translation_not_nllb_wording(self):
        engine = Mock()
        engine.recognize.return_value = OCRResult(
            text="அமைதியாக இருங்கள்",
            confidence=88.5,
            words=(),
            needs_review=False,
            warnings=(),
        )
        translator = Mock()
        translator.translate.return_value = TranslationResult("Quietly Staying")

        result = SignPipeline(engine, translator).analyze(image_bytes())

        self.assertEqual(result["english_translation"], "Please remain quiet.")
        self.assertEqual(result["transliteration"], "Amaithiyaaga irungal")
        self.assertEqual(result["category"], "warnings_prohibitions")
        self.assertEqual(result["context"], "This sign asks people to remain quiet.")
        self.assertEqual(result["translation_model"], "Reviewed phrase dictionary")
        translator.translate.assert_not_called()

    def test_pipeline_can_disable_original_processed_ocr_comparison(self):
        engine = Mock()
        engine.recognize.return_value = self.original_result
        translator = Mock()
        translator.translate.return_value = TranslationResult("Hospital")

        SignPipeline(engine, translator).analyze(
            image_bytes(),
            PipelineOptions(compare_preprocessing=False),
        )

        self.assertEqual(engine.recognize.call_count, 1)

    def test_pipeline_does_not_flag_line_break_only_ocr_difference(self):
        original, outputs_differ = SignPipeline._select_ocr_result(
            OCRResult("குப்பை\nபோடாதே", 90.0, (), False, ()),
            OCRResult("குப்பை போடாதே", 75.0, (), False, ()),
        )

        self.assertEqual(original.confidence, 90.0)
        self.assertFalse(outputs_differ)


class FastAPITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_route_reports_configured_components(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["ocr_engine"], "tesseract")

    def test_rejects_unsupported_translation_target(self):
        response = self.client.post(
            "/api/v1/translate?translation_language=hin_Deva",
            files={"file": ("sign.png", image_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 422)

    def test_rejects_invalid_image_upload(self):
        response = self.client.post(
            "/api/v1/translate",
            files={"file": ("bad.png", b"not an image", "image/png")},
        )

        self.assertEqual(response.status_code, 400)

    def test_returns_structured_result_for_valid_image(self):
        expected = {
            "tamil_text": "மருத்துவமனை",
            "transliteration": "marudhdhuvamanai",
            "english_translation": "Hospital",
            "category": "healthcare",
            "context": "Healthcare-related sign.",
            "confidence": 88.0,
            "needs_review": False,
            "processing_time_ms": 42,
            "warnings": [],
            "evidence": ["Hospital"],
            "ocr_model": "Tesseract OCR (tam+eng)",
            "translation_model": "Reviewed phrase dictionary",
            "context_model": "Rule-based classifier + reviewed phrase dictionary",
            "ocr_model": "Tesseract OCR (tam+eng)",
            "translation_model": "Reviewed phrase dictionary",
            "context_model": "Rule-based classifier + reviewed phrase dictionary",
        }
        with patch(
            "backend.main.pipeline.analyze",
            return_value=expected,
        ), patch("backend.main.history.save"):
            response = self.client.post(
                "/api/v1/translate",
                files={"file": ("sign.png", image_bytes(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)

    def test_history_routes_use_persistent_store_interface(self):
        with patch(
            "backend.main.history.list_recent",
            return_value=[{"id": 1, "created_at": "now", "file_name": "sign.png", "result": {}}],
        ) as list_recent:
            response = self.client.get("/api/v1/history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["items"]), 1)
        list_recent.assert_called_once_with(50)

        with patch("backend.main.history.clear") as clear:
            response = self.client.delete("/api/v1/history")
        self.assertEqual(response.json(), {"cleared": True})
        clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
