import unittest
from unittest.mock import Mock, patch

from backend.services.translation_service import (
    HostedTranslationAPI,
    IndicTrans2Translator,
    NLLBTranslator,
    TranslationError,
    TranslationResult,
    TranslationUnavailable,
    create_translator,
)


class TranslationServiceTests(unittest.TestCase):
    def test_no_default_translation_model_is_reported_as_unavailable(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(TranslationUnavailable):
                create_translator()

    def test_empty_text_is_not_sent_to_hosted_endpoint(self):
        translator = HostedTranslationAPI("https://example.invalid/translate")

        with self.assertRaises(TranslationError):
            translator.translate(" ")

    def test_hosted_adapter_sends_configured_target_and_returns_response(self):
        response = Mock()
        response.json.return_value = {"translation": "Hospital"}
        with patch("httpx.post", return_value=response) as request:
            result = HostedTranslationAPI(
                "https://translation.example/translate",
                api_key="test-key",
            ).translate("மருத்துவமனை")

        request.assert_called_once()
        self.assertEqual(result, TranslationResult("Hospital", model="hosted-api"))
        self.assertEqual(
            request.call_args.kwargs["json"]["source_language"],
            "tam_Taml",
        )
        self.assertEqual(
            request.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-key",
        )

    def test_indictrans2_rejects_empty_and_unbounded_text_before_loading(self):
        translator = IndicTrans2Translator()
        translator._load = Mock(side_effect=AssertionError("Model must not load"))

        with self.assertRaises(TranslationError):
            translator.translate("")
        with self.assertRaises(TranslationError):
            translator.translate("x" * 5001)

        translator._load.assert_not_called()

    def test_nllb_rejects_empty_and_unsupported_text_before_loading(self):
        translator = NLLBTranslator()
        translator._load = Mock(side_effect=AssertionError("Model must not load"))

        with self.assertRaises(TranslationError):
            translator.translate("")
        with self.assertRaises(TranslationError):
            translator.translate("மருத்துவமனை", target_language="hin_Deva")

        translator._load.assert_not_called()

    def test_create_translator_selects_configured_nllb_backend(self):
        with patch.dict("os.environ", {"SIGN_TRANSLATION_BACKEND": "nllb"}, clear=True):
            self.assertIsInstance(create_translator(), NLLBTranslator)


if __name__ == "__main__":
    unittest.main()
