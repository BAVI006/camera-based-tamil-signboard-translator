"""Runs preprocessing, OCR, translation, transliteration, and context in order."""

from dataclasses import dataclass
import time
from threading import Lock
from typing import Any

from PIL import Image

from backend.services.context_service import (
    classify_sign,
    lookup_verified_phrase,
    verified_phrase_matched_canonical_text,
    verified_phrase_matches_full_text,
)
from backend.services.ocr_service import OCREngine, OCRResult, TesseractOCREngine
from backend.services.preprocessing import PreprocessedImage, preprocess_image
from backend.services.transliteration_service import transliterate_tamil
from backend.services.translation_service import (
    TranslationError,
    TranslationUnavailable,
    Translator,
    create_translator,
)


class UnsupportedTranslationTarget(ValueError):
    """Raised when a requested target has no configured translation model."""


@dataclass(frozen=True)
class PipelineOptions:
    compare_preprocessing: bool = True
    translation_target: str = "eng_Latn"
    ocr_language: str = "tam+eng"
    transliteration_enabled: bool = True


class SignPipeline:
    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        translator: Translator | None = None,
    ):
        self._ocr_engine = ocr_engine
        self._translator = translator
        self._ocr_lock = Lock()
        self._translation_lock = Lock()

    def _get_ocr_engine(self) -> OCREngine:
        if self._ocr_engine is None:
            with self._ocr_lock:
                if self._ocr_engine is None:
                    self._ocr_engine = TesseractOCREngine()
        return self._ocr_engine

    def _get_translator(self) -> Translator:
        if self._translator is None:
            with self._translation_lock:
                if self._translator is None:
                    self._translator = create_translator()
        return self._translator

    @staticmethod
    def _select_ocr_result(original: OCRResult, processed: OCRResult) -> tuple[OCRResult, bool]:
        outputs_differ = " ".join(original.text.split()).casefold() != (
            " ".join(processed.text.split()).casefold()
        )
        if original.text and not processed.text:
            return original, False
        if processed.text and not original.text:
            return processed, False
        if not original.text and not processed.text:
            return original, False
        if original.confidence is not None and processed.confidence is not None:
            return (processed, outputs_differ) if (
                processed.confidence > original.confidence
            ) else (original, outputs_differ)
        return original, outputs_differ

    def analyze(
        self,
        image_bytes: bytes,
        options: PipelineOptions | None = None,
    ) -> dict[str, Any]:
        options = options or PipelineOptions()
        if options.translation_target != "eng_Latn":
            raise UnsupportedTranslationTarget(
                "English is currently the only enabled translation target."
            )

        started = time.perf_counter()
        prepared: PreprocessedImage = preprocess_image(image_bytes)
        engine = self._get_ocr_engine()
        original_ocr = engine.recognize(
            prepared.original,
            language=options.ocr_language,
        )
        if options.compare_preprocessing:
            processed_ocr = engine.recognize(
                prepared.processed,
                language=options.ocr_language,
            )
        else:
            processed_ocr = original_ocr

        chosen, outputs_differ = self._select_ocr_result(original_ocr, processed_ocr)
        warnings = list(chosen.warnings)
        if outputs_differ:
            warnings.append(
                "Original and preprocessed OCR results differ; review the source image."
            )

        verified_phrase = lookup_verified_phrase(chosen.text)
        translation: str | None = None
        translation_model: str | None = None
        if verified_phrase is not None:
            translation = verified_phrase["english"]
            translation_model = "Reviewed phrase dictionary"
            if verified_phrase_matched_canonical_text(chosen.text, verified_phrase):
                warnings.append(
                    "Used an exact reviewed phrase match; the translation is from the "
                    "verified phrase dictionary, not the AI model."
                )
            else:
                warnings.append(
                    "The OCR text matched a reviewed spelling variant of "
                    f"'{verified_phrase['tamil']}'; its dictionary translation is "
                    "provisional because the detected spelling differs."
                )
        else:
            try:
                translator = self._get_translator()
                translated = translator.translate(
                    chosen.text,
                    target_language=options.translation_target,
                )
                translation = translated.text
                translation_model = translated.model
                if translated.model.startswith("facebook/nllb"):
                    warnings.append(
                        "Machine-generated NLLB translation; verify its wording "
                        "against the original sign."
                    )
            except (TranslationUnavailable, TranslationError) as exc:
                warnings.append(str(exc))

        transliteration = None
        if options.transliteration_enabled and chosen.text.strip():
            if verified_phrase and verified_phrase_matches_full_text(
                chosen.text, verified_phrase
            ):
                transliteration = verified_phrase["transliteration"]
            else:
                transliterated_lines = []
                for line in chosen.text.splitlines():
                    line_phrase = lookup_verified_phrase(line)
                    transliterated_lines.append(
                        line_phrase["transliteration"]
                        if line_phrase is not None
                        else transliterate_tamil(line)
                    )
                transliteration = " ".join(
                    part for part in transliterated_lines if part
                ) or transliterate_tamil(chosen.text)
        context = classify_sign(
            chosen.text,
            english_translation=translation,
            confidence=chosen.confidence,
        )
        if context["needs_review"] or warnings:
            needs_review = True
        else:
            needs_review = False

        return {
            "tamil_text": chosen.text,
            "transliteration": transliteration,
            "english_translation": translation,
            "category": context["category"],
            "context": context["context"],
            "confidence": chosen.confidence,
            "needs_review": needs_review,
            "processing_time_ms": round((time.perf_counter() - started) * 1000),
            "warnings": warnings,
            "evidence": context["evidence"],
            "ocr_model": f"Tesseract OCR ({options.ocr_language})",
            "translation_model": translation_model,
            "context_model": "Rule-based classifier + reviewed phrase dictionary",
        }
