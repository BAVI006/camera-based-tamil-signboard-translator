"""Replaceable Tamil OCR service with a pretrained Tesseract implementation."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import unicodedata
from typing import Protocol

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output


class OCRConfigurationError(RuntimeError):
    """Raised when Tesseract or the requested language data is unavailable."""


class OCRProcessingError(RuntimeError):
    """Raised when an OCR request fails after the image is accepted."""


@dataclass(frozen=True)
class OCRWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float | None


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float | None
    words: tuple[OCRWord, ...]
    needs_review: bool
    warnings: tuple[str, ...]


class OCREngine(Protocol):
    """Interface future pretrained OCR engines can implement."""

    def recognize(self, image: Image.Image, language: str = "tam") -> OCRResult:
        ...


class TesseractOCREngine:
    """Tesseract adapter; confidence values use its native 0-100 TSV scale."""

    SUPPORTED_LANGUAGE_SETS = {"tam", "tam+eng"}

    def __init__(
        self,
        executable: str | None = None,
        timeout_seconds: float = 30,
        review_confidence_threshold: float = 50,
        retry_confidence_threshold: float = 75,
        blur_variance_threshold: float = 20,
    ):
        self.executable = executable or os.environ.get("TESSERACT_CMD")
        if self.executable is None:
            self.executable = shutil.which("tesseract")
        if self.executable is None:
            windows_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if windows_default.is_file():
                self.executable = str(windows_default)
        if self.executable is None:
            raise OCRConfigurationError(
                "Tesseract was not found. Install Tesseract OCR, then set "
                "TESSERACT_CMD to the full path of tesseract.exe."
            )

        self.timeout_seconds = timeout_seconds
        self.review_confidence_threshold = review_confidence_threshold
        self.retry_confidence_threshold = retry_confidence_threshold
        self.blur_variance_threshold = blur_variance_threshold
        self._available_languages: set[str] | None = None
        pytesseract.pytesseract.tesseract_cmd = self.executable

    def _languages(self) -> set[str]:
        if self._available_languages is None:
            try:
                self._available_languages = set(pytesseract.get_languages(config=""))
            except pytesseract.TesseractNotFoundError as exc:
                raise OCRConfigurationError(
                    "Tesseract could not be started. Check TESSERACT_CMD."
                ) from exc
            except pytesseract.TesseractError as exc:
                raise OCRConfigurationError(
                    f"Could not read Tesseract's installed language data: {exc}"
                ) from exc
        return self._available_languages

    @staticmethod
    def _validate_language(language: str) -> None:
        if language not in TesseractOCREngine.SUPPORTED_LANGUAGE_SETS:
            raise ValueError("language must be 'tam' or 'tam+eng'.")

    def _require_language_data(self, language: str) -> None:
        required = set(language.split("+"))
        missing = required - self._languages()
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise OCRConfigurationError(
                f"Tesseract language data is missing: {missing_names}. "
                "Install the matching traineddata file and retry."
            )

    @staticmethod
    def _clean_token(token: str) -> str:
        """Remove OCR punctuation noise without stripping Tamil combining marks."""
        start = 0
        end = len(token)
        while start < end and unicodedata.category(token[start])[0] not in "LMN":
            start += 1
        while end > start and unicodedata.category(token[end - 1])[0] not in "LMN":
            end -= 1
        return token[start:end]

    @classmethod
    def _clean_text(cls, text: str) -> str:
        lines = []
        for line in text.splitlines():
            tokens = [
                cleaned
                for token in line.split()
                if (cleaned := cls._clean_token(token))
            ]
            if tokens:
                lines.append(" ".join(tokens))
        return "\n".join(lines)

    def _read_candidate(self, image: Image.Image, language: str, psm: int):
        config = f"--oem 1 --psm {psm}"
        try:
            recognized_text = self._clean_text(
                pytesseract.image_to_string(
                    image,
                    lang=language,
                    config=config,
                    timeout=self.timeout_seconds,
                )
            )
            data = pytesseract.image_to_data(
                image,
                lang=language,
                config=config,
                output_type=Output.DICT,
                timeout=self.timeout_seconds,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRConfigurationError(
                "Tesseract could not be started. Check TESSERACT_CMD."
            ) from exc
        except (pytesseract.TesseractError, RuntimeError, OSError) as exc:
            raise OCRProcessingError(f"Tesseract OCR failed: {exc}") from exc

        words: list[OCRWord] = []
        confidences: list[float] = []
        for index, raw_text in enumerate(data.get("text", [])):
            word_text = self._clean_token(raw_text.strip())
            if not word_text:
                continue

            try:
                raw_confidence = float(data["conf"][index])
            except (KeyError, IndexError, TypeError, ValueError):
                raw_confidence = -1.0
            confidence = raw_confidence if raw_confidence >= 0 else None
            words.append(
                OCRWord(
                    text=word_text,
                    left=int(data["left"][index]),
                    top=int(data["top"][index]),
                    width=int(data["width"][index]),
                    height=int(data["height"][index]),
                    confidence=confidence,
                )
            )
            if confidence is not None:
                confidences.append(confidence)

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else None
        )
        return recognized_text, tuple(words), average_confidence

    @staticmethod
    def _find_sign_region(image: Image.Image) -> tuple[Image.Image, int, int] | None:
        """Find a large colored sign panel so surrounding scenery does not dominate OCR."""
        rgb = np.asarray(image.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturated = cv2.inRange(
            hsv,
            np.asarray((0, 80, 70), dtype=np.uint8),
            np.asarray((179, 255, 255), dtype=np.uint8),
        )
        min_dimension = min(image.size)
        kernel_size = max(3, min(31, int(min_dimension * 0.015) | 1))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        saturated = cv2.morphologyEx(saturated, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            saturated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_area = image.width * image.height
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            left, top, width, height = cv2.boundingRect(contour)
            aspect_ratio = width / height if height else 0
            if (
                width * height < image_area * 0.05
                or not 0.35 <= aspect_ratio <= 10
            ):
                continue
            padding = max(2, round(min_dimension * 0.02))
            x1 = max(0, left - padding)
            y1 = max(0, top - padding)
            x2 = min(image.width, left + width + padding)
            y2 = min(image.height, top + height + padding)
            return image.crop((x1, y1, x2, y2)), x1, y1
        return None

    def _recognize_single_image(self, image: Image.Image, language: str) -> OCRResult:
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        candidates = [self._read_candidate(image, language, psm=6)]
        initial_text, _, initial_confidence = candidates[0]
        if (
            not initial_text
            or initial_confidence is None
            or initial_confidence < self.retry_confidence_threshold
        ):
            candidates.append(self._read_candidate(image, language, psm=11))

        recognized_text, words, confidence = max(
            candidates,
            key=lambda candidate: (
                bool(candidate[0]),
                candidate[2] if candidate[2] is not None else -1,
            ),
        )

        warnings: list[str] = []
        if blur_variance < self.blur_variance_threshold:
            warnings.append("The image appears blurry or low-detail; OCR may be unreliable.")
        if not recognized_text:
            warnings.append("No readable text was detected.")
        elif confidence is None:
            warnings.append("Tesseract did not provide a usable confidence score.")
        elif confidence < self.review_confidence_threshold:
            warnings.append("OCR confidence is low; please review the detected text.")

        return OCRResult(
            text=recognized_text,
            confidence=confidence,
            words=words,
            needs_review=bool(warnings),
            warnings=tuple(warnings),
        )

    def recognize(self, image: Image.Image, language: str = "tam") -> OCRResult:
        self._validate_language(language)
        self._require_language_data(language)
        if not isinstance(image, Image.Image):
            raise ValueError("OCR input must be a decoded PIL image.")

        rgb_image = image.convert("RGB")
        results = [(self._recognize_single_image(rgb_image, language), 0, 0)]
        sign_region = self._find_sign_region(rgb_image)
        if sign_region is not None:
            cropped_image, offset_x, offset_y = sign_region
            results.append(
                (
                    self._recognize_single_image(cropped_image, language),
                    offset_x,
                    offset_y,
                )
            )

        result, offset_x, offset_y = max(
            results,
            key=lambda candidate: (
                bool(candidate[0].text),
                candidate[0].confidence
                if candidate[0].confidence is not None
                else -1,
            ),
        )
        return OCRResult(
            text=result.text,
            confidence=result.confidence,
            words=tuple(
                OCRWord(
                    text=word.text,
                    left=word.left + offset_x,
                    top=word.top + offset_y,
                    width=word.width,
                    height=word.height,
                    confidence=word.confidence,
                )
                for word in result.words
            ),
            needs_review=result.needs_review,
            warnings=result.warnings,
        )


def recognize_tamil_text(
    image: Image.Image,
    language: str = "tam",
    engine: OCREngine | None = None,
) -> OCRResult:
    """Recognize Tamil text with an injected engine or the Tesseract default."""
    selected_engine = engine or TesseractOCREngine()
    return selected_engine.recognize(image, language=language)
