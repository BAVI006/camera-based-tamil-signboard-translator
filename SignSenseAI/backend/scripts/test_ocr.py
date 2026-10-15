"""Standalone OCR comparison for one signboard image."""

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from backend.services.ocr_service import (  # noqa: E402
    OCRConfigurationError,
    OCRProcessingError,
    recognize_tamil_text,
)
from backend.services.preprocessing import (  # noqa: E402
    ImageValidationError,
    preprocess_image,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Tesseract OCR on an original and preprocessed image."
    )
    parser.add_argument("image", type=Path, help="Path to a JPG, PNG, or WEBP image.")
    parser.add_argument(
        "--language",
        choices=("tam", "tam+eng"),
        default="tam",
        help="Tamil only (default) or mixed Tamil and English.",
    )
    args = parser.parse_args()

    try:
        image_bytes = args.image.read_bytes()
        prepared = preprocess_image(image_bytes)
        original_result = recognize_tamil_text(
            prepared.original,
            language=args.language,
        )
        processed_result = recognize_tamil_text(
            prepared.processed,
            language=args.language,
        )
    except (OSError, ImageValidationError, OCRConfigurationError, OCRProcessingError, ValueError) as exc:
        print(f"OCR test failed: {exc}", file=sys.stderr)
        return 1

    for label, result in (
        ("Orientation-corrected original", original_result),
        ("Preprocessed", processed_result),
    ):
        print(f"\n{label}:")
        print(f"Text: {result.text or '[No readable text detected]'}")
        print(
            "Tesseract confidence (0-100): "
            f"{result.confidence:.1f}" if result.confidence is not None
            else "Tesseract confidence: unavailable"
        )
        print(f"Word boxes: {len(result.words)}")
        print(f"Needs review: {result.needs_review}")
        for warning in result.warnings:
            print(f"Warning: {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
