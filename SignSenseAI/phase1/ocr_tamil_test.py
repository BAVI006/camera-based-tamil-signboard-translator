
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytesseract


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "data" / "sample_signboard.jpg"


def find_tesseract_executable():
    """Find the installed Tesseract executable."""

    configured = shutil.which("tesseract")
    if configured:
        return configured

    windows_default = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if Path(windows_default).exists():
        return windows_default

    raise FileNotFoundError(
        "Tesseract OCR was not found.\n"
        "Check the installation at:\n"
        f"{windows_default}"
    )


def configure_tesseract():
    """Point pytesseract at the installed Tesseract executable."""
    pytesseract.pytesseract.tesseract_cmd = find_tesseract_executable()


def preprocess_image(image):
    """Prepare the image for OCR."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return processed


def read_tamil_ocr_from_image(image):
    """Extract Tamil text from an image already decoded by OpenCV."""
    processed = preprocess_image(image)

    try:
        return pytesseract.image_to_string(processed, lang="tam").strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR could not be started. Check its installation."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise RuntimeError(
            "Tamil OCR failed. Check that Tesseract's Tamil language data is installed. "
            f"Details: {exc}"
        ) from exc


def read_tamil_ocr_from_bytes(image_bytes):
    """Decode uploaded image bytes and extract Tamil text."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("This file is not a supported or readable image.")

    return read_tamil_ocr_from_image(image)


def read_tamil_ocr(image_path: Path):
    """Extract Tamil text from a signboard image."""

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            "Place sample_signboard.jpg inside the data folder."
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return read_tamil_ocr_from_image(image)


def main():
    configure_tesseract()

    print("SignSense AI - Phase 1 OCR Test")
    print(f"Reading image: {IMAGE_PATH}\n")

    print("Available OCR languages:")
    print(pytesseract.get_languages(config=""))

    text = read_tamil_ocr(IMAGE_PATH)

    print("\nOCR result:")
    print(text if text else "[No text detected]")


if __name__ == "__main__":
    main()
