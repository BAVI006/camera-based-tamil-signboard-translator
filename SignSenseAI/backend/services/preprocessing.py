"""Validate and gently improve signboard images before OCR."""

from dataclasses import dataclass
from io import BytesIO
from typing import TypeAlias

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


Point: TypeAlias = tuple[float, float]
PerspectiveCorners: TypeAlias = tuple[Point, Point, Point, Point]

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a safe, readable image."""


@dataclass(frozen=True)
class PreprocessingOptions:
    """Optional image operations; gentle contrast and denoising are defaults."""

    max_dimension: int = 2000
    enhance_contrast: bool = True
    reduce_noise: bool = True
    grayscale: bool = False
    threshold: bool = False
    sharpen: bool = False
    perspective_corners: PerspectiveCorners | None = None


@dataclass(frozen=True)
class PreprocessedImage:
    """Uploaded bytes, oriented original, and OCR-ready processed image."""

    original_bytes: bytes
    original: Image.Image
    processed: Image.Image
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


def _decode_image(image_bytes: bytes) -> Image.Image:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ImageValidationError("The uploaded image is empty or invalid.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("The image is larger than the 15 MB upload limit.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    "Unsupported image format. Upload a JPG, PNG, or WEBP image."
                )
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageValidationError(
                    "The image dimensions are too large to process safely."
                )
            image.verify()

        with Image.open(BytesIO(image_bytes)) as image:
            oriented = ImageOps.exif_transpose(image)
            oriented.load()
            if oriented.width * oriented.height > MAX_IMAGE_PIXELS:
                raise ImageValidationError(
                    "The oriented image dimensions are too large to process safely."
                )
            return oriented.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageValidationError(
            "The uploaded file is not a readable image. Try another JPG, PNG, or WEBP."
        ) from exc


def _resize(image: Image.Image, max_dimension: int) -> Image.Image:
    if max_dimension <= 0:
        raise ValueError("max_dimension must be a positive integer.")
    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_dimension:
        return image.copy()

    scale = max_dimension / largest_side
    target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(target_size, Image.Resampling.LANCZOS)


def _apply_perspective(
    image: np.ndarray,
    corners: PerspectiveCorners,
) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("perspective_corners must contain four finite (x, y) points.")

    height, width = image.shape[:2]
    if (
        (points[:, 0] < 0).any()
        or (points[:, 0] >= width).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] >= height).any()
    ):
        raise ValueError("Perspective points must lie inside the resized image.")

    top_left, top_right, bottom_right, bottom_left = points
    output_width = round(
        max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    )
    output_height = round(
        max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    )
    if output_width < 2 or output_height < 2:
        raise ValueError("Perspective points do not define a usable rectangle.")

    destination = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(
    image_bytes: bytes,
    options: PreprocessingOptions | None = None,
) -> PreprocessedImage:
    """Validate, orient, and optionally improve an uploaded image.

    The original field is never modified. Perspective corner coordinates, when
    supplied, refer to the automatically resized original in TL, TR, BR, BL order.
    """
    options = options or PreprocessingOptions()
    original = _decode_image(image_bytes)
    original_size = original.size
    working = _resize(original, options.max_dimension)
    image = cv2.cvtColor(np.asarray(working), cv2.COLOR_RGB2BGR)

    if options.perspective_corners is not None:
        image = _apply_perspective(image, options.perspective_corners)

    if options.enhance_contrast:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            lightness
        )
        image = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )

    if options.reduce_noise:
        image = cv2.bilateralFilter(image, d=5, sigmaColor=25, sigmaSpace=5)

    if options.grayscale or options.threshold:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if options.threshold:
        image = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

    if options.sharpen:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    if image.ndim == 2:
        processed = Image.fromarray(image)
    else:
        processed = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return PreprocessedImage(
        original_bytes=image_bytes,
        original=original.copy(),
        processed=processed,
        original_size=original_size,
        processed_size=processed.size,
    )
