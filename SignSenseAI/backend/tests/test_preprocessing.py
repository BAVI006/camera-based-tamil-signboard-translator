import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from backend.services.preprocessing import (
    MAX_UPLOAD_BYTES,
    ImageValidationError,
    PreprocessingOptions,
    preprocess_image,
)


def encode_image(image: Image.Image, image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


class PreprocessingTests(unittest.TestCase):
    def test_rejects_empty_and_non_image_uploads(self):
        for invalid in (b"", b"this is not an image"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ImageValidationError):
                    preprocess_image(invalid)

    def test_rejects_unsupported_and_oversized_uploads(self):
        bitmap = encode_image(Image.new("RGB", (2, 2), "white"), "BMP")
        with self.assertRaisesRegex(ImageValidationError, "Unsupported image format"):
            preprocess_image(bitmap)

        with self.assertRaisesRegex(ImageValidationError, "15 MB"):
            preprocess_image(b"x" * (MAX_UPLOAD_BYTES + 1))

    def test_preserves_oriented_original_and_makes_independent_processed_image(self):
        source = Image.new("RGB", (90, 60), (170, 180, 190))
        source_bytes = encode_image(source)
        result = preprocess_image(source_bytes)

        self.assertEqual(result.original_bytes, source_bytes)
        self.assertEqual(result.original_size, (90, 60))
        self.assertEqual(result.processed_size, (90, 60))
        self.assertEqual(result.original.getpixel((0, 0)), (170, 180, 190))
        self.assertIsNot(result.original, result.processed)

    def test_resizes_without_changing_aspect_ratio(self):
        source = Image.new("RGB", (400, 200), "white")
        result = preprocess_image(
            encode_image(source),
            PreprocessingOptions(max_dimension=100),
        )

        self.assertEqual(result.processed_size, (100, 50))
        self.assertEqual(result.original_size, (400, 200))

    def test_exif_orientation_is_corrected(self):
        source = Image.new("RGB", (80, 40), "white")
        exif = Image.Exif()
        exif[274] = 6
        buffer = BytesIO()
        source.save(buffer, format="JPEG", exif=exif)

        result = preprocess_image(buffer.getvalue())

        self.assertEqual(result.original_size, (40, 80))
        self.assertEqual(result.processed_size, (40, 80))

    def test_optional_grayscale_threshold_and_sharpen_are_supported(self):
        pixels = np.zeros((80, 120, 3), dtype=np.uint8)
        pixels[:, 60:] = 255
        source = Image.fromarray(pixels, mode="RGB")

        gray_result = preprocess_image(
            encode_image(source),
            PreprocessingOptions(
                enhance_contrast=False,
                reduce_noise=False,
                grayscale=True,
                sharpen=True,
            ),
        )
        threshold_result = preprocess_image(
            encode_image(source),
            PreprocessingOptions(
                enhance_contrast=False,
                reduce_noise=False,
                threshold=True,
            ),
        )

        self.assertEqual(gray_result.processed.mode, "L")
        self.assertEqual(threshold_result.processed.mode, "L")
        self.assertEqual(set(threshold_result.processed.getdata()), {0, 255})

    def test_perspective_correction_rejects_points_outside_image(self):
        source = Image.new("RGB", (100, 100), "white")
        options = PreprocessingOptions(
            enhance_contrast=False,
            reduce_noise=False,
            perspective_corners=(
                (0, 0),
                (100, 0),
                (100, 99),
                (0, 99),
            ),
        )

        with self.assertRaisesRegex(ValueError, "inside the resized image"):
            preprocess_image(encode_image(source), options)

    def test_perspective_correction_rectifies_selected_region(self):
        source = Image.new("RGB", (100, 100), "white")
        options = PreprocessingOptions(
            enhance_contrast=False,
            reduce_noise=False,
            perspective_corners=((5, 5), (94, 5), (94, 94), (5, 94)),
        )

        result = preprocess_image(encode_image(source), options)

        self.assertEqual(result.processed_size, (89, 89))


if __name__ == "__main__":
    unittest.main()
