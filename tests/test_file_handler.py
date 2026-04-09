"""
Tests for modules/file_handler.py

Uses in-memory image bytes — no real PDF files required.
"""

import unittest
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from modules.file_handler import detect_file_type, load_image, get_image_for_display


def _make_png_bytes(width=200, height=200) -> bytes:
    """Create minimal valid PNG bytes in memory."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestDetectFileType(unittest.TestCase):

    def test_pdf(self):
        self.assertEqual(detect_file_type("drawing.pdf"), "pdf")

    def test_jpg(self):
        self.assertEqual(detect_file_type("photo.jpg"), "jpg")

    def test_jpeg(self):
        self.assertEqual(detect_file_type("photo.jpeg"), "jpg")

    def test_png(self):
        self.assertEqual(detect_file_type("scan.png"), "png")

    def test_unknown(self):
        self.assertEqual(detect_file_type("file.dwg"), "unknown")

    def test_case_insensitive(self):
        self.assertEqual(detect_file_type("DRAWING.PDF"), "pdf")

    def test_no_extension(self):
        self.assertEqual(detect_file_type("drawing"), "unknown")


class TestLoadImage(unittest.TestCase):

    def test_returns_pil_image(self):
        png_bytes = _make_png_bytes()
        img = load_image(png_bytes)
        self.assertIsInstance(img, Image.Image)

    def test_converts_to_rgb(self):
        png_bytes = _make_png_bytes()
        img = load_image(png_bytes)
        self.assertEqual(img.mode, "RGB")

    def test_correct_dimensions(self):
        png_bytes = _make_png_bytes(320, 240)
        img = load_image(png_bytes)
        self.assertEqual(img.width, 320)
        self.assertEqual(img.height, 240)


class TestGetImageForDisplay(unittest.TestCase):

    def test_small_image_unchanged(self):
        img = Image.new("RGB", (800, 600))
        result = get_image_for_display(img, max_width=1200)
        self.assertEqual(result.width, 800)

    def test_large_image_resized(self):
        img = Image.new("RGB", (2400, 1600))
        result = get_image_for_display(img, max_width=1200)
        self.assertLessEqual(result.width, 1200)

    def test_aspect_ratio_preserved(self):
        img = Image.new("RGB", (2400, 1200))  # 2:1 ratio
        result = get_image_for_display(img, max_width=1200)
        ratio = result.width / result.height
        self.assertAlmostEqual(ratio, 2.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
