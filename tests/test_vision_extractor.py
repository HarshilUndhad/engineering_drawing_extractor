"""
Tests for modules/vision_extractor.py

Ollama API calls are mocked with unittest.mock — no Ollama required.
"""

import unittest
import base64
import sys
import os
from unittest.mock import patch, MagicMock
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.vision_extractor import (
    image_to_base64,
    parse_vision_response,
    _empty_result,
)


class TestImageToBase64(unittest.TestCase):

    def _make_image(self, width, height):
        return Image.new("RGB", (width, height), color=(128, 128, 128))

    def test_small_image_not_resized(self):
        img = self._make_image(800, 600)
        b64 = image_to_base64(img, max_size=1920)
        # Should be valid base64
        decoded = base64.b64decode(b64)
        result_img = Image.open(io.BytesIO(decoded))
        self.assertEqual(result_img.width, 800)
        self.assertEqual(result_img.height, 600)

    def test_large_image_resized_to_1920(self):
        img = self._make_image(3000, 2000)
        b64 = image_to_base64(img, max_size=1920)
        decoded = base64.b64decode(b64)
        result_img = Image.open(io.BytesIO(decoded))
        self.assertLessEqual(result_img.width, 1920)
        self.assertLessEqual(result_img.height, 1920)

    def test_returns_string(self):
        img = self._make_image(100, 100)
        result = image_to_base64(img)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestParseVisionResponse(unittest.TestCase):

    VALID_JSON = """{
        "General Info": ["Project: Test Road"],
        "Chainage": ["Start: CH 0+000"],
        "Structures": [],
        "Road Geometry": [],
        "Utilities & Features": [],
        "Annotations": []
    }"""

    def test_valid_json_parsed(self):
        result = parse_vision_response(self.VALID_JSON)
        self.assertIn("General Info", result)
        self.assertEqual(result["General Info"], ["Project: Test Road"])

    def test_markdown_wrapped_json(self):
        wrapped = f"```json\n{self.VALID_JSON}\n```"
        result = parse_vision_response(wrapped)
        self.assertEqual(result["Chainage"], ["Start: CH 0+000"])

    def test_trailing_comma_fixed(self):
        bad_json = """{
            "General Info": ["item1",],
            "Chainage": [],
            "Structures": [],
            "Road Geometry": [],
            "Utilities & Features": [],
            "Annotations": [],
        }"""
        result = parse_vision_response(bad_json)
        # Should not raise — trailing commas are cleaned
        self.assertIn("General Info", result)

    def test_empty_response_returns_empty(self):
        result = parse_vision_response("no JSON here at all")
        for cat in result:
            if cat != "_error":
                self.assertEqual(result[cat], [])

    def test_unknown_categories_ignored(self):
        json_with_extra = """{
            "General Info": ["info"],
            "Chainage": [],
            "Structures": [],
            "Road Geometry": [],
            "Utilities & Features": [],
            "Annotations": [],
            "Unknown Category": ["ignored"]
        }"""
        result = parse_vision_response(json_with_extra)
        self.assertNotIn("Unknown Category", result)


class TestEmptyResult(unittest.TestCase):

    def test_empty_result_has_all_categories(self):
        result = _empty_result()
        from config import EXTRACTION_CATEGORIES
        for cat in EXTRACTION_CATEGORIES:
            self.assertIn(cat, result)
            self.assertEqual(result[cat], [])

    def test_error_message_stored(self):
        result = _empty_result("Connection refused")
        self.assertEqual(result["_error"], "Connection refused")


class TestExtractWithVision(unittest.TestCase):
    """Tests for the Ollama-calling function — mocked with unittest.mock."""

    @patch("modules.vision_extractor.requests.post")
    def test_successful_extraction(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '{"General Info": ["Authority: NHAI"], "Chainage": [], "Structures": [], "Road Geometry": [], "Utilities & Features": [], "Annotations": []}'
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        from modules.vision_extractor import extract_with_vision
        img = Image.new("RGB", (100, 100))
        result = extract_with_vision(img, "llava:7b")
        self.assertEqual(result["General Info"], ["Authority: NHAI"])

    @patch("modules.vision_extractor.requests.post", side_effect=Exception("timeout"))
    def test_extraction_error_returns_empty(self, mock_post):
        from modules.vision_extractor import extract_with_vision
        img = Image.new("RGB", (100, 100))
        result = extract_with_vision(img, "llava:7b")
        self.assertIn("_error", result)


if __name__ == "__main__":
    unittest.main()
