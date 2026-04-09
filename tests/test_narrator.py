"""
Tests for modules/narrator.py

Ollama API calls are mocked with unittest.mock — no Ollama required.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.narrator import _format_data_for_prompt, _fallback_narrative, generate_narrative


SAMPLE_DATA = {
    "General Info": ["Authority: NHAI", "Drawing Number: NHAI/GL/001"],
    "Chainage": ["Start Chainage: CH 0+000", "End Chainage: CH 3+000"],
    "Structures": ["Box Culvert 2x1m at CH 1+200", "ROB at CH 2+500"],
    "Road Geometry": ["Design Speed: 100 km/h", "Radius: R = 600 m"],
    "Utilities & Features": ["Gas Pipeline at CH 1+400"],
    "Annotations": ["BM No. 5 RL 123.45"],
}

EMPTY_DATA = {cat: [] for cat in SAMPLE_DATA}


class TestFormatDataForPrompt(unittest.TestCase):

    def test_non_empty_data_has_headings(self):
        result = _format_data_for_prompt(SAMPLE_DATA)
        self.assertIn("### General Info", result)
        self.assertIn("### Chainage", result)
        self.assertIn("Authority: NHAI", result)

    def test_empty_categories_omitted(self):
        data = {**EMPTY_DATA, "General Info": ["Authority: NHAI"]}
        result = _format_data_for_prompt(data)
        self.assertNotIn("### Chainage", result)
        self.assertNotIn("### Structures", result)

    def test_fully_empty_data(self):
        result = _format_data_for_prompt(EMPTY_DATA)
        self.assertIn("No data", result)


class TestFallbackNarrative(unittest.TestCase):

    def test_returns_string(self):
        result = _fallback_narrative(SAMPLE_DATA)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20)

    def test_mentions_structures(self):
        result = _fallback_narrative(SAMPLE_DATA)
        self.assertIn("structure", result.lower())

    def test_empty_data_returns_no_data_message(self):
        result = _fallback_narrative(EMPTY_DATA)
        self.assertIn("No sufficient data", result)


class TestGenerateNarrative(unittest.TestCase):

    @patch("modules.narrator.requests.post")
    def test_uses_narrative_model_tag(self, mock_post):
        """Verify the correct (narrative) model tag is sent in the payload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "This drawing covers..."}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        generate_narrative(SAMPLE_DATA, narrative_model_tag="llama3.2:3b")

        call_payload = mock_post.call_args[1]["json"]
        self.assertEqual(call_payload["model"], "llama3.2:3b")

    @patch("modules.narrator.requests.post")
    def test_returns_narrative_text(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Project overview narrative text."}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = generate_narrative(SAMPLE_DATA, narrative_model_tag="llama3.2:3b")
        self.assertEqual(result, "Project overview narrative text.")

    @patch("modules.narrator.requests.post")
    def test_empty_response_triggers_fallback(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = generate_narrative(SAMPLE_DATA, narrative_model_tag="llama3.2:3b")
        # Should return fallback text, not empty string
        self.assertGreater(len(result), 0)

    @patch("modules.narrator.requests.post", side_effect=Exception("connection error"))
    def test_connection_error_returns_warning(self, mock_post):
        result = generate_narrative(SAMPLE_DATA, narrative_model_tag="llama3.2:3b")
        self.assertIn("⚠️", result)


if __name__ == "__main__":
    unittest.main()
