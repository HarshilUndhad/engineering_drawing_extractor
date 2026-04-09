"""
Tests for modules/merger.py

Covers merge priority logic and the _is_similar deduplication fix.
No external dependencies required.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.merger import merge_extractions, get_extraction_summary, _normalise, _is_similar


class TestIsSimilar(unittest.TestCase):

    def test_exact_match(self):
        self.assertTrue(_is_similar("radius r  600 m", "radius r  600 m"))

    def test_genuine_substring_match(self):
        """A clear substring should still be caught when strings are close in length."""
        a = _normalise("Design Speed: 100 km/h")
        b = _normalise("Design Speed 100 km/h")
        self.assertTrue(_is_similar(a, b))

    def test_false_dedup_short_substring_numbers(self):
        """Regression: normalised short strings '200' and '2000' must NOT be deduplicated
        via substring match (the original bug before the length guard was added)."""
        # These represent raw numeric strings that could appear as values
        self.assertFalse(_is_similar("200", "2000"))
        self.assertFalse(_is_similar("r  200", "r  2000"))

    def test_false_dedup_short_numbers(self):
        """Short strings like '10' and '100' should not match."""
        self.assertFalse(_is_similar("10", "100"))

    def test_empty_strings(self):
        self.assertFalse(_is_similar("", "anything"))
        self.assertFalse(_is_similar("anything", ""))

    def test_numeric_overlap_match(self):
        """Strings sharing most tokens should still be caught as duplicates."""
        a = _normalise("Box Culvert 2x1m CH 1+200 LHS")
        b = _normalise("culvert 2x1m at ch 1+200 lhs reinforced")
        # Significant word overlap — should be similar
        self.assertTrue(_is_similar(a, b))


class TestMergeExtractions(unittest.TestCase):

    def setUp(self):
        self.empty = {
            "General Info": [], "Chainage": [], "Structures": [],
            "Road Geometry": [], "Utilities & Features": [], "Annotations": [],
        }

    def test_text_preferred_over_vision(self):
        text_data = {**self.empty, "General Info": ["Authority: NHAI"]}
        vision_data = {**self.empty, "General Info": ["Project Authority: NHAI (vision)"]}
        merged = merge_extractions(text_data, vision_data)
        # Text item should appear first
        self.assertEqual(merged["General Info"][0], "Authority: NHAI")

    def test_vision_fills_gap(self):
        text_data = {**self.empty}  # empty text extraction
        vision_data = {**self.empty, "Structures": ["Bridge at CH 1+500"]}
        merged = merge_extractions(text_data, vision_data)
        self.assertIn("Bridge at CH 1+500", merged["Structures"])

    def test_duplicate_not_added_from_vision(self):
        text_data = {**self.empty, "Chainage": ["Start Chainage: CH 0+000"]}
        vision_data = {**self.empty, "Chainage": ["start chainage ch 0+000"]}
        merged = merge_extractions(text_data, vision_data)
        # Should not double-count
        chainage_items = merged["Chainage"]
        self.assertEqual(len(chainage_items), 1)

    def test_unique_vision_item_added(self):
        text_data = {**self.empty, "Annotations": ["Benchmark: BM No. 5"]}
        vision_data = {**self.empty, "Annotations": ["GPS Point at CH 1+200"]}
        merged = merge_extractions(text_data, vision_data)
        self.assertEqual(len(merged["Annotations"]), 2)

    def test_both_empty(self):
        merged = merge_extractions(self.empty, self.empty)
        for cat in merged:
            self.assertEqual(merged[cat], [])


class TestGetExtractionSummary(unittest.TestCase):

    def test_counts_correct(self):
        data = {
            "General Info": ["a", "b"],
            "Chainage": ["c"],
            "Structures": [],
            "Road Geometry": ["d", "e", "f"],
            "Utilities & Features": [],
            "Annotations": ["g"],
        }
        summary = get_extraction_summary(data)
        self.assertEqual(summary["General Info"], 2)
        self.assertEqual(summary["Chainage"], 1)
        self.assertEqual(summary["Structures"], 0)
        self.assertEqual(summary["Road Geometry"], 3)


if __name__ == "__main__":
    unittest.main()
