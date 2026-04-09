"""
Merger Module
Combines outputs from the text extractor (PyMuPDF) and vision extractor (LLM).

Merge strategy:
- Text extractor data is PREFERRED where both sources have data (higher accuracy)
- Vision extractor fills GAPS where text extraction found nothing
- Duplicate items are removed
"""

from typing import Dict, List
from config import EXTRACTION_CATEGORIES


def merge_extractions(
    text_data: Dict[str, List[str]],
    vision_data: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Merge text-extracted and vision-extracted data into a unified output.
    
    Priority: text_data > vision_data (text extraction is more accurate for PDFs).
    Vision data supplements where text extraction has gaps.
    
    Args:
        text_data:   Structured data from PyMuPDF text extraction.
        vision_data: Structured data from Ollama vision model.
        
    Returns:
        Merged dict with deduplicated items per category.
    """
    merged = {}
    
    for category in EXTRACTION_CATEGORIES:
        text_items = text_data.get(category, [])
        vision_items = vision_data.get(category, [])
        
        if text_items and vision_items:
            # Both have data — start with text (more accurate), add unique vision items
            merged[category] = _merge_lists(text_items, vision_items)
        elif text_items:
            # Only text data available
            merged[category] = text_items
        elif vision_items:
            # Only vision data available
            merged[category] = vision_items
        else:
            # Neither source found data
            merged[category] = []
    
    return merged


def _merge_lists(primary: List[str], secondary: List[str]) -> List[str]:
    """
    Merge two lists, keeping all primary items and adding non-duplicate secondary items.
    Uses fuzzy matching to detect duplicates (case-insensitive substring check).
    
    Args:
        primary:   Higher priority items (from text extraction).
        secondary: Lower priority items (from vision extraction).
        
    Returns:
        Combined list with duplicates removed.
    """
    result = list(primary)  # Keep all primary items
    
    # Normalise primary items for comparison
    primary_normalised = [_normalise(item) for item in primary]
    
    for item in secondary:
        item_norm = _normalise(item)
        
        # Check if this item is a duplicate of any primary item
        is_duplicate = any(
            _is_similar(item_norm, p_norm)
            for p_norm in primary_normalised
        )
        
        if not is_duplicate:
            result.append(item)
    
    return result


def _normalise(text: str) -> str:
    """Lowercase, strip whitespace and punctuation for comparison."""
    return text.lower().strip().replace(":", "").replace("-", "").replace("_", "")


def _is_similar(a: str, b: str) -> bool:
    """
    Check if two normalised strings are similar enough to be duplicates.

    Substring check is guarded by a length + ratio test to prevent false
    positives like "r  200 m" being deduplicated against "r  2000 m".
    """
    if not a or not b:
        return False
    # Exact match
    if a == b:
        return True
    # Substring match — only deduplicate when both strings are substantial
    # and close in length (ratio > 0.7 prevents "200" matching "2000")
    if (a in b or b in a) and min(len(a), len(b)) > 6:
        ratio = min(len(a), len(b)) / max(len(a), len(b))
        return ratio > 0.7
    # Check significant overlap (shared key numbers/values)
    a_nums = set(a.split())
    b_nums = set(b.split())
    if len(a_nums) > 2 and len(b_nums) > 2:
        overlap = a_nums & b_nums
        if len(overlap) >= min(len(a_nums), len(b_nums)) * 0.6:
            return True
    return False


def get_extraction_summary(merged_data: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Get a count summary of extracted items per category.
    
    Args:
        merged_data: Merged extraction results.
        
    Returns:
        Dict mapping category name to item count.
    """
    return {
        category: len(items)
        for category, items in merged_data.items()
        if category in EXTRACTION_CATEGORIES
    }
