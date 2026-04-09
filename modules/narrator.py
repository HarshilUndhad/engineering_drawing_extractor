"""
Narrator Module
Generates a professional plain-English narrative summary of an engineering drawing
using the merged structured data and an LLM call via Ollama.

The narrative reads as if a senior engineer is briefing a project manager.
"""

import json
import requests
from typing import Dict, List

from config import OLLAMA_GENERATE_URL, OLLAMA_TIMEOUT, EXTRACTION_CATEGORIES
from prompts.templates import NARRATIVE_PROMPT


def generate_narrative(
    merged_data: Dict[str, List[str]],
    narrative_model_tag: str,
) -> str:
    """
    Generate a professional narrative summary from the merged extraction data.
    
    Uses a text-only LLM (e.g. llama3.2:3b) to produce a 3-4 paragraph summary
    written as if a senior NHAI engineer is briefing a non-technical stakeholder.
    
    Args:
        merged_data:          Merged structured data from text + vision extraction.
        narrative_model_tag:  Ollama model tag for narrative generation (text-only preferred).
        
    Returns:
        Narrative summary string. Returns error message on failure.
    """
    # Format the structured data as readable text for the prompt
    structured_text = _format_data_for_prompt(merged_data)
    
    prompt = NARRATIVE_PROMPT.format(structured_data=structured_text)
    
    payload = {
        "model": narrative_model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,    # Slightly creative but still factual
            "num_predict": 2048,
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        
        result = response.json()
        narrative = result.get("response", "").strip()
        
        if not narrative:
            return _fallback_narrative(merged_data)
        
        return narrative
        
    except requests.exceptions.ConnectionError:
        return "⚠️ Could not generate narrative — Ollama is not running. Start with: `ollama serve`"
    except requests.exceptions.Timeout:
        return "⚠️ Narrative generation timed out. Try using a smaller model."
    except Exception as e:
        return f"⚠️ Narrative generation error: {str(e)}"


def _format_data_for_prompt(data: Dict[str, List[str]]) -> str:
    """Format the merged data as readable text sections for the LLM prompt."""
    sections = []
    for category in EXTRACTION_CATEGORIES:
        items = data.get(category, [])
        if items:
            section = f"\n### {category}\n"
            for item in items:
                section += f"- {item}\n"
            sections.append(section)
    
    return "\n".join(sections) if sections else "No data was extracted from the drawing."


def _fallback_narrative(data: Dict[str, List[str]]) -> str:
    """
    Generate a basic narrative without LLM if the model fails.
    Uses template-based approach with the available data.
    """
    parts = []
    
    # General Info
    general = data.get("General Info", [])
    if general:
        parts.append(
            "This engineering drawing pertains to " 
            + ". ".join(general[:3]) + "."
        )
    
    # Chainage
    chainage = data.get("Chainage", [])
    if chainage:
        parts.append(
            "The drawing covers " + ", ".join(chainage[:3]) + "."
        )
    
    # Structures
    structures = data.get("Structures", [])
    if structures:
        parts.append(
            f"The drawing shows {len(structures)} structure(s) including: "
            + "; ".join(structures[:5]) + "."
        )
    
    # Road Geometry
    geometry = data.get("Road Geometry", [])
    if geometry:
        parts.append(
            "Road geometry details include: " + ", ".join(geometry[:4]) + "."
        )
    
    # Utilities
    utilities = data.get("Utilities & Features", [])
    if utilities:
        parts.append(
            "Notable utilities and features: " + ", ".join(utilities[:4]) + "."
        )
    
    if not parts:
        return "No sufficient data could be extracted to generate a narrative summary."
    
    return " ".join(parts)
