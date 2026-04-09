"""
Vision Extractor Module
Uses Ollama vision LLM (LLaVA / Qwen2-VL) to analyse engineering drawings visually.
Sends the drawing image to the model and receives structured JSON output.

This complements the text extractor by capturing spatial relationships,
visual layout, and any details that text extraction might miss.
"""

import base64
import json
import re
import io
import requests
from typing import Dict, List, Optional
from PIL import Image

from config import OLLAMA_BASE_URL, OLLAMA_GENERATE_URL, OLLAMA_TIMEOUT, EXTRACTION_CATEGORIES
from prompts.templates import VISION_EXTRACTION_PROMPT


def image_to_base64(image: Image.Image, max_size: int = 1920) -> str:
    """
    Convert a PIL Image to a base64-encoded string.
    Resizes if needed to stay within model input limits.
    
    Args:
        image:    PIL Image to convert.
        max_size: Maximum dimension (width or height) in pixels.
        
    Returns:
        Base64-encoded JPEG string.
    """
    # Resize large images to avoid exceeding model context
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_with_vision(image: Image.Image, model_tag: str) -> Dict[str, List[str]]:
    """
    Send drawing image to Ollama vision model and extract structured data.
    
    Args:
        image:     PIL Image of the engineering drawing (one page).
        model_tag: Ollama model tag (e.g. 'llava:7b', 'qwen2-vl:2b').
        
    Returns:
        Dict with category keys and lists of extracted items.
        Returns empty categories on failure.
    """
    img_b64 = image_to_base64(image)
    
    payload = {
        "model": model_tag,
        "prompt": VISION_EXTRACTION_PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,      # Low temperature for factual extraction
            "num_predict": 4096,      # Allow long responses
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        
        # 500 most commonly means Ollama ran out of VRAM with the full-size image.
        # Retry once at 1280px before giving up.
        if response.status_code == 500:
            img_b64_small = image_to_base64(image, max_size=1280)
            payload["images"] = [img_b64_small]
            response = requests.post(
                OLLAMA_GENERATE_URL,
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
        
        if response.status_code == 503:
            return _empty_result(
                "Ollama is still loading the model — wait a few seconds and try again"
            )
        
        response.raise_for_status()
        
        result = response.json()
        raw_text = result.get("response", "")
        
        return parse_vision_response(raw_text)
        
    except requests.exceptions.ConnectionError:
        return _empty_result("Ollama not running — start with 'ollama serve'")
    except requests.exceptions.Timeout:
        return _empty_result("Vision model timed out — try a smaller model (Qwen2.5-VL 3B)")
    except requests.exceptions.HTTPError as e:
        body = e.response.text[:300] if e.response is not None else ""
        return _empty_result(
            f"Ollama HTTP {e.response.status_code if e.response is not None else '?'} error — "
            f"likely out of VRAM. Try a smaller vision model (Qwen2.5-VL 3B). "
            f"Ollama says: {body}"
        )
    except Exception as e:
        return _empty_result(f"Vision extraction error: {str(e)}")


def parse_vision_response(raw_text: str) -> Dict[str, List[str]]:
    """
    Parse the LLM's raw text response into a structured dict.
    Handles cases where the model wraps JSON in markdown code blocks.
    
    Args:
        raw_text: Raw text response from the LLM.
        
    Returns:
        Parsed dict with category keys and item lists.
    """
    # Try to extract JSON from the response
    # Handle markdown code blocks: ```json ... ```
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return _empty_result("Could not parse JSON from vision model response")
    
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        json_str = json_str.replace("'", '"')  # single to double quotes
        json_str = re.sub(r",\s*}", "}", json_str)  # trailing commas
        json_str = re.sub(r",\s*]", "]", json_str)  # trailing commas in arrays
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return _empty_result("Vision model returned invalid JSON")
    
    # Normalize to expected category keys
    result = {}
    for category in EXTRACTION_CATEGORIES:
        items = parsed.get(category, [])
        if isinstance(items, list):
            result[category] = [str(item) for item in items if item]
        elif isinstance(items, str):
            result[category] = [items] if items else []
        else:
            result[category] = []
    
    return result


def check_ollama_connection(model_tag: str) -> dict:
    """
    Check if Ollama is running and the requested model is available.
    
    Args:
        model_tag: Model tag to check (e.g. 'llava:7b').
        
    Returns:
        Dict with 'ok' (bool), 'message' (str), and 'models' (list).
    """
    try:
        # Check if Ollama is running
        resp = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        
        available_models = [m["name"] for m in data.get("models", [])]
        
        # Check if requested model is pulled
        model_found = any(
            model_tag in m or model_tag.split(":")[0] in m
            for m in available_models
        )
        
        if model_found:
            return {
                "ok": True,
                "message": f"✅ Ollama running, model '{model_tag}' is available.",
                "models": available_models,
            }
        else:
            return {
                "ok": False,
                "message": (
                    f"⚠️ Ollama is running but model '{model_tag}' is not pulled.\n"
                    f"Run: `ollama pull {model_tag}`\n"
                    f"Available models: {', '.join(available_models) or 'none'}"
                ),
                "models": available_models,
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "message": "❌ Cannot connect to Ollama. Start it with: `ollama serve`",
            "models": [],
        }
    except Exception as e:
        return {
            "ok": False,
            "message": f"❌ Error checking Ollama: {str(e)}",
            "models": [],
        }


def _empty_result(error_msg: str = "") -> Dict[str, List[str]]:
    """Return an empty result dict with an optional error message."""
    result = {cat: [] for cat in EXTRACTION_CATEGORIES}
    if error_msg:
        result["_error"] = error_msg
    return result
