"""
Configuration settings for the Engineering Drawing Intelligence Extractor.
Contains model registry, Ollama connection settings, and extraction categories.
"""

# ──────────────────────────────────────────────
# Ollama Connection
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TIMEOUT = 300  # seconds — vision models can be slow on low-RAM machines

# ──────────────────────────────────────────────
# Model Registry — Vision Models
# Users can select from these in the Streamlit sidebar
# ──────────────────────────────────────────────
AVAILABLE_MODELS = {
    "Qwen2.5-VL 3B (Fast, ~2 GB)": "qwen2.5vl:3b",
    "Qwen2.5-VL 7B (Balanced, ~4.5 GB)": "qwen2.5vl:7b",
    "LLaVA 7B (Accurate, ~4.5 GB)": "llava:7b",
}

DEFAULT_MODEL_LABEL = "LLaVA 7B (Accurate, ~4.5 GB)"

# ──────────────────────────────────────────────
# Model Registry — Narrative Models (text-only, faster)
# ──────────────────────────────────────────────
NARRATIVE_MODELS = {
    "Llama 3.2 3B (Fast, ~2 GB)": "llama3.2:3b",
    "Llama 3 8B (Quality, ~4.5 GB)": "llama3:8b",
}

DEFAULT_NARRATIVE_MODEL = "Llama 3.2 3B (Fast, ~2 GB)"

# ──────────────────────────────────────────────
# Extraction Categories
# ──────────────────────────────────────────────
EXTRACTION_CATEGORIES = [
    "General Info",
    "Chainage",
    "Structures",
    "Road Geometry",
    "Utilities & Features",
    "Annotations",
]

# Category descriptions — used in UI section headers
CATEGORY_DESCRIPTIONS = {
    "General Info": "Project name, authority, consultant, drawing number, date, scale, paper size",
    "Chainage": "Start chainage, end chainage, sheet coverage",
    "Structures": "Box culverts, bridges, ROBs, MNBs, underpasses, SVUPs — with chainage, size, type",
    "Road Geometry": "Curve numbers, radius, deflection angle, design speed, superelevation",
    "Utilities & Features": "Bus bays, gas pipelines, canals, railway crossings, wire heights",
    "Annotations": "Benchmarks, GPS points, TBMs, kilometre stones, land use labels",
}

# ──────────────────────────────────────────────
# File Handling
# ──────────────────────────────────────────────
SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png", "pdf"]
MAX_FILE_SIZE_MB = 50
PDF_DPI = 300  # DPI for PDF → image conversion (300 for dense A2 drawings)
