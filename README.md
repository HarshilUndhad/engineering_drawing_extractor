# Engineering Drawing Intelligence Extractor

AI-powered web application that extracts structured engineering information from drawings and generates professional narrative summaries.

## Architecture

**Hybrid 4-Stage Pipeline** designed for maximum accuracy, robust edge-case handling, and blazing fast data transformation:

1. **High-Fidelity Rendering (`pypdfium2`)**
   - Automatically handles CAD-origin PDF edge-cases (like AutoCAD Civil 3D exports). By using Chrome's PDFium engine internally, it guarantees overlapping vector layers, transparency formats, and Optional Content Group (OCG) hatches render visibly, rather than failing silently as they do in standard rasterizers.
2. **Deterministic Text Extraction (`PyMuPDF` + Heuristics)**
   - Extracts structured and semi-structured metadata securely without LLM hallucinations.
   - Leverages cascading custom RegEx and deduplication heuristic algorithms. Intelligently recovers missing or disassociated bounding boxes produced natively by AutoCAD tables.
3. **Spatial Vision Analysis (Ollama Vision LLM)**
   - Uses localized multi-modal VLMs (e.g., LLaVA, Qwen2.5-VL) prompted with zero-shot spatial layout parameters to identify components text-parsers miss (e.g. mapping legend symbols to map markers, connecting floating tables to lines of geometry).
4. **Smart Deduplication Merge & Narrative Generation**
   - Aggregates Regex and Vision outputs efficiently. 
   - A secondary, highly-optimized text-only LLM (like `Llama 3.2:3b`) synthesizes the unified JSON parameters into a professional 3-paragraph plain English summary for non-technical stakeholders (slashing summary generation time from 3 minutes to ~15 seconds).

## Prerequisites

- **Python 3.9+**
- **Ollama** — [Install from ollama.com](https://ollama.com)

## Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull models via Ollama
```bash
# Vision models — for drawing analysis (pull at least one)
ollama pull llava:7b          # Accurate, ~4.5 GB (default)
ollama pull qwen2.5vl:7b      # Balanced, ~4.5 GB
ollama pull qwen2.5vl:3b      # Fast, ~2 GB (low-RAM laptops)

# Narrative models — for summary generation (pull at least one)
ollama pull llama3.2:3b        # Fast, ~2 GB (default)
ollama pull llama3:8b          # Higher quality, ~4.5 GB
```

### 3. Start Ollama server
```bash
ollama serve
```

### 4. Run the app
```bash
streamlit run app.py
```

## Usage

1. Select your **Vision Model** and **Narrative Model** from the sidebar
2. Upload an engineering drawing (JPG, PNG, or PDF)
3. Preview the drawing in the UI
4. Click **"Extract Intelligence"** to run the analysis
5. View structured output in 6 categorised sections (sortable, searchable tables)
6. Read the AI-generated narrative summary at the bottom
7. Download structured data or narrative as `.txt` files

## Supported Models

### Vision Models (for drawing analysis)

| Model | Ollama Tag | VRAM | Speed | Best For |
|-------|-----------|------|-------|----------|
| Qwen2.5-VL 3B | `qwen2.5vl:3b` | ~2 GB | Fast | Low-RAM laptops |
| Qwen2.5-VL 7B | `qwen2.5vl:7b` | ~4.5 GB | Medium | Balanced |
| LLaVA 7B | `llava:7b` | ~4.5 GB | Medium | High accuracy |

### Narrative Models (for summary generation — text-only)

| Model | Ollama Tag | VRAM | Speed | Best For |
|-------|-----------|------|-------|----------|
| Llama 3.2 3B | `llama3.2:3b` | ~2 GB | Fast (~15s) | Default recommended |
| Llama 3 8B | `llama3:8b` | ~4.5 GB | Medium (~45s) | Higher quality prose |

> **Why separate models?** Vision models (LLaVA, Qwen2-VL) are much slower for text generation than dedicated text models. Using Llama 3.2 3B for narrative reduces summary time from ~3 min to ~15 sec.

## Estimated Processing Times

| Setup | Per page (vision) | Narrative | 3-page PDF total |
|-------|------------------|-----------|-----------------|
| Before (LLaVA for everything) | ~3 min | ~3 min | ~12–15 min |
| After (LLaVA + Llama 3.2 3B) | ~2.5 min | ~15 sec | ~8–9 min |

## Project Structure

```
engineering_drawing_extractor/
├── app.py                  ← Streamlit entry point
├── config.py               ← Model registry & settings
├── requirements.txt
├── README.md
├── modules/
│   ├── file_handler.py     ← Upload routing, PDF→image conversion
│   ├── text_extractor.py   ← PyMuPDF text extraction + regex parsing
│   ├── vision_extractor.py ← Ollama vision LLM API calls
│   ├── merger.py           ← Merges text + vision outputs
│   └── narrator.py         ← Generates narrative summary
├── prompts/
│   └── templates.py        ← All prompt templates
└── tests/
    ├── test_text_extractor.py
    ├── test_merger.py
    ├── test_vision_extractor.py
    ├── test_narrator.py
    └── test_file_handler.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests use `unittest.mock` to mock Ollama API calls — **no Ollama server required** to run the test suite.

## Sample Drawing

Tested against NHAI Plan & Longitudinal Profile — Maliya to Pipaliya (CH: 0+000 to CH: 3+000).
