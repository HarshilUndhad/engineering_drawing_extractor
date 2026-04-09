"""
Prompt Templates
All LLM prompt strings used by the vision extractor and narrative generator.
Centralised here for easy tuning and review.
"""

# ──────────────────────────────────────────────
# Vision Extraction Prompt
# Sent to the vision LLM along with the drawing image.
# ──────────────────────────────────────────────

VISION_EXTRACTION_PROMPT = """You are an expert civil / highway engineer analysing an NHAI engineering drawing.
Study this drawing VERY carefully. Your job is to READ and COPY text exactly as printed — do NOT guess, paraphrase, or infer.

CRITICAL ACCURACY RULES (read before extracting):
1. Copy ALL text VERBATIM — exact spelling, exact numbers, exact codes
2. Drawing numbers follow the format: NHAI/<STATE>/<PROJECT>/P&P/<SHEET> — find and copy exactly
3. Project name usually appears as "City1-City2 / CSH-N" or "City1-City2 / NH-N" in the title block
4. Every structure MUST include its exact chainage (CH X+XXX) as written on the drawing
5. If you cannot read a value clearly, write "unreadable" — do NOT guess

Return ONLY valid JSON in this exact format:

{
  "General Info": [
    "Project name: <copy exactly from title block>",
    "Authority: <copy exactly>",
    "Consultant: <copy exactly>",
    "Drawing number: <NHAI/XX/XX/P&P/XX — copy exactly>",
    "Sheet: <sheet number if shown>",
    "Date: <copy exactly>",
    "Scale: <copy exactly e.g. 1:2500>",
    "Paper size: <copy exactly e.g. A2>"
  ],
  "Chainage": [
    "Start Chainage: CH <X+XXX — copy exactly>",
    "End Chainage: CH <X+XXX — copy exactly>",
    "Coverage: <start> to <end>"
  ],
  "Structures": [
    "<Structure type> at CH <X+XXX> — <size/type details>",
    "e.g.: Gas Pipe Bridge at CH 0+340",
    "e.g.: Canal Underpass at CH 0+586",
    "e.g.: ROB at CH 0+695.5"
  ],
  "Road Geometry": [
    "Curve: R = <value> m at CH <X+XXX>",
    "Design speed: <value> km/h",
    "Superelevation: <value>%"
  ],
  "Utilities & Features": [
    "Gas pipeline at CH <X+XXX>",
    "Bus bay at CH <X+XXX>",
    "Railway crossing at CH <X+XXX>",
    "Canal at CH <X+XXX>"
  ],
  "Annotations": [
    "BM No. <N> — RL <value>",
    "GPS point at CH <X+XXX>",
    "KM stone at <location>",
    "Land use: <description>"
  ]
}

EXTRACTION CHECKLIST — verify each before returning:
- Title block (bottom strip): project name, drawing number, authority, consultant, date, scale
- Sheet coverage chainages: read start CH and end CH from the chainage bar or notes
- Every symbol on the drawing that represents a structure: include its exact chainage
- Geometry table (if present): read radius, deflection angle, design speed
- Legend (right panel): check for any structure or utility symbols
- Return ONLY valid JSON — no text before or after"""


# ──────────────────────────────────────────────
# Narrative Summary Prompt
# Sent to the LLM with the merged structured data.
# ──────────────────────────────────────────────

NARRATIVE_PROMPT = """You are a senior highway engineer at NHAI briefing a project manager who is not an engineer.
Based on the following structured data extracted from an engineering drawing, write a clear, professional
3-4 paragraph summary that describes the drawing.

The summary should:
1. Open with the project identity — name, authority, consultant, drawing coverage
2. Describe the major structures (culverts, bridges, underpasses) with their locations
3. Cover the road geometry — curves, design speed, any notable alignment features
4. Mention utilities, features, and important annotations
5. Be written in plain English that a non-technical stakeholder can understand
6. Use professional but accessible language — avoid unnecessary jargon
7. Reference specific chainages and dimensions where available

STRUCTURED DATA:
{structured_data}

Write the narrative summary below:"""



