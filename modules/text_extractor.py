"""
Text Extractor Module
Uses PyMuPDF (fitz) to extract text directly from PDF engineering drawings.
Parses the raw text into 6 structured categories using regex and keyword matching.

This is the HIGH-ACCURACY extractor — text-based PDFs yield perfect text extraction,
which is far more reliable than vision-based OCR for dense engineering annotations.
"""

import re
from typing import Dict, List, Optional
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_bytes: bytes) -> List[str]:
    """
    Extract all text from each page of a PDF.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    doc.close()
    return pages_text




def parse_structured_data(pdf_bytes: bytes) -> Dict[str, List[str]]:
    """
    Main entry point: extract text from PDF and parse into 6 categories.
    
    """
    pages_text = extract_text_from_pdf(pdf_bytes)
    full_text = "\n".join(pages_text)
    
    result = {
        "General Info": extract_general_info(full_text),
        "Chainage": extract_chainage(full_text),
        "Structures": extract_structures(full_text),
        "Road Geometry": extract_road_geometry(full_text),
        "Utilities & Features": extract_utilities(full_text),
        "Annotations": extract_annotations(full_text),
    }
    
    return result


# ──────────────────────────────────────────────
# Category-specific extraction functions
# ──────────────────────────────────────────────

def extract_general_info(text: str) -> List[str]:
    """Extract project name, authority, consultant, drawing number, date, scale, paper size."""
    items = []
    
    # Authority / Organization
    authority_patterns = [
        r"(?:NATIONAL\s+HIGHWAYS?\s+AUTHORITY\s+OF\s+INDIA|NHAI)",
        r"(?:MINISTRY\s+OF\s+ROAD\s+TRANSPORT)",
        r"(?:PUBLIC\s+WORKS\s+DEPARTMENT|PWD)",
    ]
    for pat in authority_patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            items.append(f"Authority: {match.group(0).strip()}")
            break
    
    # Consultant — collect all matches, then keep only the longest unique ones
    # (avoids "s & Technocrats" fragment when "Intercontinental Consultants & Technocrats" is also present)
    consultant_names = []
    consultant_match = re.search(
        r"(?:Consultant|Consultancy)[:\s]*([^\n]{5,100})", text, re.IGNORECASE
    )
    if consultant_match:
        consultant_names.append(consultant_match.group(1).strip(" \t\n\r,."))
    
    if "Intercontinental" in text:
        icl_match = re.search(r"Intercontinental[^\n]*(?:Ltd|Pvt)[.\s]*(?:Ltd)?[.\s]*", text, re.IGNORECASE)
        if icl_match:
            consultant_names.append(icl_match.group(0).strip(" \t\n\r,."))
    
    # Deduplicate: remove any name that is a substring of a longer name in the list
    consultant_names_deduped = []
    for name in consultant_names:
        name_clean = name.lower().replace(" ", "")
        if not any(
            name_clean != other.lower().replace(" ", "") and name_clean in other.lower().replace(" ", "")
            for other in consultant_names
        ):
            consultant_names_deduped.append(name)
    for name in consultant_names_deduped:
        # Preserve order, remove exact duplicates
        if name not in [i.split(": ", 1)[1] for i in items if i.startswith("Consultant: ")]:
            items.append(f"Consultant: {name}")
    
    # Project Title
    title_match = re.search(
        r"(?:Project\s+Title|Project\s+Name)[:\s]*([A-Za-z][^\n]{4,200})", text, re.IGNORECASE
    )
    if title_match:
        items.append(f"Project Title: {title_match.group(1).strip()}")
    else:
        # Fallback for disconnected text blocks typical of AutoCAD PDF exports
        # Look for standard NHAI project preamble text, cleaning up newlines
        clean_text = text.replace("\n", " ")
        desc_match = re.search(
            r"(Consultancy\s+services.*?State\s+of\s+[A-Za-z]+|Up-?gradation.*?Highway)", 
            clean_text, re.IGNORECASE
        )
        if desc_match:
            items.append(f"Project Title: {desc_match.group(1).strip()}")
        else:
            # Highway project name — pattern: "City1-City2 / CSH-N" or "City1-City2 / NH-N"
            highway_match = re.search(
                r"([A-Za-z]+(?:[\s-][A-Za-z]+)+)\s*/\s*(CSH|NH|SH|MDR|NE)-?\s*(\d+[A-Z]?)",
                text, re.IGNORECASE
            )
            if highway_match:
                project_name = f"{highway_match.group(1).strip()} / {highway_match.group(2).upper()}-{highway_match.group(3)}"
                items.append(f"Project Title: {project_name}")
    
    # Drawing Name / Title — must contain at least one letter (rejects ":-" separators)
    drawing_match = re.search(
        r"(?:Drawing\s+(?:Name|Title))[:\s]*([^\n]{3,200})", text, re.IGNORECASE
    )
    if drawing_match:
        val = drawing_match.group(1).strip()
        if re.search(r"[A-Za-z]", val):  # reject pure punctuation/separator matches
            items.append(f"Drawing Name: {val}")
    
    # Drawing Number — with label prefix
    # Must contain BOTH letters and digits to reject plain words like "Revisions"
    drg_num_match = re.search(
        r"(?:DRAWING\s+NUMBER|DRG\.?\s*NO\.?|Drawing\s+No\.?)[:\s\-]*([A-Z0-9/\-_]+(?:/[A-Z0-9/\-_]+)*)",
        text, re.IGNORECASE
    )
    if drg_num_match:
        val = drg_num_match.group(1).strip()
        # Must look like a code: contains a digit AND a slash or hyphen (e.g. "01" alone is not enough)
        if re.search(r"\d", val) and re.search(r"[A-Za-z/\-]", val):
            items.append(f"Drawing Number: {val}")
            drg_num_match = drg_num_match  # keep as truthy
        else:
            drg_num_match = None  # reset so NHAI fallback runs

    # Drawing Number — standalone NHAI code (e.g. NHAI/GUJ/MAL-PIP/P&P/01)
    if not drg_num_match:
        nhai_num_match = re.search(
            r"\bNHAI/[A-Z]{2,5}/[A-Z0-9\-]{3,}/P&P/\d+\b",
            text, re.IGNORECASE
        )
        if nhai_num_match:
            items.append(f"Drawing Number: {nhai_num_match.group(0).strip()}")
    
    # Scale — deduplicate across repeated labels (same scale mentioned on every page)
    scale_matches = re.findall(
        r"(?:Scale|SCALE)[:\s]*(?:=\s*)?(\d+\s*:\s*\d+)", text, re.IGNORECASE
    )
    seen_scales = set()
    for s in scale_matches:
        s_norm = s.replace(" ", "")
        if s_norm not in seen_scales:
            seen_scales.add(s_norm)
            items.append(f"Scale: {s.strip()}")
    
    # Paper Size
    paper_match = re.search(
        r"(?:Paper\s+Size|PAPER\s+SIZE)[:\s\-]*([A-Z0-9]+\s*(?:Sheet|SHEET)?)", text, re.IGNORECASE
    )
    if paper_match:
        items.append(f"Paper Size: {paper_match.group(1).strip()}")
    
    # Date
    date_patterns = [
        r"(?:Date|DATE)[:\s]*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})",
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{4}",
    ]
    for pat in date_patterns:
        d_match = re.search(pat, text, re.IGNORECASE)
        if d_match:
            items.append(f"Date: {d_match.group(0).strip()}")
            break
    
    # Revision
    rev_match = re.search(r"\b(?:Rev|REV|Revision)\b[.\s:]*([A-Za-z0-9]+)", text, re.IGNORECASE)
    if rev_match and rev_match.group(1).lower() not in ["isions", "iews", "date"]:
        items.append(f"Revision: {rev_match.group(1).strip()}")
    
    return items


def extract_chainage(text: str) -> List[str]:
    """Extract chainage information — start, end, coverage ranges."""
    items = []
    
    # Match CH: X+XXX patterns
    chainage_pattern = r"(?:CH|Ch|Chainage|KM)[:\s.]*(\d+\s*\+\s*\d+)"
    all_chainages = re.findall(chainage_pattern, text, re.IGNORECASE)
    
    if all_chainages:
        # Normalize and deduplicate
        unique_chainages = list(dict.fromkeys(
            ch.replace(" ", "") for ch in all_chainages
        ))
        
        if len(unique_chainages) >= 2:
            items.append(f"Start Chainage: CH {unique_chainages[0]}")
            items.append(f"End Chainage: CH {unique_chainages[-1]}")
        elif len(unique_chainages) == 1:
            items.append(f"Chainage: CH {unique_chainages[0]}")
    
    # Coverage range (e.g., "CH: 0+000 To CH: 1+000")
    coverage_match = re.search(
        r"CH[:\s]*(\d+\+\d+)\s*(?:To|to|TO|-)\s*CH[:\s]*(\d+\+\d+)",
        text, re.IGNORECASE
    )
    if coverage_match:
        items.append(
            f"Sheet Coverage: CH {coverage_match.group(1)} to CH {coverage_match.group(2)}"
        )
    
    # Note: "All Chainages Found" list is intentionally omitted — start/end + coverage is sufficient
    return items


def extract_structures(text: str) -> List[str]:
    """Extract structures — culverts, bridges, ROBs, underpasses, etc."""
    items = []
    
    # Box Culvert patterns
    culvert_patterns = re.findall(
        r"(?:BOX\s+CULVERT|Box\s+Culvert|CULVERT|Culvert)[^\n]*(?:CH[:\s]*\d+\+\d+)?[^\n]*",
        text, re.IGNORECASE
    )
    for c in culvert_patterns:
        cleaned = " ".join(c.split())  # normalize whitespace
        if len(cleaned) > 20 or "CH" in cleaned.upper():
            items.append(f"Culvert: {cleaned}")
    
    # Bridge / ROB / Flyover patterns
    bridge_patterns = re.findall(
        r"(?:BRIDGE|Bridge|ROB|R\.O\.B|FLYOVER|Flyover|OVERPASS|RUB|R\.U\.B)[^\n]*",
        text, re.IGNORECASE
    )
    for b in bridge_patterns:
        cleaned = " ".join(b.split())
        if len(cleaned) > 20 or "CH" in cleaned.upper():
            items.append(f"Bridge/ROB: {cleaned}")
    
    # Underpass / SVUP / MNB
    underpass_patterns = re.findall(
        r"(?:UNDERPASS|Underpass|SVUP|MNB|MINOR\s+BRIDGE|VUP|PUP|LHS|RHS)[^\n]*(?:CH[:\s]*\d+\+\d+)?[^\n]*",
        text, re.IGNORECASE
    )
    for u in underpass_patterns:
        cleaned = " ".join(u.split())
        if len(cleaned) > 20 or "CH" in cleaned.upper():
            items.append(f"Underpass/Structure: {cleaned}")
    
    # Gas pipe / pipeline bridges
    pipe_bridge = re.findall(
        r"(?:GAS\s+PIPE\s+LINE\s+BRIDGE|PIPE\s*LINE\s+BRIDGE|PIPELINE\s+BRIDGE)[^\n]*",
        text, re.IGNORECASE
    )
    for p in pipe_bridge:
        cleaned = " ".join(p.split())
        if len(cleaned) > 5:
            items.append(f"Pipeline Bridge: {cleaned}")
    
    # Deduplicate
    items = list(dict.fromkeys(items))
    
    return items


def extract_road_geometry(text: str) -> List[str]:
    """Extract road geometry — curves, radius, deflection, speed, superelevation."""
    items = []
    
    # Horizontal curves: R = xxx
    radius_matches = re.findall(
        r"R\s*=\s*(\d+[\d.]*)\s*(?:m|M)?", text
    )
    for r in radius_matches:
        items.append(f"Radius: R = {r} m")
    
    # Design speed — two groups to avoid matching easting coords (e.g. N = 2555316)
    speed_matches = re.findall(
        r"(?:Design\s+Speed|Speed)[:\s=]*(\d+)\s*km|V\s*=\s*(\d+)\s*km",
        text, re.IGNORECASE
    )
    for match in speed_matches:
        s = next((m for m in match if m), None)
        if s and 40 <= int(s) <= 120:  # reasonable design speed range
            items.append(f"Design Speed: {s} km/h")
    
    # Superelevation
    se_matches = re.findall(
        r"(?:Super\s*elevation|e%|SE)[:\s=]*(-?\d+[.\d]*)\s*%?",
        text, re.IGNORECASE
    )
    for se in se_matches:
        items.append(f"Superelevation: {se}%")
    
    # Deflection angle
    deflection_matches = re.findall(
        r"(?:Deflection\s+Angle|Δ|Delta)[:\s=]*(\d+[°.\d]*)",
        text, re.IGNORECASE
    )
    for d in deflection_matches:
        items.append(f"Deflection Angle: {d}")
    
    # Curve length / Transition length
    curve_len = re.findall(
        r"(?:L[cs]|Length\s+of\s+(?:Circular|Spiral)\s+Curve)[:\s=]*(\d+[.\d]*)\s*(?:m)?",
        text, re.IGNORECASE
    )
    for cl in curve_len:
        items.append(f"Curve Length: {cl} m")
    
    # Tangent length
    tangent_matches = re.findall(
        r"(?:Tangent\s+Length|T[sL])[:\s=]*(\d+[.\d]*)\s*(?:m)?",
        text, re.IGNORECASE
    )
    for t in tangent_matches:
        items.append(f"Tangent Length: {t} m")
    
    # Gradient
    gradient_matches = re.findall(
        r"(?:Gradient|Grade|G)[:\s=]*(-?\d+[.\d]*)\s*%",
        text, re.IGNORECASE
    )
    for g in gradient_matches:
        items.append(f"Gradient: {g}%")
    
    # Horizontal Geometry section data
    horiz_match = re.search(
        r"HORIZONTAL\s+GEOMETRY[^\n]*\n((?:.*\n){1,10})",
        text, re.IGNORECASE
    )
    if horiz_match:
        items.append(f"Horizontal Geometry Data: {' '.join(horiz_match.group(1).split())[:200]}")
    
    # Vertical Geometry section data
    vert_match = re.search(
        r"VERTICAL\s+GEOMETRY[^\n]*\n((?:.*\n){1,10})",
        text, re.IGNORECASE
    )
    if vert_match:
        items.append(f"Vertical Geometry Data: {' '.join(vert_match.group(1).split())[:200]}")
    
    # Deduplicate
    items = list(dict.fromkeys(items))
    
    return items


def extract_utilities(text: str) -> List[str]:
    """Extract utilities & features — bus bays, pipelines, canals, railway crossings."""
    items = []
    
    # Gas Pipeline
    gas_matches = re.findall(
        r"(?:GAS\s+PIPE\s*LINE|Gas\s+Pipeline|GAIL|GAS\s+LINE)[^\n]*",
        text, re.IGNORECASE
    )
    for g in gas_matches:
        cleaned = " ".join(g.split())
        if len(cleaned) > 5:
            items.append(f"Gas Pipeline: {cleaned}")
    
    # Bus Bay
    bus_matches = re.findall(
        r"(?:BUS\s+BAY|Bus\s+Bay|BUS\s+STOP)[^\n]*",
        text, re.IGNORECASE
    )
    for b in bus_matches:
        items.append(f"Bus Bay: {' '.join(b.split())}")
    
    # Canal
    canal_matches = re.findall(
        r"(?:CANAL|Canal|IRRIGATION)[^\n]*",
        text, re.IGNORECASE
    )
    for c in canal_matches:
        cleaned = " ".join(c.split())
        if len(cleaned) > 5:
            items.append(f"Canal: {cleaned}")
    
    # Railway crossing
    railway_matches = re.findall(
        r"(?:RAILWAY|Railway|RAIL\s+LINE|LC\s+NO|LEVEL\s+CROSSING)[^\n]*",
        text, re.IGNORECASE
    )
    for r in railway_matches:
        items.append(f"Railway: {' '.join(r.split())}")
    
    # Wire heights / HT / LT lines
    wire_matches = re.findall(
        r"(?:WIRE\s+HEIGHT|HT\s+LINE|LT\s+LINE|HIGH\s+TENSION|LOW\s+TENSION|ELECTRIC|TELEPHONE)[^\n]*",
        text, re.IGNORECASE
    )
    for w in wire_matches:
        cleaned = " ".join(w.split())
        if len(cleaned) > 5:
            items.append(f"Utility Line: {cleaned}")
    
    # Water pipeline / supply
    water_matches = re.findall(
        r"(?:WATER\s+(?:PIPE|SUPPLY|LINE)|W\.?P\.?L)[^\n]*",
        text, re.IGNORECASE
    )
    for w in water_matches:
        items.append(f"Water Line: {' '.join(w.split())}")
    
    # OFC (Optical Fiber Cable)
    ofc_matches = re.findall(
        r"(?:OFC|OPTICAL\s+FIBER|FIBRE\s+CABLE)[^\n]*",
        text, re.IGNORECASE
    )
    for o in ofc_matches:
        items.append(f"OFC: {' '.join(o.split())}")
    
    # Deduplicate
    items = list(dict.fromkeys(items))
    
    return items


def extract_annotations(text: str) -> List[str]:
    """Extract annotations — benchmarks, GPS, TBMs, km stones, land use."""
    items = []
    
    # Bench Marks (BM)
    bm_matches = re.findall(
        r"(?:B\.?M\.?\s*(?:No\.?)?\s*\d*|BENCH\s*MARK)[^\n]*(?:\d+[.\d]*)?[^\n]*",
        text, re.IGNORECASE
    )
    for b in bm_matches:
        cleaned = " ".join(b.split())
        if len(cleaned) > 3:
            items.append(f"Benchmark: {cleaned}")
    
    # GPS Points
    gps_matches = re.findall(
        r"(?:GPS\s*(?:Point|POINT|No\.?|#)?)[^\n]*",
        text, re.IGNORECASE
    )
    for g in gps_matches:
        cleaned = " ".join(g.split())
        if len(cleaned) > 3:
            items.append(f"GPS: {cleaned}")
    
    # TBM (Temporary Bench Mark)
    tbm_matches = re.findall(
        r"(?:T\.?B\.?M\.?\s*(?:No\.?)?\s*\d*)[^\n]*",
        text, re.IGNORECASE
    )
    for t in tbm_matches:
        cleaned = " ".join(t.split())
        if len(cleaned) > 3:
            items.append(f"TBM: {cleaned}")
    
    # Kilometre Stones
    km_matches = re.findall(
        r"(?:KM\s*(?:STONE|Stone)|KILOMETRE\s+STONE|K\.?M\.?\s*\d+)[^\n]*",
        text, re.IGNORECASE
    )
    for k in km_matches:
        cleaned = " ".join(k.split())
        if len(cleaned) > 3:
            items.append(f"KM Stone: {cleaned}")
    
    # Land use labels
    land_patterns = [
        r"(?:AGRICULTURAL|BARREN|RESIDENTIAL|COMMERCIAL|INDUSTRIAL|WASTELAND|GRAZING)\s*(?:LAND)?",
        r"(?:VILLAGE|TOWN|CITY|SETTLEMENT)\s+\w+",
    ]
    for pat in land_patterns:
        land_matches = re.findall(pat, text, re.IGNORECASE)
        for l in land_matches:
            items.append(f"Land Use: {' '.join(l.split())}")
    
    # Traverse Points
    traverse_matches = re.findall(
        r"(?:TRAVERSE|Traverse)\s*(?:Point|POINT|Station)?\s*[^\n]*",
        text, re.IGNORECASE
    )
    for t in traverse_matches:
        cleaned = " ".join(t.split())
        if len(cleaned) > 5:
            items.append(f"Traverse: {cleaned}")
    
    # North direction / compass
    if re.search(r"NORTH|North\s+Symbol", text, re.IGNORECASE):
        items.append("North Direction: Indicated on drawing")
    
    # Deduplicate
    items = list(dict.fromkeys(items))
    
    return items
