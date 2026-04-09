"""
Engineering Drawing Intelligence Extractor
Main Streamlit application entry point.

Run with: streamlit run app.py

Hybrid approach:
1. PyMuPDF extracts text directly from PDF (high accuracy)
2. Ollama vision LLM analyses the drawing image (spatial understanding)
3. Both outputs are merged (text preferred, vision fills gaps)
4. Text-only LLM generates a professional narrative summary
"""

import streamlit as st
from PIL import Image

from config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_LABEL,
    NARRATIVE_MODELS,
    DEFAULT_NARRATIVE_MODEL,
    EXTRACTION_CATEGORIES,
    CATEGORY_DESCRIPTIONS,
    SUPPORTED_FILE_TYPES,
)
from modules.file_handler import process_upload, get_image_for_display
from modules.text_extractor import parse_structured_data
from modules.vision_extractor import extract_with_vision, check_ollama_connection
from modules.merger import merge_extractions, get_extraction_summary
from modules.narrator import generate_narrative


# ──────────────────────────────────────────────
# Cached text extraction wrapper
# Wraps parse_structured_data so PDF text isn't re-extracted when only the
# model selection changes (same bytes = cache hit).
# ──────────────────────────────────────────────
@st.cache_data
def _cached_text_extract(pdf_bytes: bytes) -> dict:
    return parse_structured_data(pdf_bytes)


# ──────────────────────────────────────────────
# Helper Functions (defined before main UI code)
# ──────────────────────────────────────────────

def _format_structured_export(data: dict) -> str:
    """Format merged data as a downloadable text file."""
    lines = ["=" * 60]
    lines.append("ENGINEERING DRAWING — EXTRACTED INTELLIGENCE")
    lines.append("=" * 60)
    
    for category in EXTRACTION_CATEGORIES:
        items = data.get(category, [])
        lines.append(f"\n{'─' * 40}")
        lines.append(f"  {category.upper()}")
        lines.append(f"{'─' * 40}")
        
        if items:
            for item in items:
                lines.append(f"  • {item}")
        else:
            lines.append("  (No data found)")
    
    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def _items_to_table_data(items: list) -> list:
    """Convert a list of extracted items into table-friendly rows."""
    rows = []
    for item in items:
        if ": " in item:
            field, value = item.split(": ", 1)
            rows.append({"Field": field.strip(), "Details": value.strip()})
        else:
            rows.append({"Field": "—", "Details": item.strip()})
    return rows


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Engineering Drawing Intelligence Extractor",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS for professional styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5f8a 50%, #1e8a6a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #d0e8ff;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Category cards */
    .category-header {
        background: linear-gradient(90deg, #f0f4f8 0%, #ffffff 100%);
        border-left: 4px solid #2d5f8a;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-weight: 600;
        color: #1e3a5f;
    }
    
    /* Narrative box */
    .narrative-box {
        background: linear-gradient(135deg, #f8fffe 0%, #f0f7ff 100%);
        border: 1px solid #b8d4e8;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        line-height: 1.7;
        font-size: 1.02rem;
        color: #2c3e50;
    }
    
    /* Stats cards */
    .stat-card {
        background: #f7f9fc;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d5f8a;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #6b7c8d;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background: #f0f7ff;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.85rem;
        border: 1px solid #d0e0f0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar — Model Selection & Settings
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    # ── Vision Model ──
    st.markdown("**👁️ Vision Model** — for drawing analysis")
    selected_vision_label = st.selectbox(
        "Vision Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_LABEL),
        help="Analyses the drawing image. Smaller models are faster on low-RAM machines.",
        label_visibility="collapsed",
    )
    vision_model_tag = AVAILABLE_MODELS[selected_vision_label]

    st.markdown(f"""
    <div class="sidebar-info">
        <b>Selected:</b> <code>{vision_model_tag}</code><br>
        <b>Pull command:</b><br>
        <code>ollama pull {vision_model_tag}</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Narrative Model ──
    st.markdown("**📝 Narrative Model** — for text summary")
    selected_narrative_label = st.selectbox(
        "Narrative Model",
        options=list(NARRATIVE_MODELS.keys()),
        index=list(NARRATIVE_MODELS.keys()).index(DEFAULT_NARRATIVE_MODEL),
        help="Text-only model for generating narrative summary. Much faster than vision models.",
        label_visibility="collapsed",
    )
    narrative_model_tag = NARRATIVE_MODELS[selected_narrative_label]

    st.markdown(f"""
    <div class="sidebar-info">
        <b>Selected:</b> <code>{narrative_model_tag}</code><br>
        <b>Pull command:</b><br>
        <code>ollama pull {narrative_model_tag}</code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Connection check
    if st.button("🔍 Check Ollama Connection", use_container_width=True):
        status = check_ollama_connection(vision_model_tag)
        if status["ok"]:
            st.success(status["message"])
        else:
            st.error(status["message"])
    
    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown("""
    **Hybrid extraction approach:**
    1. 📄 **Text Extraction** — PyMuPDF reads text directly from PDF (high accuracy)
    2. 👁️ **Vision Analysis** — AI model analyses the drawing image
    3. 🔀 **Smart Merge** — combines both, preferring text where available
    4. 📝 **Narrative** — fast text-only model generates professional summary
    """)


# ──────────────────────────────────────────────
# Main Content — Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📐 Engineering Drawing Intelligence Extractor</h1>
    <p>Upload an engineering drawing to extract structured information and generate a professional narrative summary.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Step 1 — File Upload
# ──────────────────────────────────────────────
st.markdown("### 📁 Step 1 — Upload Drawing")

uploaded_file = st.file_uploader(
    "Upload an engineering drawing (JPG, PNG, or PDF)",
    type=SUPPORTED_FILE_TYPES,
    help="Supports engineering drawings in JPG, PNG, or PDF format. PDF is recommended for best text extraction.",
)

if uploaded_file is not None:
    # Process the uploaded file
    try:
        file_bytes = uploaded_file.read()

        # Cache processed images in session_state to avoid re-processing on every
        # button click (buttons trigger a natural Streamlit rerun — we don't want
        # to re-convert the PDF every time the user hits ◀ / ▶)
        file_key = f"{uploaded_file.name}_{len(file_bytes)}"
        if st.session_state.get("_file_key") != file_key:
            images, file_type, raw_bytes = process_upload(file_bytes, uploaded_file.name)
            st.session_state._file_key = file_key
            st.session_state._images = images
            st.session_state._file_type = file_type
            st.session_state._raw_bytes = raw_bytes
            st.session_state.current_page = 1  # reset to page 1 on new file
        else:
            images = st.session_state._images
            file_type = st.session_state._file_type
            raw_bytes = st.session_state._raw_bytes

        st.success(
            f"✅ Loaded **{uploaded_file.name}** — "
            f"{file_type.upper()} file, {len(images)} page(s), "
            f"{len(file_bytes) / 1024:.0f} KB"
        )
        
        # ──────────────────────────────────────
        # Drawing Preview
        # ──────────────────────────────────────
        st.markdown("### 🖼️ Drawing Preview")
        
        if len(images) > 1:
            # Clamp current_page to valid range on every rerun
            cp = int(st.session_state.get("current_page", 1))
            cp = max(1, min(cp, len(images)))
            st.session_state.current_page = cp

            nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

            btn_prev = btn_next = False
            with nav_col1:
                btn_prev = st.button("◀ Prev", use_container_width=True, disabled=cp <= 1)

            with nav_col2:
                # Compute a safe index even if session state is momentarily stale
                safe_index = max(0, min(cp - 1, len(images) - 1))
                selected = st.selectbox(
                    "Select page to preview",
                    options=list(range(1, len(images) + 1)),
                    index=safe_index,
                    format_func=lambda x: f"Page {x} of {len(images)}",
                    key="page_selector",
                )

            with nav_col3:
                btn_next = st.button("Next ▶", use_container_width=True, disabled=cp >= len(images))

            # Resolve priority: buttons take precedence over selectbox changes
            if btn_prev:
                st.session_state.current_page = max(1, cp - 1)
            elif btn_next:
                st.session_state.current_page = min(len(images), cp + 1)
            else:
                st.session_state.current_page = selected

            page_num = st.session_state.current_page
            display_image = get_image_for_display(images[page_num - 1])
        else:
            page_num = 1
            display_image = get_image_for_display(images[0])
        
        st.image(
            display_image, 
            caption=f"{uploaded_file.name} — Page {page_num}", 
            use_container_width=True,
            output_format="PNG"  # Prevents JPEG compression artifacts
        )
        
        # ──────────────────────────────────────
        # Step 2 — Extract
        # ──────────────────────────────────────
        st.markdown("### 🔍 Step 2 — Extract Intelligence")
        
        extract_col1, extract_col2 = st.columns([2, 1])
        with extract_col1:
            extract_button = st.button(
                "🚀 Extract Intelligence",
                type="primary",
                use_container_width=True,
            )
        with extract_col2:
            extract_all_pages = st.checkbox(
                "Analyse all pages",
                value=True,
                help="When enabled, analyses all pages of a multi-page PDF. Disable to analyse only the selected page.",
            )
        
        if extract_button:
            # ──────────────────────────────────
            # Run extraction pipeline
            # ──────────────────────────────────
            
            progress = st.progress(0, text="Starting extraction pipeline...")
            
            # --- Text Extraction (PDF only) ---
            text_data = {cat: [] for cat in EXTRACTION_CATEGORIES}
            
            if file_type == "pdf":
                progress.progress(10, text="📄 Extracting text from PDF with PyMuPDF...")
                text_data = _cached_text_extract(raw_bytes)
                progress.progress(30, text="📄 Text extraction complete.")
            else:
                progress.progress(30, text="ℹ️ Image file — skipping text extraction (PDF only).")
            
            # --- Vision Extraction ---
            progress.progress(35, text=f"👁️ Analysing drawing with {vision_model_tag}... (this may take a few minutes)")
            
            vision_data = {cat: [] for cat in EXTRACTION_CATEGORIES}
            vision_error = None
            
            pages_to_analyse = images if extract_all_pages else [images[page_num - 1]]
            
            for i, img in enumerate(pages_to_analyse):
                page_label = f"page {i+1}/{len(pages_to_analyse)}"
                progress.progress(
                    35 + int(40 * (i / len(pages_to_analyse))),
                    text=f"👁️ Vision analysis — {page_label} with {vision_model_tag}..."
                )
                
                with st.spinner(f"Vision model processing {page_label} — this can take 2-3 min per page..."):
                    page_vision = extract_with_vision(img, vision_model_tag)
                
                # Check for errors
                if "_error" in page_vision:
                    vision_error = page_vision.pop("_error")
                
                # Merge page results into overall vision data
                for cat in EXTRACTION_CATEGORIES:
                    vision_data[cat].extend(page_vision.get(cat, []))
            
            progress.progress(75, text="🔀 Merging text and vision results...")
            
            # --- Merge ---
            merged_data = merge_extractions(text_data, vision_data)
            
            progress.progress(80, text=f"📝 Generating narrative with {narrative_model_tag}...")
            
            # --- Narrative ---
            with st.spinner(f"Generating narrative summary with {narrative_model_tag}..."):
                narrative = generate_narrative(merged_data, narrative_model_tag)
            
            progress.progress(100, text="✅ Extraction complete!")
            progress.empty()  # Clean up progress bar

            # ── Persist results in session_state so they survive reruns ──
            # (download buttons, sidebar changes, etc. all trigger reruns)
            st.session_state._results = {
                "merged_data": merged_data,
                "narrative": narrative,
                "vision_error": vision_error,
                "vision_model_used": vision_model_tag,
                "file_name": uploaded_file.name,
            }
            
        # ──────────────────────────────────────────────────────────────
        # Display results — rendered from session_state so they persist
        # across ANY rerun (download clicks, nav buttons, sidebar, etc.)
        # ──────────────────────────────────────────────────────────────
        results = st.session_state.get("_results")

        # Clear stale results if user uploads a different file
        if results and results.get("file_name") != uploaded_file.name:
            del st.session_state["_results"]
            results = None

        if results:
            merged_data = results["merged_data"]
            narrative   = results["narrative"]
            vision_error = results["vision_error"]
            vision_model_used = results["vision_model_used"]

            # Vision warning
            if vision_error:
                is_vram = bool(vision_error) and ("500" in vision_error or "VRAM" in vision_error or "out of" in vision_error.lower())
                if is_vram:
                    st.warning(
                        f"⚠️ **Vision extraction failed** (model: `{vision_model_used}`)\n\n"
                        f"**Likely cause:** Not enough RAM/VRAM to process the image at high resolution.\n\n"
                        f"**Fix options:**\n"
                        f"- Switch to **Qwen2.5-VL 3B** in the sidebar (uses ~2 GB RAM)\n"
                        f"- Close other applications to free RAM\n"
                        f"- The app will still show text-extracted data below\n\n"
                        f"*Technical detail: {vision_error}*"
                    )
                else:
                    st.warning(f"⚠️ Vision extraction issue: {vision_error}")

            # ── Step 3 — Structured Output ──
            st.markdown("---")
            st.markdown("### 📊 Step 3 — Structured Output")

            summary = get_extraction_summary(merged_data)
            total_items = sum(summary.values())

            stat_cols = st.columns(len(EXTRACTION_CATEGORIES) + 1)
            with stat_cols[0]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{total_items}</div>
                    <div class="stat-label">Total Items</div>
                </div>
                """, unsafe_allow_html=True)

            for i, category in enumerate(EXTRACTION_CATEGORIES):
                with stat_cols[i + 1]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{summary[category]}</div>
                        <div class="stat-label">{category}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("")

            for category in EXTRACTION_CATEGORIES:
                items = merged_data.get(category, [])
                description = CATEGORY_DESCRIPTIONS.get(category, "")
                with st.expander(
                    f"📌 {category} ({len(items)} items)",
                    expanded=len(items) > 0,
                ):
                    st.caption(description)
                    if items:
                        st.dataframe(_items_to_table_data(items), use_container_width=True, hide_index=True)
                    else:
                        st.info("No data found in this category.")

            # ── Step 4 — Narrative Summary ──
            st.markdown("---")
            st.markdown("### 📝 Step 4 — Narrative Summary")
            # Use a styled container — raw HTML divs don't process markdown syntax
            with st.container():
                st.markdown(
                    "<style>.narrative-box-wrapper{background:var(--secondary-background-color);"
                    "border-radius:12px;padding:1.5rem 2rem;border-left:4px solid #3b82f6;"
                    "margin-bottom:1rem;}</style>"
                    "<div class='narrative-box-wrapper'></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(narrative)

            # ── Export ──
            st.markdown("---")
            st.markdown("### 💾 Export")

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "📋 Download Structured Data (.txt)",
                    data=_format_structured_export(merged_data),
                    file_name=f"{uploaded_file.name}_extracted.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    "📝 Download Narrative (.txt)",
                    data=narrative,
                    file_name=f"{uploaded_file.name}_narrative.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

else:
    # No file uploaded — show instructions
    st.info(
        "👆 Upload an engineering drawing to get started. "
        "PDF files give the best results as text can be extracted directly."
    )
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 📄 Text Extraction
        Directly reads text from PDFs with perfect accuracy — 
        chainage values, structure tables, road geometry data.
        """)
    with col2:
        st.markdown("""
        #### 👁️ Vision Analysis
        AI model analyses the drawing image to understand
        spatial layout and visual features.
        """)
    with col3:
        st.markdown("""
        #### 📝 Smart Summary
        Professional narrative summary written as a
        senior engineer briefing a project manager.
        """)
