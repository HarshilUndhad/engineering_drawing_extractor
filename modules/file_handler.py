"""
File Handler Module
Handles file upload routing, format detection, and PDF-to-image conversion.
Supports JPG, PNG, and PDF engineering drawings.

Uses pypdfium2 (Chrome's PDFium engine) for highly accurate CAD/vector PDF rendering.
"""

import io
from typing import List, Tuple
from PIL import Image
import pypdfium2 as pdfium

from config import PDF_DPI


def detect_file_type(file_name: str) -> str:
    """
    Detect the file type from the filename extension.
    """
    extension = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if extension == "pdf":
        return "pdf"
    elif extension in ("jpg", "jpeg"):
        return "jpg"
    elif extension == "png":
        return "png"
    else:
        return "unknown"


def process_upload(file_bytes: bytes, file_name: str) -> Tuple[List[Image.Image], str, bytes]:
    """
    Process an uploaded file: detect type, convert to PIL Images.
    
    For PDFs: renders each page to a PIL Image using PyMuPDF.
    For images: loads directly as a single PIL Image.
    """
    file_type = detect_file_type(file_name)
    
    if file_type == "unknown":
        raise ValueError(
            f"Unsupported file type: {file_name}. "
            "Please upload a JPG, PNG, or PDF file."
        )
    
    if file_type == "pdf":
        images = convert_pdf_to_images(file_bytes)
    else:
        images = [load_image(file_bytes)]
    
    return images, file_type, file_bytes


def convert_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """
    Convert a PDF file to a list of PIL Images using pypdfium2.

    We use pypdfium2 (Chrome\'s PDFium engine) for rendering instead of PyMuPDF
    because CAD-generated engineering drawings heavily use Optional Content
    Groups (layers) and complex blend modes that PyMuPDF struggles to render
    correctly, causing disappearing hatching and colored bands.
    """
    doc = pdfium.PdfDocument(pdf_bytes)
    images = []

    # Calculate scale factor from target DPI (base is 72 DPI)
    scale = PDF_DPI / 72.0

    for page in doc:
        # Render the page
        bitmap = page.render(
            scale=scale,
            rev_byteorder=False,  # Outputs RGB format natively
        )
        # Convert to a PIL Image (creates a copy of the pixel buffer)
        image = bitmap.to_pil()
        
        # Ensure RGB mode (standardises image format for vision models)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        images.append(image)

    doc.close()
    return images


def load_image(image_bytes: bytes) -> Image.Image:
    """
    Load image bytes into a PIL Image.
    """
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def get_image_for_display(image: Image.Image, max_width: int = 5000) -> Image.Image:
    """
    Resize image for UI display if it's too large.

    Default max_width is 5000 px — large enough that 300 DPI engineering 
    drawings are usually not downsampled. Streamlit will gracefully fit the 
    image to screen, but zooming/opening in a new tab will retain full crisp
    text and vector sharpness.
    """
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        # LANCZOS (Lanczos3) is the highest-quality downsampling filter —
        # it preserves fine lines and text better than BILINEAR/BICUBIC.
        return image.resize(new_size, Image.LANCZOS)
    return image

