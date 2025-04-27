import logging
from typing import List, Dict

logger = logging.getLogger("pdfine.universal_extract")

# Try to import all possible backends
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except ImportError:
    pdfminer_extract_text = None
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None
import io
import os

def extract_with_pymupdf(pdf_path: str) -> List[Dict]:
    if not fitz:
        return []
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text("text") or ""
            if not text.strip() and Image and pytesseract:
                # Try OCR on image
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                text = pytesseract.image_to_string(img)
            pages.append({"text": text})
        return pages
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
        return []

def extract_with_pdfminer(pdf_path: str) -> List[Dict]:
    if not pdfminer_extract_text:
        return []
    try:
        text = pdfminer_extract_text(pdf_path)
        # Split by form feed (page break)
        pages = [p.strip() for p in text.split('\f') if p.strip()]
        return [{"text": p} for p in pages]
    except Exception as e:
        logger.warning(f"pdfminer extraction failed: {e}")
        return []

def extract_with_pypdf2(pdf_path: str) -> List[Dict]:
    if not PyPDF2:
        return []
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append({"text": text})
        return pages
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
        return []

def extract_with_ocr(pdf_path: str) -> List[Dict]:
    if not (fitz and Image and pytesseract):
        return []
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = pytesseract.image_to_string(img)
            pages.append({"text": text})
        return pages
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        return []

def universal_extract_pages(pdf_path: str) -> List[Dict]:
    """
    Try all extraction strategies in order. Return as much as possible from any PDF.
    """
    strategies = [
        extract_with_pymupdf,
        extract_with_pdfminer,
        extract_with_pypdf2,
        extract_with_ocr,
    ]
    for strat in strategies:
        pages = strat(pdf_path)
        if pages and any(p.get("text", "").strip() for p in pages):
            logger.info(f"Extracted PDF using {strat.__name__}")
            return pages
    logger.error(f"All extraction backends failed for {pdf_path}")
    return []
