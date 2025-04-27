import logging
from typing import List, Dict, Optional
import concurrent.futures
import difflib
import io
import os

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
try:
    import camelot
except ImportError:
    camelot = None
try:
    import tabula
except ImportError:
    tabula = None

# Helper for fuzzy matching
def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def extract_with_pymupdf(pdf_path: str) -> List[Dict]:
    if not fitz:
        return []
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            images = []
            # Extract all images and OCR them
            for img_index, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                if Image and pytesseract:
                    img = Image.open(io.BytesIO(img_bytes))
                    ocr_text = pytesseract.image_to_string(img)
                    images.append({"ocr_text": ocr_text, "img_index": img_index})
            if not text.strip() and Image and pytesseract:
                # Try OCR on page image
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                text = pytesseract.image_to_string(img)
            pages.append({"text": text, "source": "pymupdf", "page_num": i, "type": "text", "images": images})
        return pages
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
        return []

def extract_with_pdfminer(pdf_path: str) -> List[Dict]:
    if not pdfminer_extract_text:
        return []
    try:
        text = pdfminer_extract_text(pdf_path)
        pages = [p.strip() for p in text.split('\f')]
        return [{"text": p, "source": "pdfminer", "type": "text", "page_num": i} for i, p in enumerate(pages)]
    except Exception as e:
        logger.warning(f"pdfminer extraction failed: {e}")
        return []

def extract_with_pypdf2(pdf_path: str) -> List[Dict]:
    if not PyPDF2:
        return []
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            annots = []
            if hasattr(page, 'annots') and page.annots:
                annots = [a.get_object() for a in page.annots]
            pages.append({"text": text, "source": "pypdf2", "type": "text", "page_num": i, "annotations": annots})
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
        for i, page in enumerate(doc):
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = pytesseract.image_to_string(img)
            pages.append({"text": text, "source": "ocr", "type": "ocr", "page_num": i})
        return pages
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        return []

def extract_tables(pdf_path: str) -> List[Dict]:
    tables = []
    # Try camelot
    if camelot:
        try:
            c_tables = camelot.read_pdf(pdf_path, pages="all")
            for t in c_tables:
                tables.append({"type": "table", "source": "camelot", "page_num": t.page, "table_data": t.df.to_csv(index=False), "text": t.df.to_markdown(index=False)})
        except Exception as e:
            logger.warning(f"Camelot table extraction failed: {e}")
    # Try tabula-py
    if tabula:
        try:
            t_tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
            for idx, t in enumerate(t_tables):
                tables.append({"type": "table", "source": "tabula", "page_num": idx, "table_data": t.to_csv(index=False), "text": t.to_markdown(index=False)})
        except Exception as e:
            logger.warning(f"Tabula table extraction failed: {e}")
    return tables

def extract_metadata(pdf_path: str) -> Dict:
    meta = {}
    # Try PyPDF2 for metadata and bookmarks
    if PyPDF2:
        try:
            reader = PyPDF2.PdfReader(pdf_path)
            meta = dict(reader.metadata or {})
            # Bookmarks
            try:
                outlines = reader.outline if hasattr(reader, 'outline') else reader.get_outlines()
                meta['bookmarks'] = outlines
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
    return meta

def merge_pages(*pagesets: List[List[Dict]]) -> List[Dict]:
    # Align by number, then try to merge by similarity
    best = max(pagesets, key=lambda x: len(x) if x else 0, default=[])
    merged = []
    for i in range(len(best)):
        merged_page = {"text": "", "sources": [], "page_num": i, "type": "text"}
        for pages in pagesets:
            if i < len(pages):
                p = pages[i]
                if p.get("text", "").strip():
                    merged_page["text"] += ("\n" if merged_page["text"] else "") + p["text"]
                    merged_page["sources"].append(p.get("source", "unknown"))
        merged.append(merged_page)
    return merged

def universal_extract_pages(pdf_path: str) -> List[Dict]:
    """
    Run all extraction strategies in parallel, merge and deduplicate results, extract tables/images/metadata.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futs = {
            "pymupdf": executor.submit(extract_with_pymupdf, pdf_path),
            "pdfminer": executor.submit(extract_with_pdfminer, pdf_path),
            "pypdf2": executor.submit(extract_with_pypdf2, pdf_path),
            "ocr": executor.submit(extract_with_ocr, pdf_path),
            "tables": executor.submit(extract_tables, pdf_path),
            "meta": executor.submit(extract_metadata, pdf_path),
        }
        results = {k: f.result() for k, f in futs.items()}
    text_pagesets = [results[k] for k in ["pymupdf", "pdfminer", "pypdf2", "ocr"] if results[k]]
    merged_pages = merge_pages(*text_pagesets)
    # Insert tables at page positions
    tables = results["tables"]
    for t in tables:
        pg = t.get("page_num", 0)
        if 0 <= pg < len(merged_pages):
            merged_pages[pg]["tables"] = merged_pages[pg].get("tables", []) + [t]
        else:
            merged_pages.append(t)
    # Add metadata as first page if present
    meta = results["meta"]
    if meta:
        merged_pages = [{"text": "# PDF Metadata\n" + "\n".join(f"**{k}**: {v}" for k, v in meta.items() if v), "type": "meta", "meta": meta, "page_num": -1}] + merged_pages
    # Clean up and deduplicate text
    for page in merged_pages:
        if "text" in page:
            page["text"] = page["text"].strip()
    logger.info(f"Universal extraction complete: {len(merged_pages)} pages (including meta/tables)")
    return merged_pages
