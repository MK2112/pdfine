import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest import mock
from pdfine.universal_extract import universal_extract_pages

PDF_PATH = "dummy.pdf"

# --- EXTENDED TESTS FOR ADVANCED UNIVERSAL EXTRACTOR ---

def test_table_extraction(monkeypatch):
    # Simulate camelot and tabula returning tables
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_tables", lambda path: [
        {"type": "table", "source": "camelot", "page_num": 0, "text": "|A|B|", "table_data": "A,B\n1,2"},
        {"type": "table", "source": "tabula", "page_num": 1, "text": "|C|D|", "table_data": "C,D\n3,4"}
    ])
    monkeypatch.setattr("pdfine.universal_extract.extract_metadata", lambda path: {})
    result = universal_extract_pages(PDF_PATH)
    assert any(p.get("type") == "table" or "tables" in p for p in result)

def test_image_ocr_extraction(monkeypatch):
    # Simulate PyMuPDF extracting image OCR text
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [
        {"text": "", "images": [{"ocr_text": "Image OCR text", "img_index": 0}], "source": "pymupdf", "page_num": 0, "type": "text"}
    ])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_tables", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_metadata", lambda path: {})
    result = universal_extract_pages(PDF_PATH)
    assert "images" in result[0] and result[0]["images"][0]["ocr_text"] == "Image OCR text"

def test_metadata_and_bookmarks(monkeypatch):
    # Simulate metadata extraction
    meta = {"title": "Test PDF", "author": "Test Author", "bookmarks": ["Section 1", "Section 2"]}
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_tables", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_metadata", lambda path: meta)
    result = universal_extract_pages(PDF_PATH)
    assert result[0]["type"] == "meta"
    assert "title" in result[0]["meta"]
    assert "bookmarks" in result[0]["meta"]

def test_encrypted_pdf(monkeypatch):
    # Simulate all backends failing except OCR
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [{"text": "Recovered from OCR", "source": "ocr", "page_num": 0, "type": "ocr"}])
    monkeypatch.setattr("pdfine.universal_extract.extract_tables", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_metadata", lambda path: {})
    result = universal_extract_pages(PDF_PATH)
    assert any("Recovered" in p["text"] for p in result)

def test_rich_output_structure(monkeypatch):
    # Simulate a complex output with all fields
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [
        {"text": "Text1", "images": [{"ocr_text": "img1", "img_index": 0}], "source": "pymupdf", "page_num": 0, "type": "text"},
        {"text": "Text2", "images": [], "source": "pymupdf", "page_num": 1, "type": "text"}
    ])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [
        {"text": "Text1 miner", "source": "pdfminer", "type": "text", "page_num": 0},
        {"text": "Text2 miner", "source": "pdfminer", "type": "text", "page_num": 1}
    ])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_tables", lambda path: [
        {"type": "table", "source": "camelot", "page_num": 1, "text": "|A|B|", "table_data": "A,B\n1,2"}
    ])
    monkeypatch.setattr("pdfine.universal_extract.extract_metadata", lambda path: {"title": "X"})
    result = universal_extract_pages(PDF_PATH)
    # Check meta, text, table, images
    assert result[0]["type"] == "meta"
    assert any("images" in p or "tables" in p for p in result)

@pytest.fixture(autouse=True)
def cleanup_imports(monkeypatch):
    # Patch out logging to avoid clutter
    monkeypatch.setattr("pdfine.universal_extract.logger", mock.Mock())


def test_pymupdf_success(monkeypatch):
    # PyMuPDF returns good text, should be used
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [{"text": "PyMuPDF text"}])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [{"text": "pdfminer text"}])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [{"text": "PyPDF2 text"}])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [{"text": "OCR text"}])
    result = universal_extract_pages(PDF_PATH)
    assert result == [{"text": "PyMuPDF text"}]

def test_pdfminer_fallback(monkeypatch):
    # PyMuPDF fails, pdfminer succeeds
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [{"text": "pdfminer text"}])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [{"text": "PyPDF2 text"}])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [{"text": "OCR text"}])
    result = universal_extract_pages(PDF_PATH)
    assert result == [{"text": "pdfminer text"}]

def test_pypdf2_fallback(monkeypatch):
    # Only PyPDF2 works
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [{"text": "PyPDF2 text"}])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [{"text": "OCR text"}])
    result = universal_extract_pages(PDF_PATH)
    assert result == [{"text": "PyPDF2 text"}]

def test_ocr_fallback(monkeypatch):
    # Only OCR works
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [{"text": "OCR text"}])
    result = universal_extract_pages(PDF_PATH)
    assert result == [{"text": "OCR text"}]

def test_all_fail(monkeypatch):
    # All backends fail
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [])
    result = universal_extract_pages(PDF_PATH)
    assert result == []

def test_partial_blank_pages(monkeypatch):
    # PyMuPDF returns blank, pdfminer returns one page with text
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pymupdf", lambda path: [{"text": "   "}])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pdfminer", lambda path: [{"text": "page1"}, {"text": ""}])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_pypdf2", lambda path: [])
    monkeypatch.setattr("pdfine.universal_extract.extract_with_ocr", lambda path: [])
    result = universal_extract_pages(PDF_PATH)
    assert result == [{"text": "page1"}, {"text": ""}]
