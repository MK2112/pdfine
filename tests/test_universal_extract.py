import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest import mock
from pdfine.universal_extract import universal_extract_pages

PDF_PATH = "dummy.pdf"

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
