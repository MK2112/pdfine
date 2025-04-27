import os
import sys
import shutil
import tempfile
import pytest
from unittest import mock
from pathlib import Path

# Patch sys.path so we can import PDFine.py directly
sys.path.insert(0, str(Path(__file__).parent.parent))
import PDFine

def make_dummy_pdf(path, content="Dummy PDF content"):  # Not a real PDF, but enough for mocks
    with open(path, "w") as f:
        f.write(content)

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

@pytest.fixture(autouse=True)
def cleanup_globals():
    PDFine._seq_model = None
    PDFine._seq_tokenizer = None
    PDFine.MODEL_DIR = None
    yield
    PDFine._seq_model = None
    PDFine._seq_tokenizer = None
    PDFine.MODEL_DIR = None

def test_process_pdf_modular(monkeypatch, temp_dir):
    pdf_path = os.path.join(temp_dir, "test.pdf")
    make_dummy_pdf(pdf_path)
    # Mock raw_extract_pages to return fake pages
    monkeypatch.setattr(PDFine, "raw_extract_pages", lambda x: [{"text": "Page 1 text"}, {"text": "Page 2 text"}])
    # Mock refine_pages and pages_to_markdown
    monkeypatch.setattr(PDFine, "refine_pages", lambda path, pages: ["Refined Page 1", "Refined Page 2"])
    monkeypatch.setattr(PDFine, "pages_to_markdown", lambda refined: [f"# Markdown {i}" for i in range(len(refined))])
    result = PDFine.process_pdf(pdf_path)
    assert result == ["# Markdown 0", "# Markdown 1"]

def test_process_pdf_model(monkeypatch, temp_dir):
    pdf_path = os.path.join(temp_dir, "test.pdf")
    make_dummy_pdf(pdf_path)
    PDFine.MODEL_DIR = "dummy-model-dir"
    # Mock model/tokenizer loading and inference
    class DummyModel:
        def eval(self): pass
        def generate(self, input_ids, attention_mask, max_length): return [[101, 102, 103]]
    class DummyBatch:
        def __init__(self):
            self.input_ids = [[1,2,3]]
            self.attention_mask = [[1,1,1]]
    class DummyTokenizer:
        def __call__(self, text, return_tensors, truncation, max_length):
            return DummyBatch()
        def decode(self, ids, skip_special_tokens): return "# Model Markdown output"
        @staticmethod
        def from_pretrained(path): return DummyTokenizer()
    monkeypatch.setattr(PDFine, "T5ForConditionalGeneration", mock.Mock(from_pretrained=lambda path: DummyModel()))
    monkeypatch.setattr(PDFine, "T5TokenizerFast", DummyTokenizer)
    monkeypatch.setattr(PDFine, "raw_extract_pages", lambda x: [{"text": "Page 1 text"}, {"text": "Page 2 text"}])
    result = PDFine.process_pdf(pdf_path)
    assert result == ["# Model Markdown output"]

def test_convert_file_and_delete(monkeypatch, temp_dir):
    pdf_path = os.path.join(temp_dir, "test.pdf")
    out_path = os.path.join(temp_dir, "test.md")
    make_dummy_pdf(pdf_path)
    monkeypatch.setattr(PDFine, "process_pdf", lambda path: ["# Out"])
    PDFine.convert_file(pdf_path, out_path, delete_src=True)
    assert os.path.exists(out_path)
    assert not os.path.exists(pdf_path)

def test_convert_folder_and_concat_and_delete(monkeypatch, temp_dir):
    input_dir = os.path.join(temp_dir, "input")
    os.makedirs(input_dir)
    output_dir = os.path.join(temp_dir, "output")
    pdf1 = os.path.join(input_dir, "a.pdf")
    pdf2 = os.path.join(input_dir, "b.pdf")
    make_dummy_pdf(pdf1)
    make_dummy_pdf(pdf2)
    # Mock process_pdf
    monkeypatch.setattr(PDFine, "process_pdf", lambda path: [f"# Out {os.path.basename(path)}"])
    PDFine.convert_folder(input_dir, output_dir, concat=True, delete_src=True)
    assert os.path.exists(os.path.join(output_dir, "a.md"))
    assert os.path.exists(os.path.join(output_dir, "b.md"))
    assert os.path.exists(os.path.join(output_dir, "concatenated.md"))
    assert not os.path.exists(pdf1)
    assert not os.path.exists(pdf2)
    with open(os.path.join(output_dir, "concatenated.md")) as f:
        content = f.read()
        assert "# Out a.pdf" in content
        assert "# Out b.pdf" in content

def test_main_file(monkeypatch, temp_dir):
    pdf_path = os.path.join(temp_dir, "test.pdf")
    make_dummy_pdf(pdf_path)
    out_dir = os.path.join(temp_dir, "out")
    monkeypatch.setattr(PDFine, "convert_file", lambda f, o, delete_src=False: open(o, "w").write("# Main test"))
    test_args = ["PDFine.py", "-f", pdf_path, "-o", out_dir]
    monkeypatch.setattr(sys, "argv", test_args)
    PDFine.main()
    out_path = os.path.join(out_dir, "test.md")
    assert os.path.exists(out_path)
    with open(out_path) as f:
        assert f.read() == "# Main test"

def test_main_input(monkeypatch, temp_dir):
    input_dir = os.path.join(temp_dir, "input")
    os.makedirs(input_dir)
    pdf_path = os.path.join(input_dir, "test.pdf")
    make_dummy_pdf(pdf_path)
    out_dir = os.path.join(temp_dir, "out")
    def mock_convert_folder(i, o, concat=False, delete_src=False):
        os.makedirs(o, exist_ok=True)
        with open(os.path.join(o, "test.md"), "w") as f:
            f.write("# Folder test")
    monkeypatch.setattr(PDFine, "convert_folder", mock_convert_folder)
    test_args = ["PDFine.py", "-i", input_dir, "-o", out_dir]
    monkeypatch.setattr(sys, "argv", test_args)
    PDFine.main()
    out_path = os.path.join(out_dir, "test.md")
    assert os.path.exists(out_path)
    with open(out_path) as f:
        assert f.read() == "# Folder test"

def test_process_pdf_error(monkeypatch, temp_dir):
    pdf_path = os.path.join(temp_dir, "fail.pdf")
    make_dummy_pdf(pdf_path)
    # Force process_pdf to raise
    monkeypatch.setattr(PDFine, "raw_extract_pages", lambda x: 1/0)
    result = PDFine.process_pdf(pdf_path)
    assert result == []
