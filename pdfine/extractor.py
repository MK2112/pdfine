from .parser import (
    parse_pdf_header,
    find_xref_offset,
    parse_xref_table,
    parse_trailer,
    extract_objects,
    find_pages,
    extract_content_streams,
    extract_text_from_stream
)
from .utils import PDFParseError, logger
"""
Raw PDF page text extraction with a from-scratch parser
"""

def raw_extract_pages(pdf_path: str):
    try:
        with open(pdf_path, 'rb') as f:
            data = f.read()
        parse_pdf_header(data)
        xref_offset = find_xref_offset(data)
        xref = parse_xref_table(data, xref_offset)
        parse_trailer(data, xref_offset)
        objects = extract_objects(data, xref)
        page_nums = find_pages(objects)
        pages = []
        for idx, num in enumerate(page_nums, start=1):
            streams = extract_content_streams(objects, num)
            text = ''.join(extract_text_from_stream(s) for s in streams)
            pages.append({'page_num': idx, 'text': text.strip()})
        return pages
    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {e}")
        return []
