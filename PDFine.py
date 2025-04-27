"""
PDFine: Fast, Accurate PDF to Markdown Conversion (from-scratch parser, no pdfminer, no PyPDF2)

Usage:
    python PDFine.py -f [input file path]
    python PDFine.py -i [input folder path]
Options:
    -f, --file      Convert a single PDF file to markdown.
    -i, --input     Convert all PDF files in a folder to markdown.
    -o, --output    Output directory for the markdown files (default: ./output)
    -c, --concat    Concatenate all markdown files into a single file.
    -d, --delete    Delete source PDF file(s) after conversion.
    -m, --model-dir Directory of trained T5 model for end-to-end Markdown generation
"""

import os
import sys
import argparse
from tqdm import tqdm
from typing import List
from pdfine.utils import logger
from pdfine.layout import refine_pages
from pdfine.writer import pages_to_markdown
from pdfine.universal_extract import universal_extract_pages
from transformers import T5ForConditionalGeneration, T5TokenizerFast

_seq_model = None
_seq_tokenizer = None
MODEL_DIR = None

def process_pdf(pdf_path: str) -> List[str]:
    """
    If model-dir is set, use T5 inference; else, modular pipeline.
    Returns a list of Markdown strings, one per page or full doc.
    """
    global _seq_model, _seq_tokenizer, MODEL_DIR
    try:
        if MODEL_DIR:
            if _seq_model is None:
                _seq_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
                _seq_tokenizer = T5TokenizerFast.from_pretrained(MODEL_DIR)
                _seq_model.eval()
            pages = universal_extract_pages(pdf_path)
            text = '\n'.join(p.get('text', '') for p in pages if isinstance(p, dict))
            enc = _seq_tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
            out_ids = _seq_model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_length=512)
            md = _seq_tokenizer.decode(out_ids[0], skip_special_tokens=True)
            return [md.strip()]
        raw_pages = universal_extract_pages(pdf_path)
        refined_pages = refine_pages(pdf_path, raw_pages)
        md_pages = pages_to_markdown(refined_pages)
        return md_pages
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        return []

# File/Folder Handling
def convert_file(pdf_path: str, out_path: str, delete_src: bool = False):
    logger.info(f'Converting {pdf_path} -> {out_path}')
    pages = process_pdf(pdf_path)
    if pages:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("\n---\n".join(pages))
        logger.info(f'Success: {pdf_path}')
        if delete_src:
            try:
                os.remove(pdf_path)
                logger.info(f'Deleted source PDF: {pdf_path}')
            except Exception as e:
                logger.error(f'Failed to delete {pdf_path}: {e}')
    else:
        logger.error(f'No output for {pdf_path}')

def convert_folder(input_dir: str, output_dir: str, concat: bool = False, delete_src: bool = False):
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    os.makedirs(output_dir, exist_ok=True)
    all_md = ''
    pbar = tqdm(pdf_files, desc='Converting PDFs', unit='file')
    for pdf_file in pbar:
        in_path = os.path.join(input_dir, pdf_file)
        out_path = os.path.join(output_dir, os.path.splitext(pdf_file)[0] + '.md')
        pages = process_pdf(in_path)
        if pages:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("\n---\n".join(pages))
            if concat:
                all_md += f"\n\n# {pdf_file}\n" + "\n---\n".join(pages)
            if delete_src:
                try:
                    os.remove(in_path)
                    logger.info(f'Deleted source PDF: {in_path}')
                except Exception as e:
                    logger.error(f'Failed to delete {in_path}: {e}')
        else:
            logger.error(f'No output for {pdf_file}')
    if concat and all_md:
        concat_path = os.path.join(output_dir, 'concatenated.md')
        with open(concat_path, 'w', encoding='utf-8') as f:
            f.write(all_md.strip())
        logger.info(f'Concatenated markdown written to {concat_path}')

# CLI interaction and interpretation
#
# TODO: I don't like this one still. Make this more easily interactive,
# maybe put like a web interface on top of this like for any_to_any.py
def main():
    parser = argparse.ArgumentParser(description='PDFine: Fast, Accurate PDF to Markdown Conversion (modular)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='Convert a single PDF file to markdown')
    group.add_argument('-i', '--input', help='Convert all PDF files in a folder to markdown')
    parser.add_argument('-o', '--output', help='Output directory for markdown files (default: ./output)', default='output')
    parser.add_argument('-c', '--concat', action='store_true', help='Concatenate all markdown files into a single file')
    parser.add_argument('-d', '--delete', action='store_true', help='Delete source PDF file(s) after conversion')
    parser.add_argument('-m', '--model-dir', help='Directory of trained T5 model for end-to-end Markdown generation')
    args = parser.parse_args()
    global MODEL_DIR
    MODEL_DIR = args.model_dir
    if args.file:
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.file))[0]
        out_path = os.path.join(out_dir, base + '.md')
        convert_file(args.file, out_path, delete_src=args.delete)
    elif args.input:
        convert_folder(args.input, args.output, args.concat, delete_src=args.delete)

if __name__ == '__main__':
    main()
