# PDFine

Fast, accurate PDF to Markdown conversion.

## Requirements

- Python 3.8+
- Install dependencies from `requirements.txt`
- Input PDFs must be extractable by `pdfine.extractor.raw_extract_pages` (each page as a dict with a non-empty `'text'` string)

## Usage

**Single PDF:**
```bash
python PDFine.py -f input.pdf -o outdir
```

**Batch folder:**
```bash
python PDFine.py -i input_folder -o outdir
```

Options:
- `-f, --file` — Single PDF file to convert
- `-i, --input` — Folder of PDFs to convert
- `-o, --output` — Output directory (default: `./output`)
- `-c, --concat` — Concatenate all markdown files into one
- `-d, --delete` — Delete source PDFs after conversion
- `-m, --model-dir` — Directory of trained T5 model for end-to-end conversion

## Training

Train a model:
```bash
python train.py --dataset-name <dataset> --seq-model <t5-model> --world-model <gpt2-model> --output-dir trained_model
```
See `train.py` for all options.

## Notes

- Only PDFs compatible with `raw_extract_pages` are supported. If extraction fails or produces empty/invalid output, conversion is skipped and an error is logged.
- All errors are logged. Check logs for diagnostics.
- See `tests/` for test coverage. Run tests with:
  ```bash
  pytest tests/
  ```