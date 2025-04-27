# PDFine

**The universal PDF extraction engine.**

Fast, robust, and intelligent PDF to Markdown conversion—works on any PDF you throw at it.

---

## Features

- **Universal PDF Extraction:** Handles any PDF, including scanned, corrupted, encrypted, or nonstandard files.
- **Parallel Multi-Backend Extraction:** Uses PyMuPDF, pdfminer.six, PyPDF2, and OCR in parallel for maximal recall and speed.
- **Hybrid Page Merging:** Combines and deduplicates text from all sources, aligning by content for best results.
- **Table Extraction:** Extracts tables using `camelot` and `tabula-py` (if installed), outputs as markdown or CSV.
- **Image Extraction + OCR:** Extracts all images, runs OCR, and inserts recognized text inline.
- **Metadata, Bookmarks, Annotations:** Extracts and includes document metadata, bookmarks, and annotations in the output.
- **Corruption Recovery:** Skips unreadable pages, never fails the whole document—always returns as much as possible.
- **Rich Output:** Each page is a dict with `text`, `sources`, `page_num`, `type`, and may include tables, images, and meta info.
- **Fast & Efficient:** All extraction strategies run concurrently for speed.

---

## Requirements

- Python 3.8+
- Install dependencies from `requirements.txt`
- For best results, also install `camelot` and `tabula-py` (optional, for table extraction)

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

---

## License & Commercial Usage

PDFine is open for research and personal use, and we encourage wide adoption!
However, there are some restrictions on commercial usage:

- This repository is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International).
- **Commercial use is permitted** for organizations with **less than $5M USD** in gross revenue (last 12 months) **and** less than $5M USD in total VC/angel funding raised.
- If your organization exceeds these thresholds, or if you wish to obtain a commercial or dual license, please contact [mk2112@protonmail.com](mailto:mk2112@protonmail.com)
- You may not use PDFine in products or services that directly compete with PDFine or its hosted offerings, unless you have a separate commercial agreement.

For further details or to discuss commercial licensing, email mk2112@protonmail.com.
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