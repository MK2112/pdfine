import sys
import logging

"""
Common exceptions and logger configuration go here
"""

class PDFParseError(Exception):
    """Utilized when PDF parsing fails."""
    pass

# Shared logger
logger = logging.getLogger('PDFine')

# Ensure logger has console handler
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
