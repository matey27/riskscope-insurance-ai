"""
utils/pdf_extractor.py
======================
Responsible for reading PDF files and returning plain text.

Strategy:
  1. Try pdfplumber first (handles complex layouts, tables, columns).
  2. Fall back to PyPDF2 if pdfplumber fails.
  3. Return a tuple (text, error_message) so the caller can handle failures
     gracefully without crashing the app.
"""

from __future__ import annotations
import io
from typing import Tuple

# ── pdfplumber ─────────────────────────────────────────────────────────────────
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

# ── PyPDF2 fallback ────────────────────────────────────────────────────────────
try:
    import PyPDF2
    _HAS_PYPDF2 = True
except ImportError:
    _HAS_PYPDF2 = False


def extract_text_from_pdf(uploaded_file) -> Tuple[str, str | None]:
    """
    Extract all text from an uploaded Streamlit PDF file.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The file object from st.file_uploader.

    Returns
    -------
    (text, error)
        text  – full extracted text string (empty string on failure)
        error – human-readable error message, or None on success
    """
    if uploaded_file is None:
        return "", "No file provided."

    # Read raw bytes once; reuse for both extractors.
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)          # reset so the caller can re-read if needed

    # ── Attempt 1: pdfplumber ──────────────────────────────────────────────────
    if _HAS_PDFPLUMBER:
        try:
            text = _extract_with_pdfplumber(raw_bytes)
            if text.strip():
                return text, None
            # pdfplumber returned nothing – fall through to PyPDF2
        except Exception as exc:
            pass  # will try PyPDF2 next

    # ── Attempt 2: PyPDF2 ─────────────────────────────────────────────────────
    if _HAS_PYPDF2:
        try:
            text = _extract_with_pypdf2(raw_bytes)
            if text.strip():
                return text, None
        except Exception as exc:
            return "", f"PyPDF2 extraction failed: {exc}"

    # ── Nothing worked ────────────────────────────────────────────────────────
    if not _HAS_PDFPLUMBER and not _HAS_PYPDF2:
        return "", (
            "No PDF library found. Install pdfplumber or PyPDF2:\n"
            "  pip install pdfplumber PyPDF2"
        )

    return "", "Could not extract text (PDF may be image-based or encrypted)."


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract_with_pdfplumber(raw_bytes: bytes) -> str:
    """Use pdfplumber to extract text page-by-page."""
    pages = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n\n".join(pages)


def _extract_with_pypdf2(raw_bytes: bytes) -> str:
    """Use PyPDF2 to extract text page-by-page."""
    pages = []
    reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages)
