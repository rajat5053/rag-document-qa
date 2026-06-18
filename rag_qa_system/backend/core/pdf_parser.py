"""
PDF parsing and chunking module for the RAG Document Q&A system.

Uses PyMuPDF (fitz) to extract plain text from every page of an uploaded
PDF, then splits that text into overlapping word-level chunks to preserve
context across chunk boundaries.

Chunking strategy
-----------------
* **Chunk size  = 150 words** – Fits comfortably within the embedding model's
  512-token limit (~200 tokens), preventing truncation of text at the end of
  chunks and ensuring every word is represented in the vector space.
* **Overlap     = 50 words**  – A sliding overlap ensures that sentences or
  key facts that straddle chunk boundaries are preserved in full.
"""

import io
from typing import List

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_and_chunk_pdf(pdf_bytes: bytes, chunk_size: int = 150, overlap: int = 50) -> List[str]:
    """
    Extract text from a PDF byte blob and split it into overlapping word chunks.

    Args:
        pdf_bytes:  Raw bytes of the uploaded PDF file.
        chunk_size: Target number of words per chunk (default 500).
        overlap:    Number of words shared between consecutive chunks (default 50).

    Returns:
        A list of plain-text strings, each containing at most *chunk_size*
        words and overlapping with its neighbours by *overlap* words.

    Raises:
        ValueError: If the PDF contains no extractable text.
        fitz.FileDataError: If the bytes do not represent a valid PDF.
    """
    # Open the PDF from bytes (no temp file needed)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text_parts: List[str] = []
    for page in doc:
        page_text = page.get_text("text")  # plain text, preserves whitespace
        if page_text.strip():
            full_text_parts.append(page_text)

    doc.close()

    if not full_text_parts:
        raise ValueError("No extractable text found in the uploaded PDF.")

    full_text = "\n".join(full_text_parts)
    return _chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split *text* into overlapping word-level chunks.

    Args:
        text:       Full document text as a single string.
        chunk_size: Maximum number of words in each chunk.
        overlap:    Number of words from the end of the previous chunk
                    to prepend to the next chunk.

    Returns:
        List of chunk strings.  The last chunk may be shorter than
        *chunk_size* if the document does not divide evenly.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)  # never step backwards

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += step

    return chunks
