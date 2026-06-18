"""
Document ingestion API endpoint for the RAG Document Q&A system.

Exposes POST /ingest which accepts a PDF file upload, extracts and chunks
the text, generates embeddings, builds a FAISS vector index, and persists
everything to disk so that subsequent /ask requests can retrieve relevant
context.
"""

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.core.embeddings import embed_texts
from backend.core.pdf_parser import parse_and_chunk_pdf
from backend.core.vector_store import build_index, save_index

logger = logging.getLogger(__name__)

router = APIRouter()

# Define paths for metadata persistence
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(os.path.dirname(_BACKEND_DIR), "data")
_METADATA_FILE = os.path.join(_DATA_DIR, "metadata.json")
_INDEX_FILE = os.path.join(_DATA_DIR, "faiss_index.bin")


@router.post(
    "/ingest",
    summary="Ingest a PDF document",
    description=(
        "Upload a PDF file to extract, chunk, embed, and index its content. "
        "The FAISS index is persisted to disk and replaces any previously "
        "ingested document."
    ),
    response_description="Number of text chunks ingested into the vector store.",
)
async def ingest_document(file: UploadFile = File(...)) -> dict:
    """
    Accept a PDF upload and run the full ingestion pipeline.

    Pipeline steps:
        1. Validate that the uploaded file is a PDF.
        2. Read the raw bytes from the upload.
        3. Parse the PDF and chunk the text (150 words / 50-word overlap).
        4. Generate BAAI/bge-large-en-v1.5 embeddings for every chunk.
        5. Build a FAISS IndexFlatIP and persist it alongside the chunk list.
        6. Persist metadata (filename, chunks count, and ingestion time) to disk.

    Args:
        file: The multipart form-data file upload.  Must be a PDF.

    Returns:
        JSON body ``{"status": "success", "chunks_ingested": N}``.

    Raises:
        HTTPException 400: If the uploaded file is not a PDF.
        HTTPException 500: If any processing step fails.
    """
    # ---- 1. Validate file type ------------------------------------------
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        # Also check filename extension as some clients send wrong MIME types
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Uploaded file '{file.filename}' is not a PDF. "
                    "Only PDF files are accepted."
                ),
            )

    # ---- 2. Read bytes -------------------------------------------------
    try:
        pdf_bytes = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file.")
        raise HTTPException(
            status_code=500,
            detail=f"Could not read uploaded file: {exc}",
        ) from exc

    # ---- 3. Parse and chunk ---------------------------------------------
    try:
        chunks = parse_and_chunk_pdf(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF parsing failed.")
        raise HTTPException(
            status_code=500,
            detail=f"PDF parsing failed: {exc}",
        ) from exc

    logger.info("Parsed %d chunks from '%s'.", len(chunks), file.filename)

    # ---- 4. Embed -------------------------------------------------------
    try:
        embeddings = embed_texts(chunks)
    except Exception as exc:
        logger.exception("Embedding generation failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {exc}",
        ) from exc

    # ---- 5. Build and save FAISS index ----------------------------------
    try:
        index = build_index(embeddings)
        save_index(index, chunks)
    except Exception as exc:
        logger.exception("FAISS index build/save failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Vector index build failed: {exc}",
        ) from exc

    # ---- 6. Save metadata to disk ---------------------------------------
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        metadata = {
            "filename": file.filename,
            "chunks_count": len(chunks),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("Saved ingestion metadata for '%s' to disk.", file.filename)
    except Exception as exc:
        # Log but do not fail the request — vector store was saved successfully
        logger.warning("Failed to save ingestion metadata to disk: %s", exc)

    return {"status": "success", "chunks_ingested": len(chunks)}


@router.get(
    "/ingest/status",
    summary="Get ingestion status",
    description="Check whether a document is currently ingested and get its details.",
)
async def get_ingestion_status() -> dict:
    """
    Check if a FAISS index and metadata exist on disk and return current status.

    Returns:
        A dict containing:
            - ``ingested``: True if a valid index is active, False otherwise.
            - ``filename``: Name of the active PDF (if ingested).
            - ``chunks_count``: Number of index chunks (if ingested).
            - ``ingested_at``: ISO-format UTC timestamp (if ingested).
    """
    if not os.path.exists(_INDEX_FILE) or not os.path.exists(_METADATA_FILE):
        return {"ingested": False}

    try:
        with open(_METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return {
            "ingested": True,
            "filename": metadata.get("filename", "Unknown Document"),
            "chunks_count": metadata.get("chunks_count", 0),
            "ingested_at": metadata.get("ingested_at", ""),
        }
    except Exception as exc:
        logger.warning("Failed to read ingestion metadata from disk: %s", exc)
        # Fallback if metadata file is corrupted but index exists
        return {
            "ingested": True,
            "filename": "Ingested Document (Metadata missing)",
            "chunks_count": 0,
            "ingested_at": "",
        }

