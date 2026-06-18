"""
Question-answering API endpoint for the RAG Document Q&A system.

Exposes POST /ask which accepts a JSON body with a ``query`` field, runs the
full RAG pipeline (embed → retrieve → LLM), logs every call to SQLite, and
returns the answer together with the source chunks and latency metadata.
"""

import logging
import os

from fastapi import APIRouter, HTTPException

from backend.core.rag_pipeline import run_pipeline
from backend.db.database import log_query
from backend.models.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Path to the FAISS index file — used to detect whether a document has been
# ingested without actually loading the (potentially large) index into memory.
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
_INDEX_FILE = os.path.join(_DATA_DIR, "faiss_index.bin")


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about the ingested document",
    description=(
        "Submit a natural-language question. The system retrieves the most "
        "relevant document chunks via FAISS and generates an answer using "
        "Llama-3.1-8B-Instruct on NVIDIA NIM."
    ),
)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Run the RAG pipeline for *request.query* and return the answer.

    Validations performed before running the pipeline:
        - Query must not be blank / whitespace-only.
        - A document must have been ingested (FAISS index file must exist).

    All calls — including failures — are logged to SQLite for analytics.

    Args:
        request: JSON body containing the ``query`` field.

    Returns:
        AskResponse with ``answer``, ``sources``, and ``latency_ms``.

    Raises:
        HTTPException 400: If the query is empty or whitespace-only.
        HTTPException 503: If no document has been ingested yet.
        HTTPException 500: If the pipeline fails for any other reason.
    """
    # ---- 1. Validate query ---------------------------------------------
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty or whitespace-only.",
        )

    # ---- 2. Check that a document has been ingested --------------------
    if not os.path.exists(_INDEX_FILE):
        raise HTTPException(
            status_code=503,
            detail=(
                "No document has been ingested yet. "
                "Please upload a PDF via POST /ingest before asking questions."
            ),
        )

    # ---- 3. Run pipeline -----------------------------------------------
    try:
        result = run_pipeline(request.query.strip())
    except EnvironmentError as exc:
        logger.error("Missing NVIDIA_API_KEY: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("RAG pipeline runtime error.")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in RAG pipeline.")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {exc}",
        ) from exc

    # ---- 4. Log to SQLite ----------------------------------------------
    try:
        log_query(
            query=request.query.strip(),
            answer=result["answer"],
            sources=result["sources"],
            latency_ms=result["latency_ms"],
            answer_found=result["answer_found"],
        )
    except Exception as exc:
        # Log but do not fail the request — the user still gets their answer
        logger.warning("Failed to log query to SQLite: %s", exc)

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
    )
