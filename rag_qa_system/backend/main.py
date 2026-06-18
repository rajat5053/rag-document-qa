"""
FastAPI application entry point for the RAG Document Q&A system.

Assembles the application by:
    - Registering all three API routers (ingest, ask, analytics).
    - Adding CORS middleware to allow the Streamlit frontend to call the API
      without cross-origin restrictions.
    - Initialising the SQLite database tables on startup.
    - Exposing a GET / health-check endpoint.

Usage:
    uvicorn backend.main:app --reload --port 8000
"""

import logging

from dotenv import load_dotenv

# Load .env from the project root (rag_qa_system/.env) before anything else
load_dotenv()

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from backend.api.analytics import router as analytics_router
from backend.api.ask import router as ask_router
from backend.api.ingest import router as ingest_router
from backend.db.database import init_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Document Q&A API",
    description=(
        "A Retrieval-Augmented Generation (RAG) API that lets you upload PDF "
        "documents and ask natural-language questions answered by "
        "Llama-3.1-8B-Instruct via the NVIDIA NIM API."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """
    Run once when the FastAPI application starts.

    Initialises the SQLite ``query_logs`` table if it does not already exist,
    ensuring the database is ready to accept writes before any request arrives.
    """
    logger.info("Initialising SQLite database …")
    init_db()
    logger.info("Database ready.")


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(ingest_router, tags=["Ingestion"])
app.include_router(ask_router, tags=["Q&A"])
app.include_router(analytics_router, tags=["Analytics"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"], summary="Health check")
async def health_check() -> dict:
    """
    Return a simple health-check payload to confirm the API is running.

    Returns:
        ``{"status": "ok", "message": "RAG Document Q&A API is running."}``.
    """
    return {"status": "ok", "message": "RAG Document Q&A API is running."}
