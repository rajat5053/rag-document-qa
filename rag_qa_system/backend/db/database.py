"""
SQLite database module for the RAG Document Q&A system.

Handles table initialisation, connection management, and query log
insertion. All persistent query data is stored in data/rag_logs.db
so that the analytics endpoint can surface usage statistics without
any external database dependency.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "rag_logs.db")

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_connection() -> sqlite3.Connection:
    """
    Open and return a sqlite3 connection to the RAG logs database.

    The connection uses ``check_same_thread=False`` to be compatible with
    FastAPI's async request handling where the connection may be used
    across threads.

    Returns:
        sqlite3.Connection: An open connection to ``data/rag_logs.db``.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # allow column-name access
    return conn


def init_db() -> None:
    """
    Initialise the ``query_logs`` table if it does not already exist.

    Schema:
        - id          INTEGER PRIMARY KEY AUTOINCREMENT
        - query       TEXT    NOT NULL
        - answer      TEXT
        - sources     TEXT    (JSON-serialised list of chunk strings)
        - latency_ms  REAL
        - answer_found INTEGER (0 or 1)
        - timestamp   TEXT    (ISO-8601 UTC)

    This function is called once at application startup from the FastAPI
    lifespan/startup event so that the table is always present before any
    request is handled.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                query         TEXT    NOT NULL,
                answer        TEXT,
                sources       TEXT,
                latency_ms    REAL,
                answer_found  INTEGER,
                timestamp     TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_query(
    query: str,
    answer: str,
    sources: list,
    latency_ms: float,
    answer_found: bool,
) -> None:
    """
    Insert a single query log record into the ``query_logs`` table.

    Args:
        query:        The user's original question string.
        answer:       The LLM-generated answer returned to the user.
        sources:      List of retrieved chunk strings used as context.
        latency_ms:   End-to-end pipeline duration in milliseconds.
        answer_found: ``True`` if the LLM found a grounded answer;
                      ``False`` if it reported the answer was not in the document.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO query_logs (query, answer, sources, latency_ms, answer_found, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                query,
                answer,
                json.dumps(sources),
                latency_ms,
                1 if answer_found else 0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
