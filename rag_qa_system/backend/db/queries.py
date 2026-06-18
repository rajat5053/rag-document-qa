"""
Analytics SQL query helpers for the RAG Document Q&A system.

All functions accept an open ``sqlite3.Connection`` and return plain Python
structures so that the calling endpoint layer can freely serialise them
into Pydantic models without any ORM coupling.
"""

import sqlite3
from typing import List, Optional


def get_top_questions(conn: sqlite3.Connection, limit: int = 10) -> List[dict]:
    """
    Return the most-frequently-asked queries, sorted by ask count descending.

    Groups all rows in ``query_logs`` by the ``query`` column and counts
    occurrences, returning only the top *limit* entries.

    Args:
        conn:  An open ``sqlite3.Connection`` to ``rag_logs.db``.
        limit: Maximum number of entries to return (default 10).

    Returns:
        List of dicts, each containing:
            - ``query`` (str): The question text.
            - ``count`` (int): Number of times it was asked.
    """
    cursor = conn.execute(
        """
        SELECT query, COUNT(*) AS count
        FROM query_logs
        GROUP BY query
        ORDER BY count DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    return [{"query": row["query"], "count": row["count"]} for row in rows]


def get_unanswered_queries(conn: sqlite3.Connection) -> List[dict]:
    """
    Return all queries where the LLM could not find a grounded answer.

    These are rows where ``answer_found = 0``, indicating the LLM responded
    with the sentinel phrase "I could not find an answer to that in the
    provided document."

    Args:
        conn: An open ``sqlite3.Connection`` to ``rag_logs.db``.

    Returns:
        List of dicts, each containing:
            - ``query``     (str): The question text.
            - ``timestamp`` (str): ISO-format UTC timestamp of the request.
    """
    cursor = conn.execute(
        """
        SELECT query, timestamp
        FROM query_logs
        WHERE answer_found = 0
        ORDER BY timestamp DESC
        """,
    )
    rows = cursor.fetchall()
    return [{"query": row["query"], "timestamp": row["timestamp"]} for row in rows]


def get_avg_latency(conn: sqlite3.Connection) -> Optional[float]:
    """
    Compute the average end-to-end pipeline latency across all logged queries.

    Args:
        conn: An open ``sqlite3.Connection`` to ``rag_logs.db``.

    Returns:
        Average latency in milliseconds as a ``float``, or ``None`` if no
        queries have been logged yet (AVG of an empty set is NULL in SQLite).
    """
    cursor = conn.execute("SELECT AVG(latency_ms) AS avg_lat FROM query_logs")
    row = cursor.fetchone()
    if row is None or row["avg_lat"] is None:
        return None
    return float(row["avg_lat"])


def get_total_queries(conn: sqlite3.Connection) -> int:
    """
    Get the total number of logged queries in the database.

    Args:
        conn: An open ``sqlite3.Connection`` to ``rag_logs.db``.

    Returns:
        The total integer count of queries.
    """
    cursor = conn.execute("SELECT COUNT(*) AS total FROM query_logs")
    row = cursor.fetchone()
    return row["total"] if row else 0


def get_answered_queries(conn: sqlite3.Connection) -> int:
    """
    Get the number of queries successfully answered by the LLM (answer_found = 1).

    Args:
        conn: An open ``sqlite3.Connection`` to ``rag_logs.db``.

    Returns:
        The integer count of answered queries.
    """
    cursor = conn.execute("SELECT COUNT(*) AS answered FROM query_logs WHERE answer_found = 1")
    row = cursor.fetchone()
    return row["answered"] if row else 0
