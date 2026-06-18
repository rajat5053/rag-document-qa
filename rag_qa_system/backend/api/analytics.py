"""
Analytics API endpoint for the RAG Document Q&A system.

Exposes GET /analytics which aggregates usage data from the SQLite
``query_logs`` table and returns top questions, unanswered queries, and
average pipeline latency without requiring any external analytics service.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.db.database import get_connection
from backend.db.queries import (
    get_avg_latency,
    get_top_questions,
    get_unanswered_queries,
    get_total_queries,
    get_answered_queries,
)
from backend.models.schemas import AnalyticsResponse, TopQuestion, UnansweredQuery

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Retrieve usage analytics",
    description=(
        "Returns aggregated usage statistics from the query log: "
        "most frequently asked questions, unanswered queries, and "
        "average end-to-end response latency."
    ),
)
async def get_analytics() -> AnalyticsResponse:
    """
    Aggregate and return query usage analytics from SQLite.

    Queries:
        - ``top_questions``     — GROUP BY query, ORDER BY count DESC, LIMIT 10.
        - ``unanswered_queries``— WHERE answer_found = 0.
        - ``avg_latency_ms``    — AVG(latency_ms) across all rows.
        - ``total_queries``     — COUNT(*) across all rows.
        - ``answered_queries``  — COUNT(*) where answer_found = 1.

    Returns:
        AnalyticsResponse containing all analytics sections.

    Raises:
        HTTPException 500: If the database query fails.
    """
    try:
        conn = get_connection()
        try:
            raw_top = get_top_questions(conn, limit=10)
            raw_unanswered = get_unanswered_queries(conn)
            avg_lat = get_avg_latency(conn)
            total_q = get_total_queries(conn)
            answered_q = get_answered_queries(conn)
        finally:
            conn.close()
    except Exception as exc:
        logger.exception("Failed to retrieve analytics from SQLite.")
        raise HTTPException(
            status_code=500,
            detail=f"Analytics query failed: {exc}",
        ) from exc

    top_questions = [TopQuestion(query=r["query"], count=r["count"]) for r in raw_top]
    unanswered_queries = [
        UnansweredQuery(query=r["query"], timestamp=r["timestamp"])
        for r in raw_unanswered
    ]

    return AnalyticsResponse(
        top_questions=top_questions,
        unanswered_queries=unanswered_queries,
        avg_latency_ms=avg_lat,
        total_queries=total_q,
        answered_queries=answered_q,
    )
