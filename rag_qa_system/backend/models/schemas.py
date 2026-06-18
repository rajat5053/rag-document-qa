"""
Pydantic schemas for the RAG Document Q&A API.

Defines request and response models used across all endpoints,
ensuring consistent data validation and serialization throughout
the application.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class AskRequest(BaseModel):
    """
    Request body for the POST /ask endpoint.

    Attributes:
        query: The natural-language question the user wants answered
               from the ingested document.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="The question to answer using the ingested document.",
        examples=["What are the main findings of the document?"],
    )


class AskResponse(BaseModel):
    """
    Response body returned by the POST /ask endpoint.

    Attributes:
        answer:      The LLM-generated answer grounded in retrieved chunks.
        sources:     List of raw chunk strings used as context for the answer.
        latency_ms:  End-to-end pipeline latency in milliseconds.
    """

    answer: str = Field(..., description="LLM-generated answer from the document context.")
    sources: List[str] = Field(
        default_factory=list,
        description="Retrieved document chunks used as context.",
    )
    latency_ms: float = Field(..., description="Total pipeline latency in milliseconds.")


class TopQuestion(BaseModel):
    """
    A single entry in the top-questions analytics list.

    Attributes:
        query: The question text.
        count: How many times it was asked.
    """

    query: str
    count: int


class UnansweredQuery(BaseModel):
    """
    A single entry in the unanswered-queries analytics list.

    Attributes:
        query:     The question text.
        timestamp: ISO-format timestamp of when it was asked.
    """

    query: str
    timestamp: str


class AnalyticsResponse(BaseModel):
    """
    Response body returned by the GET /analytics endpoint.

    Attributes:
        top_questions:      Most frequently asked questions, sorted by count descending.
        unanswered_queries: Queries where the LLM could not find an answer in the document.
        avg_latency_ms:     Average end-to-end response latency across all logged queries.
        total_queries:      Total number of queries logged in the system.
        answered_queries:   Number of queries successfully answered.
    """

    top_questions: List[TopQuestion] = Field(
        default_factory=list,
        description="Most frequently asked questions.",
    )
    unanswered_queries: List[UnansweredQuery] = Field(
        default_factory=list,
        description="Queries that had no grounded answer in the document.",
    )
    avg_latency_ms: Optional[float] = Field(
        None,
        description="Average pipeline latency in milliseconds. None if no queries logged yet.",
    )
    total_queries: int = Field(0, description="Total number of queries logged.")
    answered_queries: int = Field(0, description="Number of queries successfully answered.")
