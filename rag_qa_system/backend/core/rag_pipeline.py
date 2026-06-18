"""
RAG (Retrieval-Augmented Generation) pipeline for the Document Q&A system.

Orchestrates the end-to-end query flow:
    1. Embed the user query with BAAI/bge-large-en-v1.5 (local model).
    2. Retrieve the top-k most relevant chunks from the FAISS index.
    3. Build a prompt that constrains the LLM to answer only from the context.
    4. Call the NVIDIA NIM API (OpenAI-compatible) with Llama-3.1-8B-Instruct.
    5. Return the answer, source chunks, a boolean answer_found flag, and
       the end-to-end latency in milliseconds.

NVIDIA NIM API endpoint:
    https://integrate.api.nvidia.com/v1/chat/completions
    Requires NVIDIA_API_KEY set in .env
"""

import logging
import os
import time
from typing import Any, Dict

import requests
from dotenv import load_dotenv

from backend.core.embeddings import embed_query
from backend.core.vector_store import load_index, search

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_NIM_MODEL = "meta/llama-3.1-8b-instruct"

_NO_ANSWER_SENTINEL = "I could not find an answer to that in the provided document."

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer ONLY from the context below. "
    "If the answer is not in the context, say exactly: "
    f"'{_NO_ANSWER_SENTINEL}'"
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(query: str) -> Dict[str, Any]:
    """
    Execute the full RAG pipeline for a single user query.

    Steps:
        1. Record the wall-clock start time.
        2. Load the persisted FAISS index and chunk list.
        3. Embed the query (with BGE instruction prefix).
        4. Retrieve the top-5 most similar chunks.
        5. Build an OpenAI-compatible chat message list for Llama via NVIDIA NIM.
        6. POST to the NVIDIA NIM API.
        7. Parse the generated text and determine answer_found.
        8. Return result dict including latency_ms.

    Args:
        query: The user's natural-language question.

    Returns:
        A dict with keys:
            - ``answer``       (str):   LLM-generated answer.
            - ``sources``      (list):  Retrieved chunk strings used as context.
            - ``answer_found`` (bool):  False if the sentinel phrase was returned.
            - ``latency_ms``   (float): End-to-end pipeline latency in milliseconds.

    Raises:
        FileNotFoundError: If no document has been ingested (FAISS index missing).
        RuntimeError:      If the NVIDIA NIM API call fails or returns an error.
        EnvironmentError:  If NVIDIA_API_KEY environment variable is not set.
    """
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise EnvironmentError(
            "NVIDIA_API_KEY environment variable is not set. "
            "Add it to your .env file: NVIDIA_API_KEY=nvapi-xxxxxxxx"
        )

    t_start = time.time()

    # ── 1. Load FAISS index (raises FileNotFoundError if not ingested) ──
    index, chunks = load_index()

    # ── 2. Embed the query ──────────────────────────────────────────────
    query_embedding = embed_query(query)

    # ── 3. Retrieve top-k chunks ────────────────────────────────────────
    top_chunks = search(query_embedding, index, chunks, k=5)
    joined_context = "\n\n---\n\n".join(top_chunks)

    # ── 4. Build OpenAI-compatible messages ─────────────────────────────
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{joined_context}\n\nQuestion: {query}",
        },
    ]

    # ── 5. Call NVIDIA NIM API ──────────────────────────────────────────
    headers = {
        "Authorization": f"Bearer {nvidia_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": _NIM_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.95,
        "stream": False,
    }

    try:
        response = requests.post(
            _NIM_API_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "NVIDIA NIM API timed out after 120 seconds."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"NVIDIA NIM API returned HTTP {response.status_code}: {response.text}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Network error communicating with NVIDIA NIM API: {exc}"
        ) from exc

    # ── 6. Parse response (OpenAI-compatible format) ────────────────────
    response_data = response.json()

    if "error" in response_data:
        raise RuntimeError(f"NVIDIA NIM API error: {response_data['error']}")

    try:
        answer: str = (
            response_data["choices"][0]["message"]["content"].strip()
        )
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected NVIDIA NIM API response structure: {response_data}"
        ) from exc

    # ── 7. Determine if a grounded answer was found ─────────────────────
    answer_found: bool = _NO_ANSWER_SENTINEL.lower() not in answer.lower()

    latency_ms = (time.time() - t_start) * 1000.0

    logger.info(
        "Pipeline completed in %.1f ms | answer_found=%s", latency_ms, answer_found
    )

    return {
        "answer": answer,
        "sources": top_chunks,
        "answer_found": answer_found,
        "latency_ms": round(latency_ms, 2),
    }
