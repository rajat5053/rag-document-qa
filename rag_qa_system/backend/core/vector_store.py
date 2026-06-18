"""
FAISS vector store module for the RAG Document Q&A system.

Manages index construction, persistence, loading, and similarity search.
Uses ``IndexFlatIP`` (inner product) combined with L2-normalised embeddings
to perform cosine-similarity search efficiently without an external service.

Files persisted to disk:
    data/faiss_index.bin  — the serialised FAISS index
    data/chunks.pkl       — the corresponding list of raw chunk strings
"""

import logging
import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 1024  # BAAI/bge-large-en-v1.5 output dimension

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
_CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Create an in-memory FAISS ``IndexFlatIP`` from a set of embeddings.

    Embeddings are L2-normalised before insertion so that inner-product
    search is equivalent to cosine-similarity search.

    Args:
        embeddings: NumPy array of shape ``(N, 1024)``, dtype ``float32``.

    Returns:
        A populated ``faiss.IndexFlatIP`` index containing *N* vectors.

    Raises:
        ValueError: If the embedding array is empty or has the wrong shape.
    """
    if embeddings.ndim != 2 or embeddings.shape[1] != _EMBEDDING_DIM:
        raise ValueError(
            f"Expected embeddings of shape (N, {_EMBEDDING_DIM}), "
            f"got {embeddings.shape}."
        )
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot build a FAISS index from zero embeddings.")

    # L2-normalise so that inner product == cosine similarity
    normed = embeddings.copy()
    faiss.normalize_L2(normed)

    index = faiss.IndexFlatIP(_EMBEDDING_DIM)
    index.add(normed)
    logger.info("Built FAISS index with %d vectors.", index.ntotal)
    return index


def save_index(index: faiss.IndexFlatIP, chunks: List[str]) -> None:
    """
    Persist the FAISS index and corresponding chunk list to disk.

    Args:
        index:  A populated ``faiss.IndexFlatIP`` index.
        chunks: List of raw chunk strings whose positions correspond to the
                vectors stored in *index*.

    Raises:
        OSError: If the data directory cannot be created or files cannot be
                 written.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, _INDEX_PATH)
    with open(_CHUNKS_PATH, "wb") as fh:
        pickle.dump(chunks, fh)
    logger.info(
        "Saved FAISS index to '%s' and %d chunks to '%s'.",
        _INDEX_PATH,
        len(chunks),
        _CHUNKS_PATH,
    )


def load_index() -> Tuple[faiss.IndexFlatIP, List[str]]:
    """
    Load the persisted FAISS index and chunk list from disk.

    Returns:
        A tuple ``(index, chunks)`` where *index* is the loaded
        ``faiss.IndexFlatIP`` and *chunks* is the list of raw chunk strings.

    Raises:
        FileNotFoundError: If either the index file or the chunks file does
                           not exist (i.e. no document has been ingested yet).
    """
    if not os.path.exists(_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{_INDEX_PATH}'. "
            "Please ingest a document first via POST /ingest."
        )
    if not os.path.exists(_CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at '{_CHUNKS_PATH}'. "
            "Please ingest a document first via POST /ingest."
        )

    index = faiss.read_index(_INDEX_PATH)
    with open(_CHUNKS_PATH, "rb") as fh:
        chunks: List[str] = pickle.load(fh)

    logger.info(
        "Loaded FAISS index (%d vectors) and %d chunks from disk.",
        index.ntotal,
        len(chunks),
    )
    return index, chunks


def search(
    query_embedding: np.ndarray,
    index: faiss.IndexFlatIP,
    chunks: List[str],
    k: int = 5,
) -> List[str]:
    """
    Retrieve the top-*k* most similar chunks for a query embedding.

    The query embedding is L2-normalised before the search so that it is
    consistent with the normalised document vectors stored in the index.

    Args:
        query_embedding: NumPy array of shape ``(1, 1024)``, dtype ``float32``.
        index:           Populated ``faiss.IndexFlatIP`` index.
        chunks:          List of raw chunk strings aligned with index vectors.
        k:               Number of nearest neighbours to retrieve (default 5).

    Returns:
        List of up to *k* chunk strings, ordered from most to least similar.
        May be shorter than *k* if the index contains fewer than *k* vectors.
    """
    actual_k = min(k, index.ntotal)
    if actual_k == 0:
        return []

    normed_q = query_embedding.copy().astype(np.float32)
    faiss.normalize_L2(normed_q)

    distances, indices = index.search(normed_q, actual_k)
    result_indices = indices[0].tolist()

    retrieved: List[str] = []
    for idx in result_indices:
        if 0 <= idx < len(chunks):
            retrieved.append(chunks[idx])

    return retrieved
