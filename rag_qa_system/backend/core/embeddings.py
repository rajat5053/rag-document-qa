"""
Embedding generation module for the RAG Document Q&A system.

Loads the BAAI/bge-large-en-v1.5 sentence-transformer model once at import
time (singleton pattern) to avoid repeated model initialisation on every
request.  The model is loaded from the local path specified in the
``EMBEDDING_MODEL_PATH`` environment variable (set in .env).

Device selection
----------------
By default the model runs on **CPU** to avoid ``cudaErrorNoKernelImageForDevice``
errors that occur when the installed PyTorch CUDA kernels do not match the
GPU's compute capability.  Set ``EMBEDDING_DEVICE=cuda`` in .env only after
confirming your PyTorch build matches your CUDA version
(``python -c "import torch; print(torch.version.cuda)"``).

BGE models require a task-specific instruction prefix to be prepended to the
*query* at inference time.  Document chunks must NOT receive this prefix.
"""

import logging
import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve model path
# ---------------------------------------------------------------------------

_MODEL_PATH: str = os.environ.get("EMBEDDING_MODEL_PATH", "BAAI/bge-large-en-v1.5")

if _MODEL_PATH == "BAAI/bge-large-en-v1.5":
    logger.warning(
        "EMBEDDING_MODEL_PATH is not set in .env — "
        "falling back to downloading from HuggingFace Hub."
    )
else:
    if not os.path.isdir(_MODEL_PATH):
        raise RuntimeError(
            f"EMBEDDING_MODEL_PATH='{_MODEL_PATH}' does not exist or is not a directory. "
            "Please set the correct path in your .env file."
        )
    logger.info("Using local embedding model at: %s", _MODEL_PATH)

# BGE models require this prefix on the query side only.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

# Default to CPU to avoid CUDA kernel mismatch errors.
# Override by setting EMBEDDING_DEVICE=cuda in .env once PyTorch/CUDA match.
_DEVICE: str = os.environ.get("EMBEDDING_DEVICE", "cpu").lower()
logger.info("Embedding device: %s", _DEVICE)

# ---------------------------------------------------------------------------
# Lazy-loaded model instance
# ---------------------------------------------------------------------------

_model = None


def _get_model() -> SentenceTransformer:
    """
    Get or load the singleton SentenceTransformer instance.

    Loads the model from disk/Hub only on the first call to keep imports fast
    and prevent Uvicorn reload processes from blocking incoming status requests.
    """
    global _model
    if _model is None:
        logger.info("Lazy-loading sentence-transformer model from: %s …", _MODEL_PATH)
        _model = SentenceTransformer(_MODEL_PATH, device=_DEVICE)
        logger.info("Embedding model loaded successfully on device: %s", _DEVICE)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of document chunk strings into dense vectors.

    No task-instruction prefix is applied — BGE documentation specifies that
    only the query side requires the prefix; passage/document embeddings should
    be encoded as-is.

    Args:
        texts: List of plain-text chunk strings to embed.

    Returns:
        NumPy array of shape ``(N, 1024)`` where N = len(texts), dtype float32.

    Raises:
        ValueError: If *texts* is empty.
    """
    if not texts:
        raise ValueError("embed_texts received an empty list of texts.")

    model = _get_model()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=False,  # normalisation handled in vector_store
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query string, prepending the BGE task instruction.

    The BAAI/bge-large-en-v1.5 model was fine-tuned for retrieval with this
    specific instruction prefix on the query side, which measurably improves
    recall compared to encoding the query as a plain string.

    Args:
        query: The user's natural-language question.

    Returns:
        NumPy array of shape ``(1, 1024)``, dtype float32.

    Raises:
        ValueError: If *query* is blank or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("embed_query received an empty query string.")

    instructed_query = _QUERY_INSTRUCTION + query.strip()
    model = _get_model()
    embedding: np.ndarray = model.encode(
        [instructed_query],
        show_progress_bar=False,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )
    return embedding.astype(np.float32)
