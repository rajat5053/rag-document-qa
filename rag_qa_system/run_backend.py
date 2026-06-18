"""
Backend launcher script for the RAG Document Q&A system.

Run this file from the rag_qa_system/ directory instead of calling
uvicorn directly. It ensures the correct working directory is on
sys.path so that `import backend.*` always resolves correctly.

Usage:
    python run_backend.py
"""

import os
import sys


def main() -> None:
    """
    Add the project root to sys.path and start the Uvicorn server.

    This is equivalent to running:
        uvicorn backend.main:app --reload --port 8000
    from inside the rag_qa_system/ directory.
    """
    # Ensure the directory containing this script (rag_qa_system/) is on the path.
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[project_root],
    )


if __name__ == "__main__":
    main()
