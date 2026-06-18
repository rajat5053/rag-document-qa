# RAG Document Q&A System

> A production-ready Retrieval-Augmented Generation (RAG) pipeline that lets you upload PDF documents and ask natural-language questions answered by **Llama-3.1-8B-Instruct** via the NVIDIA NIM API — entirely free, no paid API keys required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT FRONTEND                         │
│   Sidebar: PDF Upload   │   Tab 1: Ask Q   │   Tab 2: Analytics │
└──────────────┬──────────────────┬──────────────────┬────────────┘
               │ POST /ingest     │ POST /ask        │ GET /analytics
               ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FASTAPI BACKEND                           │
│                                                                 │
│  /ingest ──► pdf_parser ──► embeddings ──► vector_store (FAISS) │
│                                                   │             │
│  /ask ─────► embeddings ──► vector_store.search   │             │
│                  └──────► rag_pipeline ──► NVIDIA NIM (Llama)   │
│                                     └──► SQLite log             │
│  /analytics ──► SQLite queries                                  │
└─────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│         data/  (on disk)        │
│  faiss_index.bin  chunks.pkl    │
│  rag_logs.db                    │
└─────────────────────────────────┘
```

---

## Project Structure

```
rag_qa_system/
├── backend/
│   ├── main.py                  # FastAPI app — routers, CORS, startup
│   ├── api/
│   │   ├── ingest.py            # POST /ingest
│   │   ├── ask.py               # POST /ask
│   │   └── analytics.py         # GET  /analytics
│   ├── core/
│   │   ├── pdf_parser.py        # PyMuPDF extraction + chunking
│   │   ├── embeddings.py        # BAAI/bge-large-en-v1.5 singleton
│   │   ├── vector_store.py      # FAISS IndexFlatIP build/save/load/search
│   │   └── rag_pipeline.py      # End-to-end query pipeline
│   ├── db/
│   │   ├── database.py          # SQLite init, connection, log_query
│   │   └── queries.py           # Analytics SQL helpers
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   └── requirements.txt
├── frontend/
│   └── app.py                   # Streamlit UI
└── README.md
```

---

## Setup

### 1. Clone / enter the project directory

```bash
cd rag_qa_system
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

> **Note:** `sentence-transformers` will download the ~1.3 GB  
> `BAAI/bge-large-en-v1.5` model on first run.  
> `faiss-cpu` is the CPU-only FAISS build — no GPU required.

### 4. Configure your `.env` file

Copy the `.env.example` template to a new file named `.env`:

```bash
# In your terminal, from the rag_qa_system/ directory:
cp .env.example .env
```

Open your newly created `.env` file and fill in:
- `NVIDIA_API_KEY`: Get your free key at [build.nvidia.com](https://build.nvidia.com).
- `EMBEDDING_MODEL_PATH`: Full local directory path to your downloaded `BAAI/bge-large-en-v1.5` files.
- `EMBEDDING_DEVICE`: Set to `cpu` or `cuda`.
- `FASTAPI_URL`: Keep as `http://127.0.0.1:8000` for local use.

---

## Running the System

> **Important — working directory:** All commands below must be run from
> inside the `rag_qa_system/` folder (the one that contains `backend/`,
> `frontend/`, and `.env`).

```powershell
# Navigate to the correct directory first
cd "d:\Projects\RAG System\rag_qa_system"
```

### Start the FastAPI backend

**Option A — launcher script (recommended, avoids import errors):**
```bash
python run_backend.py
```

**Option B — direct uvicorn (must be run from `rag_qa_system/`):**
```bash
uvicorn backend.main:app --reload --port 8000
```

API docs available at <http://127.0.0.1:8000/docs>.

### Start the Streamlit frontend (new terminal)

```bash
# From rag_qa_system/
streamlit run frontend/app.py
```

Opens automatically at <http://localhost:8501>.

---

## Usage

1. **Upload a PDF** in the left sidebar → the document is chunked, embedded, and indexed.
2. **Ask a question** in the "Ask a Question" tab.
3. **View analytics** in the "Analytics" tab (most asked, unanswered, avg latency).

---

## Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | **150 words** | Fits comfortably within the embedding model's 512-token limit (~200 tokens). This prevents truncation at the end of chunks and ensures specific details (like rates, terms) aren't diluted in large text blocks. |
| Overlap | **50 words** (33%) | A sliding overlap ensures sentences or key facts that straddle chunk boundaries are preserved in full in at least one chunk. |

Word-level splitting was chosen over character-level because it produces human-readable chunks whose size is predictable and independent of Unicode composition, making debugging straightforward.

---

## Design Decisions

### Why FAISS over ChromaDB?

| Criterion | FAISS | ChromaDB |
|-----------|-------|----------|
| Dependency weight | Minimal (C++ library) | Larger (SQLite + embedding server) |
| Persistence | Binary file — single copy, easy to swap | Managed directory |
| Query speed | Nanosecond inner-product via SIMD | Adds Python wrapper overhead |
| Suitability | Single-document, single-process use case | Multi-collection, persistent production use |

For a single-user, single-document assignment, FAISS's `IndexFlatIP` with L2-normalised vectors (cosine similarity) is the simplest, fastest, and most transparent choice.

### Why BAAI/bge-large-en-v1.5?

- Ranks #1 on the MTEB (Massive Text Embedding Benchmark) English retrieval leaderboard among freely available models.
- Produces 1 024-dimensional vectors — compact enough for CPU inference but highly discriminative.
- The retrieval-specific query instruction (`"Represent this sentence for searching relevant passages: "`) measurably improves recall compared to symmetric models like MiniLM.

### Why Llama-3.1-8B-Instruct on NVIDIA NIM?

- **Free tier**: The NVIDIA API Catalog provides free credits to test and run model requests.
- **State-of-the-Art performance**: Llama-3.1-8B-Instruct is highly optimized for instruction-following and context adherence.
- **Context fidelity**: It remains highly faithful to custom system prompts and retrieved context, minimising hallucinations.

---

## Database Schema

Table: **`query_logs`** (SQLite, `data/rag_logs.db`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-incrementing row ID |
| `query` | TEXT | Raw question string from the user |
| `answer` | TEXT | LLM-generated answer |
| `sources` | TEXT | JSON array of retrieved chunk strings |
| `latency_ms` | REAL | End-to-end pipeline time in milliseconds |
| `answer_found` | INTEGER | `1` = grounded answer found; `0` = sentinel response |
| `timestamp` | TEXT | ISO-8601 UTC timestamp of the request |

Analytics queries:
- **Top questions**: `GROUP BY query ORDER BY COUNT(*) DESC LIMIT 10`
- **Unanswered**: `WHERE answer_found = 0`
- **Avg latency**: `SELECT AVG(latency_ms) FROM query_logs`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | HuggingFace API token |
| `FASTAPI_URL` | `http://127.0.0.1:8000` | Backend URL used by the Streamlit app |

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/ingest` | Upload and index a PDF (`multipart/form-data`) |
| `GET` | `/ingest/status` | Retrieve active document ingestion status and metadata |
| `POST` | `/ask` | Ask a question (`{"query": "..."}`) |
| `GET` | `/analytics` | Retrieve usage statistics |

Full interactive docs: <http://127.0.0.1:8000/docs>
