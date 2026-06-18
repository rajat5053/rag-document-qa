"""
Streamlit frontend for the RAG Document Q&A system.

Provides a clean, interactive web UI with:
    - Sidebar: PDF upload → calls POST /ingest.
    - "Ask a Question" tab: text input → calls POST /ask → displays answer + sources.
    - "Analytics" tab: calls GET /analytics → displays usage statistics.

The backend URL defaults to http://localhost:8000 but can be overridden via
the FASTAPI_URL environment variable for deployment flexibility.

Usage:
    streamlit run frontend/app.py
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env file at startup
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark glassmorphism aesthetic
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Cards / containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, rgba(0,210,140,0.15), rgba(0,150,100,0.08));
        border: 1px solid rgba(0,210,140,0.4);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #e0fff6;
        font-size: 1rem;
        line-height: 1.7;
        margin-top: 1rem;
    }

    /* Error box */
    .error-box {
        background: rgba(255, 70, 70, 0.12);
        border: 1px solid rgba(255,70,70,0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #ffcccc;
        margin-top: 0.5rem;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.55);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: rgba(255,255,255,0.6);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Robust Text Input and Textarea wrapper overrides for Light/Dark mode consistency */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    .stTextInput input,
    .stTextArea textarea {
        background-color: rgba(15, 15, 35, 0.75) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }

    /* Change placeholder text color to be legible but muted */
    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }

    /* Ensure text inside input is always white, even on focus */
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        color: #ffffff !important;
        background-color: rgba(10, 10, 25, 0.9) !important;
    }

    h1, h2, h3 {
        color: #e2d9f3 !important;
    }
    p, label, .stMarkdown {
        color: rgba(255,255,255,0.8) !important;
    }

    /* Success banners */
    [data-testid="stSuccess"] {
        background: rgba(0,210,140,0.15);
        border: 1px solid rgba(0,210,140,0.4);
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — PDF Ingestion Status & Upload
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📄 Document Ingestion")
    
    # ── Check current active document status from backend ─────────────────
    active_doc = None
    try:
        status_resp = requests.get(f"{FASTAPI_URL}/ingest/status", timeout=5)
        if status_resp.status_code == 200:
            status_data = status_resp.json()
            if status_data.get("ingested"):
                active_doc = status_data
    except Exception:
        pass # fail silently, default to showing empty state

    if active_doc:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 1rem; margin-bottom: 1.5rem; border-color: rgba(0,210,140,0.3);">
                <p style="margin: 0; font-size: 0.75rem; color: #00d28c; font-weight: 600; text-transform: uppercase;">
                    🟢 Active Document
                </p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; font-weight: 500; color: #ffffff; word-break: break-all;">
                    {active_doc['filename']}
                </p>
                <p style="margin: 0; font-size: 0.78rem; color: rgba(255,255,255,0.5);">
                    Size: <b>{active_doc['chunks_count']}</b> chunks
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:0.8rem; color:rgba(255,255,255,0.55); margin-bottom: 0.2rem;'>"
            "Upload a different PDF to replace the active document:</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='font-size:0.85rem; color:rgba(255,255,255,0.55);'>"
            "Upload a PDF to index its contents for Q&A.</p>",
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="pdf_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Check session state to prevent infinite ingestion rerun loops
        state_key = f"ingested_{uploaded_file.name}_{uploaded_file.size}"
        if state_key not in st.session_state:
            with st.spinner("⚙️ Ingesting document…"):
                try:
                    response = requests.post(
                        f"{FASTAPI_URL}/ingest",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=300,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # Mark as ingested in session state BEFORE calling rerun
                        st.session_state[state_key] = True
                        st.success(
                            f"✅ Ingested **{data.get('chunks_ingested', '?')}** chunks "
                            f"from *{uploaded_file.name}*"
                        )
                        # Rerun so the 'Active Document' UI card updates immediately
                        st.rerun()
                    else:
                        detail = response.json().get("detail", response.text)
                        st.error(f"❌ Ingestion failed ({response.status_code}): {detail}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to the backend. "
                        "Is `uvicorn backend.main:app --port 8000` running?"
                    )
                except requests.exceptions.Timeout:
                    st.error("❌ The ingestion request timed out. Try a smaller PDF.")
                except requests.exceptions.RequestException as exc:
                    st.error(f"❌ Request error: {exc}")

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.78rem; color:rgba(255,255,255,0.35);'>"
        "Backend: <code style='color:#a78bfa'>"
        f"{FASTAPI_URL}</code></p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style='text-align:center; padding: 2rem 0 1rem 0;'>
      <h1 style='font-size:2.6rem; font-weight:700;
                 background: linear-gradient(135deg, #a78bfa, #60a5fa);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                 margin-bottom:0.3rem;'>
        📚 RAG Document Q&amp;A
      </h1>
      <p style='color:rgba(255,255,255,0.5); font-size:1rem;'>
        Ask questions about your documents — powered by Llama-3.1 &amp; FAISS
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ask, tab_analytics = st.tabs(["💬 Ask a Question", "📊 Analytics"])

# ============================================================
# TAB 1 — Ask a Question
# ============================================================

with tab_ask:
    st.markdown("### Ask anything about your document")

    query_input = st.text_area(
        "Your question",
        placeholder="e.g. What are the main conclusions of this document?",
        height=100,
        key="query_input",
        label_visibility="collapsed",
    )

    col_btn, col_spacer = st.columns([1, 5])
    with col_btn:
        submit = st.button("🔍 Submit", key="submit_btn", use_container_width=True)

    if submit:
        if not query_input or not query_input.strip():
            st.warning("⚠️ Please enter a question before submitting.")
        else:
            with st.spinner("🤔 Thinking…"):
                try:
                    resp = requests.post(
                        f"{FASTAPI_URL}/ask",
                        json={"query": query_input.strip()},
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])
                        latency = data.get("latency_ms", 0)

                        # Answer display
                        st.markdown(
                            f"<div class='answer-box'>{answer}</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"⏱ Response time: **{latency:.0f} ms**")

                        # Sources expander
                        if sources:
                            with st.expander(f"📎 View {len(sources)} source chunk(s)"):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(
                                        f"**Chunk {i}**\n\n"
                                        f"<div class='glass-card' style='font-size:0.85rem;"
                                        f"color:rgba(255,255,255,0.75);'>{src}</div>",
                                        unsafe_allow_html=True,
                                    )

                    elif resp.status_code == 503:
                        st.warning(
                            "⚠️ No document has been ingested yet. "
                            "Please upload a PDF in the sidebar first."
                        )
                    elif resp.status_code == 400:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"❌ Bad request: {detail}")
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"❌ Error ({resp.status_code}): {detail}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to the backend. "
                        "Is `uvicorn backend.main:app --port 8000` running?"
                    )
                except requests.exceptions.Timeout:
                    st.error("❌ The request timed out. The LLM may be loading — try again.")
                except requests.exceptions.RequestException as exc:
                    st.error(f"❌ Request error: {exc}")

# ============================================================
# TAB 2 — Analytics
# ============================================================

with tab_analytics:
    st.markdown("### Usage Analytics")

    refresh = st.button("🔄 Refresh", key="refresh_analytics")

    if refresh or True:  # auto-load on tab open
        try:
            resp = requests.get(f"{FASTAPI_URL}/analytics", timeout=30)
            if resp.status_code == 200:
                adata = resp.json()

                # ---- Metrics row ----
                total_q = adata.get("total_queries", 0)
                answered_q = adata.get("answered_queries", 0)
                avg_lat = adata.get("avg_latency_ms")
                
                # Calculate success/answer rate
                answer_rate = (answered_q / total_q * 100.0) if total_q > 0 else 100.0
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric(
                        label="📊 Total Queries Logged",
                        value=str(total_q),
                        help="Total number of queries run and saved in the SQLite query log."
                    )
                with col_m2:
                    st.metric(
                        label="✅ Answered Queries",
                        value=str(answered_q),
                        help="Number of queries successfully answered from the document context."
                    )
                with col_m3:
                    st.metric(
                        label="🎯 Answer Success Rate",
                        value=f"{answer_rate:.1f}%",
                        help="Percentage of user queries successfully answered from the context."
                    )
                with col_m4:
                    st.metric(
                        label="⚡ Avg Response Latency",
                        value=f"{avg_lat:.1f} ms" if avg_lat is not None else "N/A",
                        help="Average end-to-end pipeline latency across all logged queries."
                    )

                st.markdown("---")

                col_top, col_unans = st.columns(2)

                # ---- Top questions ----
                with col_top:
                    st.markdown("#### 🏆 Top 10 Most Asked Questions")
                    st.markdown(
                        "<p style='font-size:0.82rem; color:rgba(255,255,255,0.5); margin-top:-0.5rem;'>"
                        "Aggregated unique queries, sorted by frequency count.</p>",
                        unsafe_allow_html=True,
                    )
                    top_q = adata.get("top_questions", [])
                    if top_q:
                        import pandas as pd

                        df_top = pd.DataFrame(top_q)
                        df_top.columns = ["Question", "Ask Count"]
                        st.dataframe(
                            df_top,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No queries logged yet.")

                # ---- Unanswered queries ----
                with col_unans:
                    st.markdown(f"#### ❓ Unanswered Queries ({total_q - answered_q})")
                    st.markdown(
                        "<p style='font-size:0.82rem; color:rgba(255,255,255,0.5); margin-top:-0.5rem;'>"
                        "Queries where answer was not found in the context document.</p>",
                        unsafe_allow_html=True,
                    )
                    unans = adata.get("unanswered_queries", [])
                    if unans:
                        import pandas as pd

                        df_unans = pd.DataFrame(unans)
                        df_unans.columns = ["Question", "Timestamp"]
                        st.dataframe(
                            df_unans,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.success("🎉 All queries were answered from the document!")

            else:
                detail = resp.json().get("detail", resp.text)
                st.error(f"❌ Analytics error ({resp.status_code}): {detail}")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to the backend. "
                "Is `uvicorn backend.main:app --port 8000` running?"
            )
        except requests.exceptions.Timeout:
            st.error("❌ Analytics request timed out.")
        except requests.exceptions.RequestException as exc:
            st.error(f"❌ Request error: {exc}")
