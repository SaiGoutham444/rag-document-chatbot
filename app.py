"""
app.py — RAG Document Chatbot
==============================
Advanced dark UI. Pure Streamlit + minimal inline CSS.
Run: streamlit run app.py
"""

import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from loguru import logger

from src.rag_pipeline import RAGPipeline


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title            = "RAG Document Chatbot",
    page_icon             = "📚",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

.stApp { background-color: #0d1117 !important; }
section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #21262d !important; }
.main .block-container { padding-top: 1.5rem !important; max-width: 860px !important; }
#MainMenu, footer, header { visibility: hidden; }

p, div { color: #8b949e; }
.stMarkdown p { color: #c9d1d9 !important; line-height: 1.7; }
h1,h2,h3 { color: #f0f6fc !important; font-weight: 400 !important; }

.stButton>button { background:#21262d !important; border:1px solid #30363d !important; color:#8b949e !important; border-radius:6px !important; font-size:0.8rem !important; }
.stButton>button:hover { border-color:#58a6ff !important; color:#58a6ff !important; }

[data-testid="stMetric"] { background:#161b22 !important; border:1px solid #21262d !important; border-radius:8px !important; padding:0.75rem !important; }
[data-testid="stMetricLabel"] p { color:#8b949e !important; font-size:0.68rem !important; text-transform:uppercase !important; letter-spacing:0.08em !important; }
[data-testid="stMetricValue"] { color:#58a6ff !important; font-family:'JetBrains Mono',monospace !important; }

[data-testid="stExpander"] { background:#161b22 !important; border:1px solid #21262d !important; border-radius:8px !important; }
[data-testid="stExpander"] summary p { color:#8b949e !important; font-size:0.75rem !important; text-transform:uppercase !important; letter-spacing:0.06em !important; }

[data-testid="stChatInput"] textarea { background:#161b22 !important; color:#f0f6fc !important; }
[data-testid="stProgressBar"]>div { background:#21262d !important; height:3px !important; border-radius:2px !important; }
[data-testid="stProgressBar"]>div>div { background: linear-gradient(90deg,#58a6ff,#79c0ff) !important; }

textarea[disabled] { background:#0d1117 !important; color:#8b949e !important; border:1px solid #21262d !important; font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important; line-height:1.6 !important; }
hr { border-color:#21262d !important; }
::-webkit-scrollbar { width:4px; } ::-webkit-scrollbar-track { background:#0d1117; } ::-webkit-scrollbar-thumb { background:#30363d; border-radius:2px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def hl_citations(text: str) -> str:
    """Wrap [SOURCE:...] tags in cyan highlight spans."""
    return re.sub(
        r'(\[SOURCE:[^\]]+\])',
        r'<span style="font-family:JetBrains Mono,monospace;font-size:0.68rem;'
        r'background:rgba(88,166,255,0.12);color:#58a6ff;'
        r'border:1px solid rgba(88,166,255,0.3);padding:1px 5px;'
        r'border-radius:3px;">\1</span>',
        text, flags=re.IGNORECASE,
    )


def score_html(score: float) -> str:
    """Return colored score badge HTML."""
    if score >= 0.8:   c, lbl = "#2ea043", "HIGH"
    elif score >= 0.6: c, lbl = "#d29922", "MED"
    else:              c, lbl = "#da3633", "LOW"
    return (
        f'<span style="background:{c}22;color:{c};border:1px solid {c}44;'
        f'padding:2px 8px;border-radius:4px;font-family:JetBrains Mono,monospace;'
        f'font-size:0.67rem;font-weight:600;">{lbl} {score:.3f}</span>'
    )


def mono(text: str, color: str = "#8b949e", size: str = "0.68rem") -> str:
    """Return monospace span HTML."""
    return (
        f'<span style="font-family:JetBrains Mono,monospace;'
        f'font-size:{size};color:{color};'
        f'text-transform:uppercase;letter-spacing:0.08em;">{text}</span>'
    )


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

def init_session():
    for k, v in {
        "pipeline": None, "pipeline_ready": False,
        "messages": [], "current_doc": None,
        "processing_result": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════

def get_pipeline() -> RAGPipeline:
    if st.session_state.pipeline is None:
        with st.spinner("Loading models — embedding + reranker + LLM…"):
            try:
                st.session_state.pipeline       = RAGPipeline()
                st.session_state.pipeline_ready = True
            except Exception as e:
                st.error(f"Pipeline init failed: {e}")
                st.stop()
    return st.session_state.pipeline


# ══════════════════════════════════════════════════════════════════
# UPLOAD HANDLER
# ══════════════════════════════════════════════════════════════════

def handle_upload(uploaded_file, pipeline: RAGPipeline):
    if uploaded_file.name == st.session_state.current_doc:
        return
    tmp_path = None
    try:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        prog    = st.sidebar.progress(0)
        status  = st.sidebar.empty()

        for label, pct in [
            ("Loading…", 20), ("Chunking…", 40),
            ("Embedding…", 70), ("Indexing…", 90), ("Ready!", 100),
        ]:
            status.markdown(mono(label, "#58a6ff", "0.72rem"), unsafe_allow_html=True)
            prog.progress(pct / 100)

        result = pipeline.process_document(tmp_path)
        result.source_name = uploaded_file.name

        st.session_state.current_doc        = uploaded_file.name
        st.session_state.processing_result  = result
        st.session_state.messages           = []
        status.empty()
        prog.empty()

    except Exception as e:
        st.sidebar.error(f"❌ {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

def render_sidebar(pipeline: RAGPipeline) -> dict:

    # Brand
    st.sidebar.markdown(
        '<h2 style="color:#f0f6fc;font-size:1.25rem;margin-bottom:0.1rem;font-weight:500;">'
        '📚 RAG Document Chatbot</h2>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        mono("Hybrid · Reranked · Cited", "#58a6ff"),
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    # Upload
    st.sidebar.markdown(mono("Upload Document"), unsafe_allow_html=True)
    uploaded = st.sidebar.file_uploader(
        "file", type=["pdf","docx","txt","csv","html"],
        label_visibility="collapsed", key="uploader",
    )
    if uploaded:
        handle_upload(uploaded, pipeline)

    # Doc info
    r = st.session_state.processing_result
    if r:
        st.sidebar.success(f"✅ **{r.source_name}**")
        a, b = st.sidebar.columns(2)
        a.metric("Chunks", r.num_chunks)
        b.metric("Pages",  r.num_pages)
        a.metric("Avg",    f"{r.avg_chunk_size:.0f}c")
        b.metric("Time",   f"{r.processing_time:.1f}s")
        if r.already_existed:
            st.sidebar.caption("⚡ Loaded from cache")

    st.sidebar.divider()

    # Settings
    st.sidebar.markdown(mono("Retrieval Settings"), unsafe_allow_html=True)
    top_k     = st.sidebar.slider("Retrieval Top-K", 5,  30, 20, 5)
    rerank_k  = st.sidebar.slider("Rerank Top-K",    2,   8,  3, 1)

    st.sidebar.divider()
    if st.sidebar.button("↺  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Status
    st.sidebar.divider()
    st.sidebar.markdown(mono("System Status"), unsafe_allow_html=True)
    s = pipeline.get_pipeline_status()
    for name, val, ok in [
        ("Embeddings", s.get("embedding_provider","—"), True),
        ("LLM",        s.get("llm_provider","—"),        True),
        ("Reranker",   "ms-marco",                       s.get("reranker_loaded", False)),
        ("Document",   s.get("current_source","none"),   s.get("document_loaded", False)),
    ]:
        dot = "🟢" if ok else "🔴"
        st.sidebar.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:0.25rem 0;border-bottom:1px solid #21262d;">'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#8b949e;">'
            f'{dot} {name}</span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#58a6ff;">'
            f'{str(val)[:20]}</span></div>',
            unsafe_allow_html=True,
        )

    return {"top_k": top_k, "rerank_k": rerank_k}


# ══════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════

def render_welcome():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2.5, 1])
    with col:
        st.markdown(
            '<div style="text-align:center;font-size:3rem;margin-bottom:1rem;">📚</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h1 style="text-align:center;color:#f0f6fc;font-size:2rem;'
            'font-weight:300;letter-spacing:-0.02em;line-height:1.3;'
            'margin-bottom:0.75rem;">'
            'Ask anything about<br>your documents'
            '</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="text-align:center;color:#8b949e;font-size:0.9rem;'
            'line-height:1.7;margin-bottom:2rem;">'
            'Every answer is grounded in your document with verifiable citations.<br>'
            'Powered by hybrid BM25 + vector search and cross-encoder reranking.'
            '</p>',
            unsafe_allow_html=True,
        )

        # Feature pills
        pills = ["🔀 Hybrid BM25+Vector", "🏆 Cross-Encoder Reranking",
                 "📎 Citations Enforced", "✅ Hallucination Detection"]
        pills_html = (
            '<div style="display:flex;flex-wrap:wrap;gap:0.5rem;'
            'justify-content:center;margin-bottom:2rem;">'
        )
        for p in pills:
            pills_html += (
                f'<span style="font-family:JetBrains Mono,monospace;'
                f'font-size:0.67rem;background:rgba(88,166,255,0.08);'
                f'color:#58a6ff;border:1px solid rgba(88,166,255,0.2);'
                f'padding:0.3rem 0.75rem;border-radius:20px;">{p}</span>'
            )
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)
        st.info("👈  Upload a document in the sidebar to get started")


# ══════════════════════════════════════════════════════════════════
# ASSISTANT RESPONSE
# ══════════════════════════════════════════════════════════════════

def render_response(response, msg_id: int):
    with st.chat_message("assistant"):

        # Answer with citation highlights
        st.markdown(
            f'<div style="color:#c9d1d9;font-size:0.92rem;line-height:1.75;'
            f'margin-bottom:0.85rem;">{hl_citations(response.answer)}</div>',
            unsafe_allow_html=True,
        )

        # Validity badge
        if response.citation_valid:
            st.success(
                f"✅  Citations valid · **{response.coverage_score:.0%}** coverage "
                f"· {len(response.citations)} source(s)"
            )
        else:
            st.warning(
                f"⚠️  Citation score **{response.coverage_score:.0%}** "
                f"— below threshold, verify carefully"
            )

        # Stats
        st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
        stats = response.retrieval_stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieved",   response.num_chunks_retrieved)
        c2.metric("Reranked to", response.num_chunks_reranked)
        c3.metric("BM25/Vec",    f"{stats.get('bm25_count',0)}/{stats.get('vector_count',0)}")
        c4.metric("Time",        f"{response.processing_time_ms:.0f}ms")

        # Sources
        if response.source_chunks:
            n = len(response.source_chunks)
            with st.expander(f"◈  Source Chunks  ({n})", expanded=False):
                for i, chunk in enumerate(response.source_chunks, 1):
                    score   = response.rerank_scores[i-1] if i-1 < len(response.rerank_scores) else 0.0
                    source  = chunk.metadata.get("source", "—")
                    page    = chunk.metadata.get("page",   "—")
                    ret_src = chunk.metadata.get("retrieval_source", "—")
                    cid     = chunk.metadata.get("chunk_id", "—")

                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'align-items:center;padding:0.5rem 0;'
                        f'border-bottom:1px solid #21262d;margin-bottom:0.5rem;">'
                        f'<span style="font-family:JetBrains Mono,monospace;'
                        f'font-size:0.67rem;color:#8b949e;">'
                        f'#{i} &nbsp;·&nbsp; {source} &nbsp;·&nbsp; '
                        f'pg {page} &nbsp;·&nbsp; {ret_src}</span>'
                        f'{score_html(score)}</div>',
                        unsafe_allow_html=True,
                    )
                    # NEW — proper label, hidden visually
                    st.text_area(
                        label            = f"Chunk {i} content",
                        value            = chunk.page_content,
                        height           = 95,
                        disabled         = True,
                        key              = f"c_{msg_id}_{i}",
                        label_visibility = "collapsed",
                    )
                    st.markdown(
                        f'<div style="font-family:JetBrains Mono,monospace;'
                        f'font-size:0.6rem;color:#30363d;margin-bottom:0.75rem;'
                        f'word-break:break-all;">{cid}</div>',
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    init_session()
    pipeline = get_pipeline()
    settings = render_sidebar(pipeline)

    # Welcome
    if st.session_state.current_doc is None:
        render_welcome()
        st.stop()

    # Chat header
    st.markdown(
        f'<div style="padding-bottom:1rem;margin-bottom:1.5rem;'
        f'border-bottom:1px solid #21262d;">'
        f'<h2 style="color:#f0f6fc;font-size:1.35rem;font-weight:400;'
        f'margin-bottom:0.2rem;">💬 {st.session_state.current_doc}</h2>'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:0.62rem;'
        f'color:#8b949e;text-transform:uppercase;letter-spacing:0.1em;">'
        f'retrieve top-{settings["top_k"]}  →  rerank top-{settings["rerank_k"]}  →  cited answer'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # Chat history
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(
                    f'<div style="color:#f0f6fc;font-size:0.92rem;line-height:1.65;">'
                    f'{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
        elif msg.get("response"):
            render_response(msg["response"], i)
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # Input
    question = st.chat_input("Ask anything about your document…")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question, "response": None}
        )
        with st.chat_message("user"):
            st.markdown(
                f'<div style="color:#f0f6fc;font-size:0.92rem;line-height:1.65;">'
                f'{question}</div>',
                unsafe_allow_html=True,
            )

        with st.spinner("Retrieving · Reranking · Generating…"):
            try:
                response = pipeline.query(question=question, stream=False)
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()

        render_response(response, len(st.session_state.messages))
        st.session_state.messages.append(
            {"role": "assistant", "content": response.answer, "response": response}
        )
        st.rerun()


if __name__ == "__main__":
    main()