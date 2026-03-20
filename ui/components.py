"""
components.py — Advanced UI Components
========================================
Production-grade Streamlit components with dark intelligence theme.
Dark navy background, electric cyan accents, JetBrains Mono data font.
"""

import re
import streamlit as st
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

from src.citation_enforcer import CitationRef
from src.rag_pipeline import RAGResponse, ProcessingResult


# ══════════════════════════════════════════════════════════════════
# CSS LOADER
# ══════════════════════════════════════════════════════════════════

def load_css():
    """Injects the custom CSS stylesheet into the Streamlit app."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        # encoding="utf-8" — fixes Windows cp1252 default encoding error
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_sidebar_header():
    """Renders the branded app title in the sidebar."""
    st.sidebar.markdown(
        """
        <div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 1px solid #1e2d4a; margin-bottom: 1.5rem;">
            <div style="
                font-family: 'Instrument Serif', Georgia, serif;
                font-size: 1.5rem;
                font-style: italic;
                color: #e8edf5;
                letter-spacing: -0.03em;
                line-height: 1.2;
                margin-bottom: 0.4rem;
            ">RAG Document<br>Chatbot</div>
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.6rem;
                color: #00d4ff;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                opacity: 0.8;
            ">Hybrid · Reranked · Cited</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_file_uploader() -> Optional[Any]:
    """Renders styled file upload section."""
    st.sidebar.markdown(
        """
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: #4a5a7a;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.5rem;
        ">Upload Document</div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader(
        label       = "Choose a file",
        type        = ["pdf", "docx", "txt", "csv", "html"],
        help        = "PDF · Word · TXT · CSV · HTML  (max 50 MB)",
        key         = "file_uploader",
        label_visibility = "collapsed",
    )
    return uploaded_file


def render_document_info(result: ProcessingResult):
    """Renders document stats after successful processing."""
    st.sidebar.markdown(
        f"""
        <div style="
            background: rgba(0,230,118,0.06);
            border: 1px solid rgba(0,230,118,0.2);
            border-left: 3px solid #00e676;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.75rem 0;
        ">
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.65rem;
                color: #00e676;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin-bottom: 0.5rem;
            ">✓ Document Loaded</div>
            <div style="
                font-family: 'Instrument Serif', serif;
                font-size: 0.95rem;
                font-style: italic;
                color: #e8edf5;
                margin-bottom: 0.6rem;
                word-break: break-all;
            ">{result.source_name}</div>
            <div style="
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.4rem;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.68rem;
                color: #8896b3;
            ">
                <span>Chunks: <span style="color:#00d4ff">{result.num_chunks}</span></span>
                <span>Pages: <span style="color:#00d4ff">{result.num_pages}</span></span>
                <span>Avg: <span style="color:#00d4ff">{result.avg_chunk_size:.0f}c</span></span>
                <span>Time: <span style="color:#00d4ff">{result.processing_time:.1f}s</span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.already_existed:
        st.sidebar.caption("⚡ Loaded from cache")


def render_settings() -> Dict[str, Any]:
    """Renders retrieval settings with styled labels."""
    st.sidebar.markdown(
        """
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: #4a5a7a;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin: 1.25rem 0 0.5rem 0;
            padding-top: 1rem;
            border-top: 1px solid #1e2d4a;
        ">Retrieval Settings</div>
        """,
        unsafe_allow_html=True,
    )

    retrieval_top_k = st.sidebar.slider(
        "Retrieval Top-K",
        min_value = 5,
        max_value = 30,
        value     = 20,
        step      = 5,
        help      = "Chunks retrieved before reranking",
    )

    rerank_top_k = st.sidebar.slider(
        "Rerank Top-K",
        min_value = 2,
        max_value = 8,
        value     = 3,
        step      = 1,
        help      = "Chunks kept after reranking",
    )

    return {
        "retrieval_top_k": retrieval_top_k,
        "rerank_top_k"   : rerank_top_k,
    }


def render_clear_button() -> bool:
    """Renders the clear conversation button."""
    st.sidebar.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    return st.sidebar.button(
        "↺  Clear Conversation",
        use_container_width = True,
        help = "Clear chat history (document stays loaded)",
    )


def render_pipeline_status(status: Dict[str, Any]):
    """Renders pipeline component status badges."""
    st.sidebar.markdown(
        """
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: #4a5a7a;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin: 1.25rem 0 0.75rem 0;
            padding-top: 1rem;
            border-top: 1px solid #1e2d4a;
        ">System Status</div>
        """,
        unsafe_allow_html=True,
    )

    components = [
        ("Embeddings", status.get("embedding_provider", "—"), True),
        ("LLM",        status.get("llm_provider", "—"),        True),
        ("Reranker",   "ms-marco",                             status.get("reranker_loaded", False)),
        ("Document",   status.get("current_source", "none"),   status.get("document_loaded", False)),
    ]

    rows = ""
    for name, value, ok in components:
        dot_color = "#00e676" if ok else "#ff4757"
        rows += f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding: 0.3rem 0; border-bottom: 1px solid #1e2d4a;">
            <span style="color:#4a5a7a; font-size:0.65rem">{name}</span>
            <span style="display:flex; align-items:center; gap:0.35rem;">
                <span style="width:5px; height:5px; border-radius:50%;
                             background:{dot_color}; display:inline-block;"></span>
                <span style="color:#8896b3; font-size:0.65rem;
                             max-width:100px; overflow:hidden; text-overflow:ellipsis;
                             white-space:nowrap;">{str(value)[:18]}</span>
            </span>
        </div>
        """

    st.sidebar.markdown(
        f'<div style="font-family: JetBrains Mono, monospace;">{rows}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════

def render_welcome_message():
    """Renders the welcome screen using pure Streamlit components."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 📚")
        st.markdown("## *Ask anything about your documents*")
        st.markdown(
            "Every answer is grounded in your document with "
            "**verifiable citations**. Powered by hybrid BM25 + vector "
            "retrieval and cross-encoder reranking."
        )
        st.markdown("---")
        st.markdown("**Features:**")
        st.markdown("- 🔀 BM25 + Vector Hybrid Search")
        st.markdown("- 🏆 Cross-Encoder Reranking")
        st.markdown("- 📎 Citation Enforcement")
        st.markdown("- ✅ Hallucination Detection")
        st.markdown("---")
        st.info("👈 Upload a document in the sidebar to begin")

# ══════════════════════════════════════════════════════════════════
# CHAT HEADER
# ══════════════════════════════════════════════════════════════════

def render_chat_header(doc_name: str, settings: Dict):
    """Renders the chat area header with document name and settings."""
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #1e2d4a;
        ">
            <div>
                <div style="
                    font-family: 'Instrument Serif', Georgia, serif;
                    font-size: 1.6rem;
                    font-style: italic;
                    color: #e8edf5;
                    letter-spacing: -0.03em;
                    line-height: 1.2;
                ">{doc_name}</div>
                <div style="
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 0.62rem;
                    color: #4a5a7a;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    margin-top: 0.2rem;
                ">top-{settings['retrieval_top_k']} retrieve → top-{settings['rerank_top_k']} rerank</div>
            </div>
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.62rem;
                color: #00d4ff;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                opacity: 0.7;
            ">Hybrid RAG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# MESSAGE RENDERING
# ══════════════════════════════════════════════════════════════════

def render_user_message(content: str):
    """Renders a user chat message."""
    with st.chat_message("user"):
        st.markdown(
            f'<div style="color: #e8edf5; font-family: Inter, sans-serif; '
            f'font-size: 0.92rem; line-height: 1.6;">{content}</div>',
            unsafe_allow_html=True,
        )


def _highlight_citations(text: str) -> str:
    """Wraps [SOURCE: ...] tags in styled spans."""
    pattern = re.compile(r'(\[SOURCE:[^\]]+\])', re.IGNORECASE)
    return pattern.sub(
        r'<span style="font-family: JetBrains Mono, monospace; font-size: 0.68rem; '
        r'background: rgba(0,212,255,0.1); color: #00d4ff; padding: 1px 5px; '
        r'border-radius: 3px; border: 1px solid rgba(0,212,255,0.2); '
        r'white-space: nowrap;">\1</span>',
        text,
    )


def render_assistant_message(response: RAGResponse, message_id: int):
    """Renders a complete assistant response with stats and citation panel."""
    with st.chat_message("assistant"):

        # ── Answer text with highlighted citations ──────────────
        highlighted = _highlight_citations(response.answer)
        st.markdown(
            f'<div style="color: #e8edf5; font-family: Inter, sans-serif; '
            f'font-size: 0.92rem; line-height: 1.7; margin-bottom: 1rem;">'
            f'{highlighted}</div>',
            unsafe_allow_html=True,
        )

        # ── Citation validity badge ──────────────────────────────
        if response.citation_valid:
            badge_html = (
                f'<span style="font-family: JetBrains Mono, monospace; '
                f'font-size: 0.68rem; background: rgba(0,230,118,0.1); '
                f'color: #00e676; border: 1px solid rgba(0,230,118,0.25); '
                f'padding: 3px 10px; border-radius: 4px; letter-spacing: 0.04em;">'
                f'✓ Citations valid · {response.coverage_score:.0%} coverage · '
                f'{len(response.citations)} source{"s" if len(response.citations) != 1 else ""}</span>'
            )
        else:
            badge_html = (
                f'<span style="font-family: JetBrains Mono, monospace; '
                f'font-size: 0.68rem; background: rgba(245,166,35,0.1); '
                f'color: #f5a623; border: 1px solid rgba(245,166,35,0.25); '
                f'padding: 3px 10px; border-radius: 4px; letter-spacing: 0.04em;">'
                f'⚠ Citation score {response.coverage_score:.0%} — verify carefully</span>'
            )

        st.markdown(badge_html, unsafe_allow_html=True)

        # ── Stats row ────────────────────────────────────────────
        st.markdown("<div style='margin-top: 0.85rem;'></div>", unsafe_allow_html=True)
        stats = response.retrieval_stats

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Retrieved",
                response.num_chunks_retrieved,
                help="Hybrid BM25 + vector search"
            )
        with col2:
            st.metric(
                "Reranked to",
                response.num_chunks_reranked,
                help="After cross-encoder"
            )
        with col3:
            bm25_n = stats.get("bm25_count", 0)
            vec_n  = stats.get("vector_count", 0)
            st.metric(
                "BM25 / Vec",
                f"{bm25_n} / {vec_n}",
                help="Keyword vs semantic hits"
            )
        with col4:
            st.metric(
                "Time",
                f"{response.processing_time_ms:.0f}ms",
                help="Total query time"
            )

        # ── Sources expander ─────────────────────────────────────
        if response.source_chunks:
            with st.expander(
                f"◈  View Sources  ·  {len(response.source_chunks)} chunk"
                f"{'s' if len(response.source_chunks) != 1 else ''}",
                expanded=False,
            ):
                for i, chunk in enumerate(response.source_chunks, start=1):
                    score = (
                        response.rerank_scores[i - 1]
                        if i - 1 < len(response.rerank_scores)
                        else 0.0
                    )
                    render_source_card(chunk, score, i, message_id)


def render_source_card(
    chunk      : Document,
    score      : float,
    card_index : int,
    message_id : int,
):
    """Renders one source chunk as a styled card."""

    # Score styling
    if score >= 0.8:
        score_color = "#00e676"
        score_label = "HIGH"
    elif score >= 0.6:
        score_color = "#f5a623"
        score_label = "MED"
    else:
        score_color = "#ff4757"
        score_label = "LOW"

    source   = chunk.metadata.get("source",    "—")
    page     = chunk.metadata.get("page",      "—")
    chunk_id = chunk.metadata.get("chunk_id",  "—")
    ret_src  = chunk.metadata.get("retrieval_source", "—")

    # Card header
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.6rem 0;
            border-bottom: 1px solid #1e2d4a;
            margin-bottom: 0.6rem;
        ">
            <span style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.68rem;
                color: #8896b3;
                letter-spacing: 0.08em;
            ">SOURCE {card_index}
            &nbsp;·&nbsp; {source}
            &nbsp;·&nbsp; Page {page}
            &nbsp;·&nbsp; <span style="color: #4a5a7a">{ret_src}</span>
            </span>
            <span style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.65rem;
                font-weight: 600;
                color: {score_color};
                background: rgba(0,0,0,0.3);
                border: 1px solid {score_color}40;
                padding: 2px 8px;
                border-radius: 4px;
                letter-spacing: 0.06em;
            ">{score_label} {score:.3f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Chunk text
    st.text_area(
        label       = "",
        value       = chunk.page_content,
        height      = 100,
        disabled    = True,
        key         = f"src_{message_id}_{card_index}",
        label_visibility = "collapsed",
    )

    # Chunk ID footer
    st.markdown(
        f'<div style="font-family: JetBrains Mono, monospace; font-size: 0.62rem; '
        f'color: #2a3f6b; margin-top: 0.2rem; margin-bottom: 0.75rem; '
        f'word-break: break-all;">{chunk_id}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# PROCESSING PROGRESS
# ══════════════════════════════════════════════════════════════════

def render_processing_progress(stage: str, percent: int):
    """Renders a progress bar with mono-font stage label."""
    st.sidebar.markdown(
        f'<div style="font-family: JetBrains Mono, monospace; font-size: 0.68rem; '
        f'color: #00d4ff; letter-spacing: 0.05em; margin-bottom: 0.3rem;">'
        f'{stage}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.progress(percent / 100)


# ══════════════════════════════════════════════════════════════════
# ERROR MESSAGE
# ══════════════════════════════════════════════════════════════════

def render_error_message(error: str, suggestion: str = ""):
    """Renders a styled error box."""
    st.markdown(
        f"""
        <div style="
            background: rgba(255,71,87,0.07);
            border: 1px solid rgba(255,71,87,0.25);
            border-left: 3px solid #ff4757;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin: 0.5rem 0;
        ">
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                color: #ff4757;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 0.4rem;
            ">Error</div>
            <div style="
                font-family: 'Inter', sans-serif;
                font-size: 0.85rem;
                color: #e8edf5;
                line-height: 1.5;
            ">{error}</div>
            {'<div style="font-family: Inter, sans-serif; font-size: 0.78rem; color: #8896b3; margin-top: 0.5rem;">' + suggestion + '</div>' if suggestion else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# THINKING INDICATOR
# ══════════════════════════════════════════════════════════════════

def render_thinking_indicator():
    """Returns a spinner context manager."""
    return st.spinner("Retrieving · Reranking · Generating…")