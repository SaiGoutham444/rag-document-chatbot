"""
components.py — Reusable Streamlit UI Components
=================================================
All UI building blocks used by app.py.
Each function renders one piece of the interface.

Keeping components separate from app.py means:
  - app.py stays clean and readable
  - Components can be tested and modified independently
  - Easy to add new UI elements without touching main logic
"""

import streamlit as st
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

from src.citation_enforcer import CitationRef
from src.rag_pipeline import RAGResponse, ProcessingResult


def render_sidebar_header():
    """Renders the app title and description in the sidebar."""
    st.sidebar.title("📚 RAG Document Chatbot")
    st.sidebar.markdown(
        "Upload a document and ask questions. "
        "Every answer is grounded in your document with citations."
    )
    st.sidebar.divider()


def render_file_uploader() -> Optional[Any]:
    """
    Renders the file upload widget in the sidebar.

    Returns:
        Uploaded file object if a file was uploaded, None otherwise
    """
    st.sidebar.subheader("📄 Upload Document")

    uploaded_file = st.sidebar.file_uploader(
        label       = "Choose a file",
        # Accept all our supported types
        type        = ["pdf", "docx", "txt", "csv", "html"],
        help        = "Supported: PDF, Word, TXT, CSV, HTML (max 50MB)",
        # key ensures widget state is stable across reruns
        key         = "file_uploader",
    )
    return uploaded_file


def render_processing_progress(stage: str, percent: int):
    """
    Renders a progress bar with stage description.
    Called during document processing to show live progress.

    Args:
        stage  : human-readable stage name
        percent: 0-100 completion percentage
    """
    st.sidebar.info(f"⏳ {stage}")
    st.sidebar.progress(percent / 100)


def render_document_info(result: ProcessingResult):
    """
    Renders document statistics after successful processing.
    Shows in the sidebar as a green success box.

    Args:
        result: ProcessingResult from pipeline.process_document()
    """
    st.sidebar.success(f"✅ **{result.source_name}** loaded!")

    # Display key stats in a clean column layout
    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.metric("Chunks",  result.num_chunks)
        st.metric("Pages",   result.num_pages)

    with col2:
        st.metric("Avg Size", f"{result.avg_chunk_size:.0f}c")
        st.metric("Time",     f"{result.processing_time:.1f}s")

    # Show cache status
    if result.already_existed:
        st.sidebar.caption("⚡ Loaded from cache (instant)")
    else:
        st.sidebar.caption("🔄 Freshly processed and indexed")


def render_settings() -> Dict[str, Any]:
    """
    Renders retrieval settings sliders in the sidebar.
    Returns the current values as a dict.

    Returns:
        Dict with retrieval_top_k and rerank_top_k values
    """
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Settings")

    retrieval_top_k = st.sidebar.slider(
        label   = "Retrieval Top-K",
        min_value = 5,
        max_value = 30,
        value   = 20,
        step    = 5,
        help    = "How many chunks to retrieve before reranking. "
                  "Higher = better recall, slower speed.",
    )

    rerank_top_k = st.sidebar.slider(
        label   = "Rerank Top-K",
        min_value = 2,
        max_value = 8,
        value   = 3,
        step    = 1,
        help    = "How many chunks to keep after reranking. "
                  "Higher = more context for LLM, higher cost.",
    )

    return {
        "retrieval_top_k": retrieval_top_k,
        "rerank_top_k"   : rerank_top_k,
    }


def render_clear_button() -> bool:
    """
    Renders the Clear Conversation button.

    Returns:
        True if button was clicked, False otherwise
    """
    st.sidebar.divider()
    return st.sidebar.button(
        "🗑️ Clear Conversation",
        use_container_width = True,
        help = "Clears chat history but keeps document loaded",
    )


def render_pipeline_status(status: Dict[str, Any]):
    """
    Renders pipeline component status in the sidebar.
    Shows which models are loaded and ready.

    Args:
        status: dict from pipeline.get_pipeline_status()
    """
    st.sidebar.divider()
    st.sidebar.subheader("🔧 Pipeline Status")

    # Embedding model
    st.sidebar.caption(
        f"🔢 Embeddings: **{status.get('embedding_provider', 'unknown')}**"
    )

    # LLM
    st.sidebar.caption(
        f"🤖 LLM: **{status.get('llm_provider', 'unknown')}**"
    )

    # Reranker
    reranker_status = "✅ loaded" if status.get("reranker_loaded") else "⏳ not loaded"
    st.sidebar.caption(f"🏆 Reranker: {reranker_status}")

    # Document
    if status.get("document_loaded"):
        st.sidebar.caption(
            f"📄 Document: **{status.get('current_source')}** "
            f"({status.get('num_chunks')} chunks)"
        )
    else:
        st.sidebar.caption("📄 Document: **none loaded**")


def render_user_message(content: str):
    """
    Renders a user chat message (right side, blue).

    Args:
        content: the user's question text
    """
    with st.chat_message("user"):
        st.write(content)


def render_assistant_message(
    response  : RAGResponse,
    message_id: int,
):
    """
    Renders a complete assistant response with citations panel.

    Shows:
      1. The answer text with citation tags highlighted
      2. Retrieval statistics (chunks retrieved → reranked)
      3. Expandable "View Sources" section with chunk details

    Args:
        response  : RAGResponse from pipeline.query()
        message_id: unique ID for this message (for widget keys)
    """
    with st.chat_message("assistant"):

        # ── Answer text ─────────────────────────────────────────────
        st.write(response.answer)

        # ── Citation validity badge ──────────────────────────────────
        if response.citation_valid:
            st.success(
                f"✅ Citation score: {response.coverage_score:.0%} "
                f"| {len(response.citations)} sources cited",
                icon="✅"
            )
        else:
            st.warning(
                f"⚠️ Citation score: {response.coverage_score:.0%} "
                f"(below 70% threshold — verify answer carefully)",
                icon="⚠️"
            )

        # ── Retrieval stats bar ──────────────────────────────────────
        stats = response.retrieval_stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Retrieved",
                response.num_chunks_retrieved,
                help="Chunks from hybrid BM25 + vector search"
            )
        with col2:
            st.metric(
                "Reranked to",
                response.num_chunks_reranked,
                help="Chunks kept after cross-encoder reranking"
            )
        with col3:
            bm25_count = stats.get("bm25_count", 0)
            st.metric(
                "BM25 hits",
                bm25_count,
                help="Chunks found by keyword search"
            )
        with col4:
            st.metric(
                "Time",
                f"{response.processing_time_ms:.0f}ms",
                help="Total query processing time"
            )

        # ── Sources panel ────────────────────────────────────────────
        if response.source_chunks:
            with st.expander(
                f"📎 View Sources ({len(response.source_chunks)} chunks)",
                expanded=False,
            ):
                for i, (chunk, score) in enumerate(
                    zip(response.source_chunks, response.rerank_scores),
                    start=1
                ):
                    render_citation_card(
                        chunk      = chunk,
                        score      = score,
                        card_index = i,
                        message_id = message_id,
                    )


def render_citation_card(
    chunk      : Document,
    score      : float,
    card_index : int,
    message_id : int,
):
    """
    Renders one source chunk as a citation card inside the sources panel.

    Shows:
      - Source filename, page number, chunk ID
      - Rerank score with color-coded badge
      - Retrieval source (BM25 / Vector / Both)
      - Full chunk text in a code block

    Args:
        chunk     : the source Document
        score     : rerank score (0.0 to 1.0)
        card_index: position number (1, 2, 3...)
        message_id: parent message ID for unique widget keys
    """
    # Determine score color
    if score >= 0.8:
        score_color = "🟢"
        label       = "High relevance"
    elif score >= 0.6:
        score_color = "🟡"
        label       = "Medium relevance"
    else:
        score_color = "🔴"
        label       = "Low relevance"

    # Get metadata
    source   = chunk.metadata.get("source",    "unknown")
    page     = chunk.metadata.get("page",      "?")
    chunk_id = chunk.metadata.get("chunk_id",  "unknown")
    ret_src  = chunk.metadata.get("retrieval_source", "unknown")

    # Render the card
    st.markdown(f"**Source {card_index}** {score_color} `{score:.4f}` — {label}")

    # Metadata row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"📄 **File:** {source}")
    with col2:
        st.caption(f"📖 **Page:** {page}")
    with col3:
        st.caption(f"🔍 **Found by:** {ret_src}")

    # Chunk text
    st.text_area(
        label       = f"Chunk text",
        value       = chunk.page_content,
        height      = 120,
        disabled    = True,    # read-only
        key         = f"chunk_{message_id}_{card_index}",
    )

    # Chunk ID (small, for debugging)
    st.caption(f"🔑 Chunk ID: `{chunk_id}`")
    st.divider()


def render_retrieval_breakdown(stats: Dict[str, Any]):
    """
    Renders a detailed retrieval breakdown table.
    Shows how many chunks came from BM25, vector, or both.

    Args:
        stats: dict from hybrid_retriever.get_retrieval_stats()
    """
    if not stats:
        return

    in_both     = stats.get("in_both",      0)
    bm25_only   = stats.get("bm25_only",    0)
    vector_only = stats.get("vector_only",  0)

    st.markdown("**Retrieval Breakdown:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🔀 In Both",     in_both,
                  help="Found by BOTH BM25 and vector — highest confidence")
    with col2:
        st.metric("🔤 BM25 Only",   bm25_only,
                  help="Found only by keyword search")
    with col3:
        st.metric("🧠 Vector Only", vector_only,
                  help="Found only by semantic search")


def render_error_message(error: str, suggestion: str = ""):
    """
    Renders a red error message with optional suggestion.

    Args:
        error     : error message to display
        suggestion: optional fix suggestion for the user
    """
    st.error(f"❌ **Error:** {error}")
    if suggestion:
        st.info(f"💡 **Suggestion:** {suggestion}")


def render_welcome_message():
    """
    Renders the welcome screen shown before any document is uploaded.
    """
    st.markdown(
        """
        ## 👋 Welcome to RAG Document Chatbot

        This chatbot lets you upload any document and ask questions about it.
        Every answer is grounded in your document with **verifiable citations**.

        ### How it works:
        1. **Upload** a PDF, Word, TXT, CSV, or HTML file in the sidebar
        2. **Wait** for the document to be processed (chunked + indexed)
        3. **Ask** any question about your document
        4. **Get** a cited answer with source references

        ### What makes this advanced:
        - 🔀 **Hybrid search** — BM25 keywords + vector semantics
        - 🏆 **Cross-encoder reranking** — precision relevance scoring
        - 📎 **Citation enforcement** — every claim traced to source
        - ✅ **Quality validation** — hallucinations detected and flagged

        **👈 Start by uploading a document in the sidebar!**
        """
    )


def render_thinking_indicator():
    """
    Renders a spinner while the pipeline is processing.
    Used as a context manager.

    Usage:
        with render_thinking_indicator():
            response = pipeline.query(question)
    """
    return st.spinner("🤔 Thinking... retrieving, reranking, generating...")