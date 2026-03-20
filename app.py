"""
app.py — Main Streamlit Application
=====================================
Entry point for the RAG Document Chatbot web interface.

Run with: streamlit run app.py

Architecture:
  - Session state manages pipeline and chat history across reruns
  - Sidebar handles document upload and settings
  - Main area shows chat interface with citation panels
  - All rendering delegated to ui/components.py
"""

import os
import tempfile                    # Save uploaded files to disk temporarily
from pathlib import Path

import streamlit as st
from loguru import logger

# Import pipeline and UI components
from src.rag_pipeline import RAGPipeline
from ui.components import (
    render_sidebar_header,
    render_file_uploader,
    render_processing_progress,
    render_document_info,
    render_settings,
    render_clear_button,
    render_pipeline_status,
    render_user_message,
    render_assistant_message,
    render_error_message,
    render_welcome_message,
    render_thinking_indicator,
)


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# Must be the FIRST Streamlit call in the script
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "RAG Document Chatbot",
    page_icon  = "📚",
    layout     = "wide",           # use full browser width
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# These values persist across Streamlit reruns
# ══════════════════════════════════════════════════════════════════

def init_session_state():
    """
    Initializes all session state variables on first run.
    On subsequent reruns, existing values are preserved.
    """
    # The RAG pipeline — initialized once, reused for all queries
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    # Chat message history
    # Each message: {"role": "user"/"assistant", "content": str,
    #                "response": RAGResponse or None}
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Currently loaded document name
    if "current_doc" not in st.session_state:
        st.session_state.current_doc = None

    # Processing result from last document upload
    if "processing_result" not in st.session_state:
        st.session_state.processing_result = None

    # Track if pipeline is initialized
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False


# ══════════════════════════════════════════════════════════════════
# PIPELINE INITIALIZATION
# ══════════════════════════════════════════════════════════════════

def get_or_create_pipeline() -> RAGPipeline:
    """
    Returns existing pipeline or creates a new one.
    Pipeline is cached in session_state — only created once per session.

    Returns:
        Ready RAGPipeline instance
    """
    if st.session_state.pipeline is None:
        with st.spinner(
            "🚀 Initializing pipeline... "
            "Loading embedding model and reranker (~15s first time)..."
        ):
            try:
                st.session_state.pipeline       = RAGPipeline()
                st.session_state.pipeline_ready = True
                logger.info("Pipeline initialized successfully")
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {e}")
                logger.error(f"Pipeline init failed: {e}")
                st.stop()

    return st.session_state.pipeline


# ══════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSING
# ══════════════════════════════════════════════════════════════════

def handle_document_upload(uploaded_file, pipeline: RAGPipeline):
    """
    Handles a newly uploaded file:
      1. Saves uploaded bytes to a temp file on disk
      2. Calls pipeline.process_document()
      3. Updates session state with result
      4. Shows progress during processing

    Args:
        uploaded_file: Streamlit UploadedFile object
        pipeline     : initialized RAGPipeline
    """
    # Check if this is a NEW file (not the same one already loaded)
    if uploaded_file.name == st.session_state.current_doc:
        return   # same file already processed — do nothing

    st.sidebar.info(f"📥 Processing '{uploaded_file.name}'...")

    try:
        # Streamlit gives us file bytes — we need a real file path
        # Save to a temporary file that gets deleted after processing
        with tempfile.NamedTemporaryFile(
            delete = False,
            suffix = Path(uploaded_file.name).suffix,  # keep extension
        ) as tmp_file:
            # Write uploaded bytes to temp file
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Show progress stages
        render_processing_progress("Loading document...", 10)
        render_processing_progress("Chunking text...", 30)
        render_processing_progress("Building embeddings...", 60)
        render_processing_progress("Building BM25 index...", 80)

        # Process the document
        result = pipeline.process_document(tmp_path)

        # Rename the source to the original filename
        # (temp file has a random name like /tmp/tmp1234.pdf)
        result.source_name = uploaded_file.name

        render_processing_progress("Ready!", 100)

        # Update session state
        st.session_state.current_doc        = uploaded_file.name
        st.session_state.processing_result  = result

        # Clear old chat when new document is loaded
        st.session_state.messages = []

        logger.info(
            f"Document processed: '{uploaded_file.name}' | "
            f"Chunks: {result.num_chunks}"
        )

    except Exception as e:
        render_error_message(
            error      = str(e),
            suggestion = "Check that the file is not corrupted and try again.",
        )
        logger.error(f"Document processing failed: {e}")

    finally:
        # Always clean up temp file
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════

def main():
    """
    Main application function.
    Streamlit reruns this entire function on every user interaction.
    Session state preserves values between reruns.
    """
    # ── Initialize session state ─────────────────────────────────
    init_session_state()

    # ── Initialize pipeline (once per session) ───────────────────
    pipeline = get_or_create_pipeline()

    # ══════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════

    render_sidebar_header()

    # File uploader
    uploaded_file = render_file_uploader()

    # Handle new file upload
    if uploaded_file is not None:
        handle_document_upload(uploaded_file, pipeline)

    # Show document info if a document is loaded
    if st.session_state.processing_result is not None:
        render_document_info(st.session_state.processing_result)

    # Settings sliders
    settings = render_settings()

    # Clear conversation button
    if render_clear_button():
        st.session_state.messages = []
        st.rerun()

    # Pipeline status
    if st.session_state.pipeline_ready:
        render_pipeline_status(pipeline.get_pipeline_status())

    # ══════════════════════════════════════════════════════════════
    # MAIN CHAT AREA
    # ══════════════════════════════════════════════════════════════

    # Show welcome screen if no document loaded
    if st.session_state.current_doc is None:
        render_welcome_message()
        st.stop()   # don't render chat input until doc is loaded

    # ── Chat header ───────────────────────────────────────────────
    st.markdown(f"### 💬 Chat — *{st.session_state.current_doc}*")
    st.caption(
        f"Ask anything about your document. "
        f"Retrieval: top-{settings['retrieval_top_k']} → "
        f"reranked to {settings['rerank_top_k']}"
    )

    # ── Render existing chat history ─────────────────────────────
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            render_user_message(message["content"])
        else:
            # Render assistant message with full citation panel
            if message.get("response"):
                render_assistant_message(
                    response   = message["response"],
                    message_id = i,
                )
            else:
                # Fallback for messages without response object
                with st.chat_message("assistant"):
                    st.write(message["content"])

    # ── Chat input ────────────────────────────────────────────────
    question = st.chat_input(
        placeholder = "Ask a question about your document...",
        key         = "chat_input",
    )

    # ── Handle new question ───────────────────────────────────────
    if question:
        # Add user message to history
        st.session_state.messages.append({
            "role"    : "user",
            "content" : question,
            "response": None,
        })

        # Render user message immediately
        render_user_message(question)

        # Generate answer
        with render_thinking_indicator():
            try:
                response = pipeline.query(
                    question = question,
                    stream   = False,
                )

            except Exception as e:
                logger.error(f"Query failed: {e}")
                render_error_message(
                    error      = str(e),
                    suggestion = "Try rephrasing your question.",
                )
                st.stop()

        # Render assistant response
        render_assistant_message(
            response   = response,
            message_id = len(st.session_state.messages),
        )

        # Save to chat history
        st.session_state.messages.append({
            "role"    : "assistant",
            "content" : response.answer,
            "response": response,
        })

        # Rerun to update the UI cleanly
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()