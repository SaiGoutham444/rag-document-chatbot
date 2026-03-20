"""
rag_pipeline.py — RAG Pipeline Orchestration
=============================================
Connects all 9 components into a single RAGPipeline class.

This is the ONLY class the Streamlit UI interacts with.
It handles two operations:
  1. process_document(file_path) — load, chunk, embed, index
  2. query(question)             — retrieve, rerank, cite, generate

All components are initialized once in __init__() and reused
across multiple queries — no reloading between questions.

Data flow:
  Document → Loader → Chunker → VectorStore + BM25Index
  Question → HybridRetriever → Reranker → CitationEnforcer → LLM → Answer
"""

import time                                        # Timing operations
from dataclasses import dataclass, field           # Clean response containers
from typing import List, Optional, Dict, Any       # Type hints
from pathlib import Path                           # File path handling

from langchain_core.documents import Document
from loguru import logger

# ── Import all our components ──────────────────────────────────────
from src.config import (
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    GROQ_LLM_MODEL,
    EMBEDDING_PROVIDER,
    LLM_PROVIDER,
)
from src.document_loader import load_document, get_document_info
from src.chunker import split_documents, get_chunk_statistics
from src.embeddings import get_embedding_model
from src.vector_store import (
    get_chroma_client,
    add_documents_to_store,
    collection_exists,
    delete_collection,
    get_collection_info,
)
from src.bm25_retriever import build_or_load_bm25, BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.citation_enforcer import CitationEnforcer, CitationRef, EnforcedAnswer
from src.llm import get_llm, generate_answer, generate_answer_stream, check_context_fits


# ══════════════════════════════════════════════════════════════════
# RESPONSE DATA CLASSES
# Clean containers returned to the UI
# ══════════════════════════════════════════════════════════════════

@dataclass
class ProcessingResult:
    """
    Result returned after processing a document.
    Contains stats about what was loaded and indexed.
    """
    source_name     : str               # filename
    num_pages       : int               # pages/sections loaded
    num_chunks      : int               # chunks after splitting
    avg_chunk_size  : float             # average chars per chunk
    processing_time : float             # total seconds
    already_existed : bool              # True if was already indexed
    document_info   : Dict = field(default_factory=dict)  # full doc stats


@dataclass
class RAGResponse:
    """
    Complete response from a pipeline query.
    Everything the UI needs to display a full cited answer.
    """
    answer            : str                        # LLM answer with citation tags
    citations         : List[CitationRef]          # parsed citation objects
    source_chunks     : List[Document]             # actual chunks that were cited
    retrieval_stats   : Dict[str, Any]             # BM25/vector breakdown
    rerank_scores     : List[float]                # cross-encoder scores
    citation_valid    : bool                       # meets MIN_CITATION_SCORE?
    processing_time_ms: float                      # total query time in ms
    num_chunks_retrieved: int = 0                  # after hybrid retrieval
    num_chunks_reranked : int = 0                  # after reranking
    error             : Optional[str] = None       # error message if failed
    coverage_score    : float = 0.0                # citation coverage 0-1


# ══════════════════════════════════════════════════════════════════
# RAG PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Orchestrates the complete RAG pipeline end-to-end.

    Lifecycle:
      1. __init__()           : loads all ML models (embedding, reranker, LLM)
      2. process_document()   : indexes a document (load→chunk→embed→BM25)
      3. query()              : answers a question with citations
      4. delete_document()    : removes a document from all indexes

    The pipeline is STATEFUL — it remembers which document is loaded
    and reuses all initialized models across multiple queries.
    This avoids reloading the embedding model (12 seconds) every query.

    Usage:
        pipeline = RAGPipeline()
        result   = pipeline.process_document("reports/Q3.pdf")
        response = pipeline.query("What was Q3 revenue?")
        print(response.answer)
    """

    def __init__(self):
        """
        Initializes all pipeline components.

        Loads in order:
          1. Embedding model (~12s first run, instant after)
          2. ChromaDB client (instant)
          3. Cross-encoder reranker (~3s first run)
          4. LLM connection (instant, API-based)
          5. Citation enforcer (instant)

        State variables set to None until process_document() is called:
          self.current_source  : filename of loaded document
          self.chunks          : document chunks in memory
          self.bm25_retriever  : BM25 index for current document
          self.hybrid_retriever: combined retriever for current document

        Raises:
            RuntimeError: if any critical component fails to initialize
        """
        logger.info("=" * 60)
        logger.info("Initializing RAG Pipeline...")
        logger.info("=" * 60)

        pipeline_start = time.time()

        # ── Step 1: Embedding model ─────────────────────────────────
        logger.info("Loading embedding model...")
        t = time.time()
        self.embedding_model = get_embedding_model()
        logger.info(f"  Embedding model ready in {time.time()-t:.1f}s")

        # ── Step 2: ChromaDB client ─────────────────────────────────
        logger.info("Connecting to ChromaDB...")
        t = time.time()
        self.chroma_client = get_chroma_client()
        logger.info(f"  ChromaDB ready in {time.time()-t:.1f}s")

        # ── Step 3: Cross-encoder reranker ──────────────────────────
        logger.info("Loading cross-encoder reranker...")
        t = time.time()
        self.reranker = CrossEncoderReranker(top_k=RERANK_TOP_K)
        self.reranker.load_model()
        logger.info(f"  Reranker ready in {time.time()-t:.1f}s")

        # ── Step 4: LLM ─────────────────────────────────────────────
        logger.info("Connecting to LLM...")
        t = time.time()
        self.llm = get_llm()
        logger.info(f"  LLM ready in {time.time()-t:.1f}s")

        # ── Step 5: Citation enforcer ────────────────────────────────
        self.enforcer = CitationEnforcer()

        # ── State: set when process_document() is called ────────────
        self.current_source    : Optional[str]            = None
        self.chunks            : Optional[List[Document]] = None
        self.bm25_retriever    : Optional[BM25Retriever]  = None
        self.hybrid_retriever  : Optional[HybridRetriever]= None

        total_time = time.time() - pipeline_start
        logger.info("=" * 60)
        logger.info(f"RAG Pipeline ready in {total_time:.1f}s")
        logger.info("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # DOCUMENT PROCESSING
    # ──────────────────────────────────────────────────────────────

    def process_document(
        self,
        file_path       : str,
        force_reprocess : bool = False,
    ) -> ProcessingResult:
        """
        Loads, chunks, embeds, and indexes a document.

        If the document was already processed (collection exists in
        ChromaDB and BM25 index exists on disk), skips re-embedding
        and just loads the existing indexes. This makes re-uploads
        of the same document nearly instant.

        Full process:
          1. Validate file (exists, supported type, not too large)
          2. Check if already indexed (skip if yes, unless force=True)
          3. Load document → List[Document]
          4. Split into chunks → List[Document] with chunk_ids
          5. Embed chunks → store in ChromaDB
          6. Build BM25 index → save to disk
          7. Initialize HybridRetriever with both indexes
          8. Update pipeline state (self.current_source, etc.)

        Args:
            file_path      : path to document (PDF, DOCX, TXT, CSV, HTML)
            force_reprocess: if True, re-embed even if already indexed

        Returns:
            ProcessingResult with stats about what was processed

        Raises:
            FileNotFoundError: file doesn't exist
            ValueError       : unsupported file type or empty file
            RuntimeError     : embedding or indexing fails
        """
        logger.info(f"Processing document: '{file_path}'")
        start_time = time.time()

        # ── Step 1: Load the document ───────────────────────────────
        raw_docs = load_document(file_path)
        source_name = Path(file_path).name
        doc_info = get_document_info(raw_docs)

        logger.info(
            f"Document loaded: {len(raw_docs)} sections | "
            f"{doc_info.get('total_words', 0):,} words"
        )

        # ── Step 2: Check if already indexed ────────────────────────
        already_existed = False
        if not force_reprocess:
            chroma_ready = collection_exists(self.chroma_client, source_name)
            bm25_ready   = BM25Retriever(raw_docs[:1]).index_exists_on_disk(source_name) \
                           if raw_docs else False

            if chroma_ready and bm25_ready:
                logger.info(
                    f"Document '{source_name}' already indexed. "
                    f"Loading existing indexes..."
                )
                already_existed = True

        # ── Step 3: Split into chunks ────────────────────────────────
        # Always split — we need chunks in memory for BM25
        self.chunks = split_documents(raw_docs)
        stats = get_chunk_statistics(self.chunks)

        logger.info(
            f"Chunking complete: {stats['total_chunks']} chunks | "
            f"Avg: {stats['avg_chars']:.0f} chars"
        )

        # ── Step 4: Embed and store in ChromaDB ─────────────────────
        if not already_existed or force_reprocess:
            logger.info("Embedding chunks and storing in ChromaDB...")
            add_documents_to_store(
                chunks          = self.chunks,
                source_name     = source_name,
                embedding_model = self.embedding_model,
                client          = self.chroma_client,
            )

        # ── Step 5: Build or load BM25 index ────────────────────────
        logger.info("Building/loading BM25 index...")
        self.bm25_retriever = build_or_load_bm25(
            chunks      = self.chunks,
            source_name = source_name,
        )

        # ── Step 6: Initialize HybridRetriever ──────────────────────
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever  = self.bm25_retriever,
            embedding_model = self.embedding_model,
            source_name     = source_name,
            chroma_client   = self.chroma_client,
            top_k           = RETRIEVAL_TOP_K,
        )

        # ── Step 7: Update pipeline state ───────────────────────────
        self.current_source = source_name

        elapsed = time.time() - start_time

        result = ProcessingResult(
            source_name     = source_name,
            num_pages       = len(raw_docs),
            num_chunks      = stats["total_chunks"],
            avg_chunk_size  = stats["avg_chars"],
            processing_time = elapsed,
            already_existed = already_existed,
            document_info   = doc_info,
        )

        logger.info(
            f"Document processing complete in {elapsed:.2f}s | "
            f"Chunks: {stats['total_chunks']} | "
            f"Already existed: {already_existed}"
        )
        return result

    # ──────────────────────────────────────────────────────────────
    # QUERY — the main function
    # ──────────────────────────────────────────────────────────────

    def query(
        self,
        question : str,
        stream   : bool = False,
    ) -> RAGResponse:
        """
        Answers a question using the full RAG pipeline.

        Full process:
          1. Validate: document must be processed first
          2. Hybrid retrieval (BM25 + vector → RRF) → top-20
          3. Cross-encoder reranking → top-5
          4. Build citation prompt with labeled context
          5. Check prompt fits model context window
          6. Generate answer with LLM
          7. Enforce citations (parse + validate)
          8. Return RAGResponse with everything

        Args:
            question: user's question as a string
            stream  : if True, streams tokens (for Streamlit UI)

        Returns:
            RAGResponse with answer, citations, stats, timing

        Raises:
            RuntimeError: no document loaded, or pipeline component fails
            ValueError  : empty question
        """
        try:
            # ── Guard: question must not be empty ───────────────────
            if not question or not question.strip():
                raise ValueError("Question cannot be empty.")

            # ── Guard: document must be processed first ──────────────
            if not self.hybrid_retriever or not self.current_source:
                raise RuntimeError(
                    "No document loaded.\n"
                    "Call process_document() before query()."
                )

            logger.info(f"Query: '{question[:80]}'")
            query_start = time.time()

            # ── Step 1: Hybrid Retrieval ─────────────────────────────
            logger.info("Step 1/5: Hybrid retrieval...")
            hybrid_results = self.hybrid_retriever.retrieve(
                query = question,
                top_k = RETRIEVAL_TOP_K,
            )
            retrieval_stats = self.hybrid_retriever.get_retrieval_stats()

            # Handle case: no results found
            if not hybrid_results:
                logger.warning(
                    f"No chunks retrieved for: '{question}'\n"
                    f"Document may not contain relevant information."
                )
                return RAGResponse(
                    answer             = "I cannot find this information in the provided document.",
                    citations          = [],
                    source_chunks      = [],
                    retrieval_stats    = retrieval_stats,
                    rerank_scores      = [],
                    citation_valid     = True,
                    processing_time_ms = (time.time() - query_start) * 1000,
                    coverage_score     = 1.0,
                )

            logger.info(f"  Retrieved {len(hybrid_results)} chunks")

            # ── Step 2: Cross-Encoder Reranking ─────────────────────
            logger.info("Step 2/5: Reranking...")
            reranked = self.reranker.rerank(
                query     = question,
                documents = hybrid_results,
                top_k     = RERANK_TOP_K,
            )

            rerank_scores   = [float(score) for _, score in reranked]
            reranked_docs   = [doc for doc, _ in reranked]

            logger.info(
                f"  Reranked to {len(reranked)} chunks | "
                f"Top score: {rerank_scores[0]:.4f}"
            )

            # ── Step 3: Build Citation Prompt ────────────────────────
            logger.info("Step 3/5: Building citation prompt...")
            context = self.enforcer.build_context_with_ids(reranked_docs)
            prompt  = self.enforcer.build_citation_prompt(question, context)

            # Check if prompt fits model context window
            if not check_context_fits(prompt, GROQ_LLM_MODEL):
                # Trim to fewer chunks and rebuild
                logger.warning(
                    "Prompt too long — trimming to 3 chunks"
                )
                reranked_docs = reranked_docs[:3]
                rerank_scores = rerank_scores[:3]
                context       = self.enforcer.build_context_with_ids(reranked_docs)
                prompt        = self.enforcer.build_citation_prompt(question, context)

            logger.info(f"  Prompt: {len(prompt)} chars")

            # ── Step 4: Generate Answer ──────────────────────────────
            logger.info("Step 4/5: Generating answer...")
            raw_answer = generate_answer(
                llm    = self.llm,
                prompt = prompt,
                stream = stream,
            )

            logger.info(f"  Answer: {len(raw_answer)} chars")

            # ── Step 5: Enforce Citations ────────────────────────────
            logger.info("Step 5/5: Enforcing citations...")
            enforced = self.enforcer.enforce_citations(
                answer           = raw_answer,
                retrieved_chunks = reranked_docs,
            )

            # ── Build final response ─────────────────────────────────
            elapsed_ms = (time.time() - query_start) * 1000

            response = RAGResponse(
                answer              = enforced.answer,
                citations           = enforced.citations,
                source_chunks       = enforced.source_chunks,
                retrieval_stats     = retrieval_stats,
                rerank_scores       = rerank_scores,
                citation_valid      = enforced.is_valid,
                processing_time_ms  = elapsed_ms,
                num_chunks_retrieved= len(hybrid_results),
                num_chunks_reranked = len(reranked),
                coverage_score      = enforced.report.coverage_score,
            )

            logger.info(
                f"Query complete in {elapsed_ms:.0f}ms | "
                f"Citations: {len(enforced.citations)} | "
                f"Valid: {enforced.is_valid} | "
                f"Score: {enforced.report.coverage_score:.2f}"
            )
            return response

        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            elapsed_ms = (time.time() - query_start) * 1000
            logger.error(f"Query failed: {e}")
            return RAGResponse(
                answer              = f"An error occurred: {str(e)}",
                citations           = [],
                source_chunks       = [],
                retrieval_stats     = {},
                rerank_scores       = [],
                citation_valid      = False,
                processing_time_ms  = elapsed_ms,
                error               = str(e),
                coverage_score      = 0.0,
            )

    # ──────────────────────────────────────────────────────────────
    # STREAMING QUERY — for Streamlit UI
    # ──────────────────────────────────────────────────────────────

    def query_stream(self, question: str):
        """
        Streaming version of query() — yields tokens for Streamlit.

        Runs retrieval and reranking first (non-streaming),
        then streams the LLM generation token by token.

        After streaming completes, citation enforcement runs
        on the complete accumulated answer.

        Args:
            question: user's question

        Yields:
            Individual token strings from LLM generation

        Usage in Streamlit:
            response_text = st.write_stream(pipeline.query_stream(q))
        """
        try:
            if not question or not question.strip():
                yield "Error: Question cannot be empty."
                return

            if not self.hybrid_retriever:
                yield "Error: No document loaded. Please upload a document first."
                return

            # Run retrieval + reranking (not streamed)
            hybrid_results = self.hybrid_retriever.retrieve(question)

            if not hybrid_results:
                yield "I cannot find this information in the provided document."
                return

            reranked      = self.reranker.rerank(question, hybrid_results, RERANK_TOP_K)
            reranked_docs = [doc for doc, _ in reranked]

            context = self.enforcer.build_context_with_ids(reranked_docs)
            prompt  = self.enforcer.build_citation_prompt(question, context)

            # Stream LLM tokens
            yield from generate_answer_stream(self.llm, prompt)

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"\n\n[Error: {str(e)}]"

    # ──────────────────────────────────────────────────────────────
    # DOCUMENT MANAGEMENT
    # ──────────────────────────────────────────────────────────────

    def delete_document(self, source_name: str) -> bool:
        """
        Removes a document from ChromaDB and clears pipeline state.

        Args:
            source_name: filename of document to delete

        Returns:
            True if deleted successfully, False if not found
        """
        try:
            deleted = delete_collection(self.chroma_client, source_name)

            # Clear pipeline state if deleted document was active
            if deleted and self.current_source == source_name:
                self.current_source   = None
                self.chunks           = None
                self.bm25_retriever   = None
                self.hybrid_retriever = None
                logger.info(f"Pipeline state cleared for '{source_name}'")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete document '{source_name}': {e}")
            return False

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Returns current pipeline status.
        Used by Streamlit sidebar to show what's loaded.

        Returns:
            Dict with loaded document info, model names, component status
        """
        return {
            "document_loaded"    : self.current_source is not None,
            "current_source"     : self.current_source,
            "num_chunks"         : len(self.chunks) if self.chunks else 0,
            "embedding_provider" : EMBEDDING_PROVIDER,
            "llm_provider"       : LLM_PROVIDER,
            "reranker_loaded"    : self.reranker._model is not None,
            "retriever_ready"    : self.hybrid_retriever is not None,
        }