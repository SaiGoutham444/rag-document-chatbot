"""
hybrid_retriever.py — Hybrid BM25 + Vector Retrieval
======================================================
Combines BM25 (keyword) and vector (semantic) search results
using Reciprocal Rank Fusion (RRF) into a single ranked list.

WHY hybrid?
  BM25 alone  : misses semantic paraphrases
  Vector alone: misses exact keyword/code/name matches
  Hybrid      : catches both — higher recall than either alone

WHY RRF?
  BM25 scores (0-10) and vector scores (0-1) are incompatible scales.
  RRF uses only RANK POSITIONS — completely scale-invariant.
  A chunk ranked high in BOTH retrievers scores highest overall.

Output feeds into the cross-encoder reranker (Phase 7)
which re-scores the top-20 hybrid results with higher precision.
"""

import time  # Timing operations
from typing import List, Tuple, Dict, Optional  # Type hints

from langchain_core.documents import Document
from loguru import logger

from src.config import (
    RETRIEVAL_TOP_K,  # How many results from each retriever (default: 20)
    RRF_K,  # RRF constant k=60
)
from src.bm25_retriever import BM25Retriever
from src.vector_store import query_vector_store


# ══════════════════════════════════════════════════════════════════
# RECIPROCAL RANK FUSION
# Core algorithm that merges two ranked lists
# ══════════════════════════════════════════════════════════════════


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Document, float]],
    vector_results: List[Tuple[Document, float]],
    k: int = RRF_K,
) -> List[Tuple[Document, float]]:
    """
    Merges BM25 and vector search results using Reciprocal Rank Fusion.

    Algorithm:
      1. For every chunk in BM25 results:
           rrf_score += 1 / (k + rank_in_bm25)
      2. For every chunk in vector results:
           rrf_score += 1 / (k + rank_in_vector)
      3. If a chunk appears in BOTH lists:
           it gets contributions from BOTH — naturally boosted
      4. Sort all chunks by final RRF score descending

    Deduplication is automatic:
      We use chunk_id as the key in a dictionary.
      If the same chunk appears in both BM25 and vector results,
      its RRF score accumulates both contributions instead of
      appearing twice in the output.

    Args:
        bm25_results  : list of (Document, score) from BM25, sorted by score
        vector_results: list of (Document, score) from vector store, sorted
        k             : RRF constant (default: 60 from config)

    Returns:
        Merged, deduplicated list of (Document, rrf_score) sorted descending

    Raises:
        ValueError: if both result lists are empty
    """
    if not bm25_results and not vector_results:
        raise ValueError(
            "Cannot fuse empty result lists.\n"
            "Both BM25 and vector search returned no results."
        )

    # Dictionary to accumulate RRF scores
    # Key  : chunk_id (unique identifier for each chunk)
    # Value: dict with "document" and "rrf_score"
    rrf_scores: Dict[str, Dict] = {}

    # ── Process BM25 results ────────────────────────────────────────
    # rank starts at 1 (1-indexed, as in the original RRF paper)
    for rank, (doc, bm25_score) in enumerate(bm25_results, start=1):
        # Get the unique chunk ID from metadata
        chunk_id = doc.metadata.get("chunk_id", f"bm25_chunk_{rank}")

        # RRF contribution from BM25 rank position
        rrf_contribution = 1.0 / (k + rank)

        if chunk_id not in rrf_scores:
            # First time we see this chunk — initialize its entry
            rrf_scores[chunk_id] = {
                "document": doc,
                "rrf_score": 0.0,
                "bm25_score": bm25_score,
                "bm25_rank": rank,
                "vector_score": 0.0,
                "vector_rank": None,  # None = not in vector results
                "in_bm25": True,
                "in_vector": False,
            }

        # Add BM25 contribution to running RRF score
        rrf_scores[chunk_id]["rrf_score"] += rrf_contribution
        rrf_scores[chunk_id]["bm25_score"] = bm25_score
        rrf_scores[chunk_id]["bm25_rank"] = rank
        rrf_scores[chunk_id]["in_bm25"] = True

    # ── Process vector results ──────────────────────────────────────
    for rank, (doc, vector_score) in enumerate(vector_results, start=1):
        chunk_id = doc.metadata.get("chunk_id", f"vector_chunk_{rank}")

        # RRF contribution from vector rank position
        rrf_contribution = 1.0 / (k + rank)

        if chunk_id not in rrf_scores:
            # New chunk — only in vector results, not in BM25
            rrf_scores[chunk_id] = {
                "document": doc,
                "rrf_score": 0.0,
                "bm25_score": 0.0,
                "bm25_rank": None,  # None = not in BM25 results
                "vector_score": vector_score,
                "vector_rank": rank,
                "in_bm25": False,
                "in_vector": True,
            }
        else:
            # Chunk already seen in BM25 results — update vector fields
            rrf_scores[chunk_id]["vector_score"] = vector_score
            rrf_scores[chunk_id]["vector_rank"] = rank
            rrf_scores[chunk_id]["in_vector"] = True

        # Add vector contribution to running RRF score
        rrf_scores[chunk_id]["rrf_score"] += rrf_contribution

    # ── Sort by RRF score descending ────────────────────────────────
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    # ── Build output as (Document, rrf_score) tuples ────────────────
    # Also attach retrieval source info to document metadata
    # This lets us show "found by BM25 + Vector" in the UI
    output = []
    for item in sorted_results:
        doc = item["document"]

        # Enrich metadata with retrieval provenance information
        doc.metadata["rrf_score"] = round(item["rrf_score"], 6)
        doc.metadata["bm25_score"] = round(item["bm25_score"], 4)
        doc.metadata["vector_score"] = round(item["vector_score"], 4)
        doc.metadata["bm25_rank"] = item["bm25_rank"]
        doc.metadata["vector_rank"] = item["vector_rank"]
        doc.metadata["in_bm25"] = item["in_bm25"]
        doc.metadata["in_vector"] = item["in_vector"]

        # Label the retrieval source for UI display
        if item["in_bm25"] and item["in_vector"]:
            doc.metadata["retrieval_source"] = "BM25 + Vector"
        elif item["in_bm25"]:
            doc.metadata["retrieval_source"] = "BM25 only"
        else:
            doc.metadata["retrieval_source"] = "Vector only"

        output.append((doc, item["rrf_score"]))

    return output


# ══════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER CLASS
# ══════════════════════════════════════════════════════════════════


class HybridRetriever:
    """
    Combines BM25 and vector search using Reciprocal Rank Fusion.

    This is the main retrieval component used by the RAG pipeline.
    It orchestrates both retrievers and fuses their results.

    Usage:
        retriever = HybridRetriever(
            bm25_retriever  = bm25_retriever,
            embedding_model = embedding_model,
            source_name     = "report.pdf",
            chroma_client   = client,
            top_k           = 20,
        )
        results = retriever.retrieve("What was Q3 revenue?")
        # Returns: [(Document, rrf_score), ...] top-20 chunks
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        embedding_model,
        source_name: str,
        chroma_client,
        top_k: int = RETRIEVAL_TOP_K,
    ):
        """
        Initializes the hybrid retriever with both retrieval backends.

        Args:
            bm25_retriever : ready BM25Retriever (index built or loaded)
            embedding_model: the embedding model (same used for indexing)
            source_name    : document filename for vector store lookup
            chroma_client  : ChromaDB persistent client
            top_k          : results to fetch from each retriever
        """
        # Store both retrieval backends
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model
        self.source_name = source_name
        self.chroma_client = chroma_client
        self.top_k = top_k

        # Statistics tracking — filled during retrieve()
        self._last_stats: Dict = {}

        logger.info(
            f"HybridRetriever initialized | "
            f"Document: '{source_name}' | "
            f"Top-K per retriever: {top_k}"
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Runs hybrid retrieval: BM25 + vector → RRF fusion.

        Full process:
          1. Run BM25 retrieval → top_k results with BM25 scores
          2. Run vector retrieval → top_k results with cosine scores
          3. Apply RRF fusion → merged, deduplicated, re-ranked list
          4. Return top_k results from fused list

        Args:
            query : user's question as a string
            top_k : override default top_k for this query

        Returns:
            List of (Document, rrf_score) tuples, sorted by RRF score
            Each Document's metadata includes retrieval_source,
            bm25_score, vector_score, bm25_rank, vector_rank

        Raises:
            ValueError  : empty query
            RuntimeError: both retrievers fail
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty.")

            # Use instance default if not overridden
            k = top_k or self.top_k

            logger.info(
                f"Hybrid retrieval started | " f"Query: '{query[:60]}' | " f"Top-K: {k}"
            )
            start_time = time.time()

            # ── Step 1: BM25 retrieval ──────────────────────────────
            bm25_results = []
            try:
                bm25_results = self.bm25_retriever.retrieve(
                    query=query,
                    top_k=k,
                )
                logger.info(f"  BM25    → {len(bm25_results)} results")
            except Exception as bm25_error:
                # BM25 failure is non-fatal — vector search can still run
                logger.warning(
                    f"  BM25 retrieval failed (non-fatal): {bm25_error}\n"
                    f"  Continuing with vector search only."
                )

            # ── Step 2: Vector retrieval ────────────────────────────
            vector_results = []
            try:
                vector_results = query_vector_store(
                    query=query,
                    source_name=self.source_name,
                    embedding_model=self.embedding_model,
                    top_k=k,
                    client=self.chroma_client,
                )
                logger.info(f"  Vector  → {len(vector_results)} results")
            except Exception as vector_error:
                # Vector failure is non-fatal — BM25 can still work
                logger.warning(
                    f"  Vector retrieval failed (non-fatal): {vector_error}\n"
                    f"  Continuing with BM25 only."
                )

            # ── Guard: at least one retriever must have results ─────
            if not bm25_results and not vector_results:
                logger.warning(
                    f"Both retrievers returned no results for: '{query}'\n"
                    f"The document may not contain relevant information."
                )
                return []

            # ── Step 3: RRF Fusion ──────────────────────────────────
            fused_results = reciprocal_rank_fusion(
                bm25_results=bm25_results,
                vector_results=vector_results,
                k=RRF_K,
            )

            # ── Step 4: Take top_k from fused list ──────────────────
            final_results = fused_results[:k]

            # ── Compute and store retrieval statistics ───────────────
            elapsed = time.time() - start_time

            # Count how many chunks came from each source
            both_count = sum(
                1
                for doc, _ in final_results
                if doc.metadata.get("in_bm25") and doc.metadata.get("in_vector")
            )
            bm25_only = sum(
                1
                for doc, _ in final_results
                if doc.metadata.get("in_bm25") and not doc.metadata.get("in_vector")
            )
            vector_only = sum(
                1
                for doc, _ in final_results
                if doc.metadata.get("in_vector") and not doc.metadata.get("in_bm25")
            )

            self._last_stats = {
                "query": query,
                "bm25_count": len(bm25_results),
                "vector_count": len(vector_results),
                "fused_total": len(fused_results),
                "returned": len(final_results),
                "in_both": both_count,
                "bm25_only": bm25_only,
                "vector_only": vector_only,
                "retrieval_time_ms": round(elapsed * 1000, 1),
                "top_rrf_score": final_results[0][1] if final_results else 0,
            }

            logger.info(
                f"Hybrid retrieval complete in {elapsed*1000:.0f}ms\n"
                f"  BM25 results    : {len(bm25_results)}\n"
                f"  Vector results  : {len(vector_results)}\n"
                f"  After fusion    : {len(fused_results)} unique chunks\n"
                f"  Returned top-{k}: {len(final_results)}\n"
                f"  In both         : {both_count}\n"
                f"  BM25 only       : {bm25_only}\n"
                f"  Vector only     : {vector_only}"
            )

            return final_results

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Hybrid retrieval failed: {e}") from e

    def get_retrieval_stats(self) -> Dict:
        """
        Returns statistics from the most recent retrieve() call.
        Used by the Streamlit UI to show retrieval breakdown.

        Returns:
            Dict with bm25_count, vector_count, in_both, bm25_only,
            vector_only, retrieval_time_ms, top_rrf_score
        """
        return self._last_stats.copy()


# ══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# Used by rag_pipeline.py to set up hybrid retriever in one call
# ══════════════════════════════════════════════════════════════════


def build_hybrid_retriever(
    bm25_retriever: BM25Retriever,
    embedding_model,
    source_name: str,
    chroma_client,
    top_k: int = RETRIEVAL_TOP_K,
) -> HybridRetriever:
    """
    Builds a HybridRetriever from its components.
    Simple factory function used by rag_pipeline.py.

    Args:
        bm25_retriever : ready BM25Retriever instance
        embedding_model: loaded embedding model
        source_name    : document filename
        chroma_client  : ChromaDB client
        top_k          : results per retriever

    Returns:
        Ready-to-use HybridRetriever instance
    """
    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        embedding_model=embedding_model,
        source_name=source_name,
        chroma_client=chroma_client,
        top_k=top_k,
    )
