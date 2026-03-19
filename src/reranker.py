"""
reranker.py — Cross-Encoder Reranking Module
=============================================
Re-scores retrieved chunks by reading query AND chunk together.

WHY reranking?
  Hybrid retrieval (Phase 6) finds top-20 candidates using
  fast but approximate methods (BM25 + cosine similarity).
  The cross-encoder reads each (query, chunk) pair together —
  much more accurate but too slow to run on all chunks.

  Two-stage pipeline:
    Stage 1: Hybrid retrieval  → top-20 candidates (fast)
    Stage 2: Cross-encoder     → top-5 finalists   (accurate)

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on 530K real search queries (MS MARCO dataset)
  - 22M parameters, runs on CPU in ~10ms per pair
  - Outputs relevance score 0.0 (irrelevant) to 1.0 (highly relevant)
"""

import time                                    # Timing reranking
from typing import List, Tuple, Optional       # Type hints

from langchain_core.documents import Document
from loguru import logger

from src.config import RERANKER_MODEL, RERANK_TOP_K   # Model name + top-K


# ══════════════════════════════════════════════════════════════════
# CROSS-ENCODER RERANKER CLASS
# ══════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    Reranks document chunks using a cross-encoder model.

    The cross-encoder reads (query, chunk) pairs together and
    assigns a precise relevance score to each pair.
    Much more accurate than cosine similarity but slower —
    therefore only used on the top-20 pre-retrieved chunks.

    Usage:
        reranker = CrossEncoderReranker()
        reranker.load_model()

        results = reranker.rerank(
            query     = "What was Q3 revenue?",
            documents = hybrid_results,   # top-20 from hybrid retriever
            top_k     = 5,
        )
        # Returns top-5 most relevant chunks with rerank scores
    """

    def __init__(
        self,
        model_name : str = RERANKER_MODEL,
        top_k      : int = RERANK_TOP_K,
    ):
        """
        Initializes the reranker with model name and settings.
        Model is NOT loaded here — we use lazy loading in load_model().

        WHY lazy loading?
        The model is ~85MB and takes ~3 seconds to load.
        We only load it when first needed, not at import time.
        This keeps app startup fast.

        Args:
            model_name: HuggingFace model identifier
            top_k     : how many chunks to return after reranking
        """
        self.model_name : str                    = model_name
        self.top_k      : int                    = top_k
        # Model starts as None — loaded on first use
        self._model                              = None

        logger.info(
            f"CrossEncoderReranker created | "
            f"Model: '{model_name}' | "
            f"Top-K: {top_k} | "
            f"Status: not loaded yet (lazy)"
        )

    def load_model(self) -> None:
        """
        Loads the cross-encoder model from HuggingFace.

        First run: downloads ~85MB model to ~/.cache/huggingface/
        Subsequent runs: loads from local cache in ~2-3 seconds

        The model is a fine-tuned MiniLM on MS MARCO — it learned
        from 530,000 real search queries what "relevant" means.

        Raises:
            RuntimeError: model download or load fails
        """
        try:
            # Guard: don't reload if already loaded
            if self._model is not None:
                logger.debug("Cross-encoder model already loaded — skipping")
                return

            logger.info(
                f"Loading cross-encoder model: '{self.model_name}'\n"
                f"  First run: downloads ~85MB to ~/.cache/huggingface/\n"
                f"  Subsequent runs: loads from cache in ~2-3 seconds"
            )

            start_time = time.time()

            # Import here (lazy import) — only load sentence-transformers
            # when the reranker is actually needed
            from sentence_transformers import CrossEncoder

            # CrossEncoder loads the model and tokenizer
            # max_length=512: maximum tokens per (query+chunk) pair
            # If combined length > 512, it truncates from the end
            self._model = CrossEncoder(
                model_name  = self.model_name,
                max_length  = 512,
            )

            elapsed = time.time() - start_time
            logger.info(
                f"Cross-encoder model loaded in {elapsed:.1f}s | "
                f"Ready for inference"
            )

        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed.\n"
                "Fix: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{self.model_name}': {e}\n"
                f"Check your internet connection for first-time download."
            ) from e

    def _ensure_model_loaded(self) -> None:
        """
        Ensures the model is loaded before inference.
        Called at the start of rerank() for safety.
        """
        if self._model is None:
            self.load_model()

    def rerank(
        self,
        query     : str,
        documents : List[Tuple[Document, float]],
        top_k     : Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Reranks documents by reading query + each chunk together.

        Process:
          1. Build (query, chunk_text) pairs for every document
          2. Run cross-encoder on all pairs simultaneously (batched)
          3. Apply sigmoid to convert raw logits → probabilities
          4. Sort by rerank score descending
          5. Return top_k with scores

        The cross-encoder REPLACES the previous scores entirely.
        BM25 scores and vector scores are discarded.
        Only the cross-encoder's judgment matters from here on.

        Args:
            query    : user's question as a string
            documents: list of (Document, score) from hybrid retriever
            top_k    : how many to return (default: self.top_k)

        Returns:
            List of (Document, rerank_score) tuples sorted descending
            rerank_score is 0.0-1.0 (sigmoid of raw logit)
            Each Document has "rerank_score" added to its metadata

        Raises:
            ValueError  : empty query or documents list
            RuntimeError: model inference fails
        """
        try:
            # ── Input validation ─────────────────────────────────────
            if not query or not query.strip():
                raise ValueError("Query cannot be empty.")

            if not documents:
                raise ValueError(
                    "Cannot rerank empty document list.\n"
                    "Ensure hybrid retriever returned results."
                )

            # Use instance default if not overridden
            k = top_k or self.top_k

            # ── Ensure model is loaded ───────────────────────────────
            self._ensure_model_loaded()

            logger.info(
                f"Reranking {len(documents)} documents | "
                f"Query: '{query[:60]}' | "
                f"Return top-{k}"
            )
            start_time = time.time()

            # ── Extract Document objects from (Document, score) pairs ─
            # We only need the documents for reranking
            # The previous scores (BM25/vector/RRF) are discarded
            docs_only = [doc for doc, _ in documents]

            # ── Build (query, chunk_text) pairs ──────────────────────
            # The cross-encoder expects a list of [text_a, text_b] pairs
            # text_a = the query (same for all pairs)
            # text_b = the chunk content (different for each)
            sentence_pairs = [
                [query, doc.page_content]
                for doc in docs_only
            ]

            # ── Run cross-encoder inference ──────────────────────────
            # predict() processes all pairs in one batched forward pass
            # Returns: numpy array of raw logit scores
            # Shape: (num_documents,)
            raw_scores = self._model.predict(
                sentence_pairs,
                # show_progress_bar: show tqdm bar during inference
                # False for cleaner logs (loguru handles our logging)
                show_progress_bar=False,
            )

            # ── Convert raw logits → probabilities via sigmoid ───────
            # Raw logits can be any real number (-inf to +inf)
            # Sigmoid maps them to (0.0, 1.0) for interpretability
            # sigmoid(x) = 1 / (1 + e^(-x))
            import numpy as np
            sigmoid_scores = 1.0 / (1.0 + np.exp(-raw_scores))

            # ── Pair documents with their rerank scores ───────────────
            doc_score_pairs = list(zip(docs_only, sigmoid_scores))

            # ── Sort by rerank score descending ─────────────────────
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # ── Take top_k results ───────────────────────────────────
            top_results = doc_score_pairs[:k]

            # ── Enrich metadata with rerank information ───────────────
            # Add score and rank to each document's metadata
            # This information is displayed in the Streamlit UI
            final_results = []
            for rank, (doc, score) in enumerate(top_results, start=1):
                score_float = float(score)

                # Add reranking info to metadata
                doc.metadata["rerank_score"] = round(score_float, 4)
                doc.metadata["rerank_rank"]  = rank

                # Add score label for UI color coding
                if score_float >= 0.8:
                    doc.metadata["relevance_label"] = "high"
                elif score_float >= 0.6:
                    doc.metadata["relevance_label"] = "medium"
                else:
                    doc.metadata["relevance_label"] = "low"

                final_results.append((doc, score_float))

            # ── Log results ──────────────────────────────────────────
            elapsed = time.time() - start_time
            scores_list = [round(float(s), 4) for _, s in final_results]

            logger.info(
                f"Reranking complete in {elapsed*1000:.0f}ms\n"
                f"  Input documents : {len(documents)}\n"
                f"  Returned top-{k} : {len(final_results)}\n"
                f"  Score range     : "
                f"{min(scores_list):.4f} – {max(scores_list):.4f}\n"
                f"  Top scores      : {scores_list}"
            )

            return final_results

        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(
                f"Reranking failed: {e}"
            ) from e

    def rerank_with_threshold(
        self,
        query     : str,
        documents : List[Tuple[Document, float]],
        threshold : float = 0.3,
        top_k     : Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Reranks and filters out chunks below a relevance threshold.

        Used when we want to avoid passing irrelevant chunks to the LLM.
        If the top-5 chunks all score below 0.3, the question probably
        cannot be answered from the document.

        Args:
            query    : user's question
            documents: hybrid retriever results
            threshold: minimum rerank score to keep (default: 0.3)
            top_k    : max results to return

        Returns:
            Filtered list of (Document, score) above threshold

        Note:
            If ALL chunks fall below threshold, returns the top-1
            rather than an empty list (so the LLM can say
            "I cannot find this information" with context)
        """
        # First do normal reranking
        results = self.rerank(query, documents, top_k)

        # Filter below threshold
        filtered = [
            (doc, score) for doc, score in results
            if score >= threshold
        ]

        # Always return at least 1 result so LLM has context
        # to say "I cannot find this information"
        if not filtered and results:
            logger.warning(
                f"All chunks scored below threshold {threshold}. "
                f"Returning top-1 for LLM to handle gracefully."
            )
            return results[:1]

        if len(filtered) < len(results):
            logger.info(
                f"Threshold filter: kept {len(filtered)}/{len(results)} "
                f"chunks above {threshold}"
            )

        return filtered

    def get_model_info(self) -> dict:
        """
        Returns information about the loaded model.
        Used for logging and UI display.

        Returns:
            Dict with model_name, is_loaded, top_k
        """
        return {
            "model_name" : self.model_name,
            "is_loaded"  : self._model is not None,
            "top_k"      : self.top_k,
        }