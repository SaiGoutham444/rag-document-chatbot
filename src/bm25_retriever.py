"""
bm25_retriever.py — BM25 Keyword Search Retriever
===================================================
Implements sparse (keyword-based) retrieval using the BM25 algorithm.

BM25 excels at finding chunks containing EXACT words from the query:
  - Product codes, invoice numbers, clause references
  - Person names, place names, technical jargon
  - Any query where exact wording matters more than meaning

Complements vector search which finds SEMANTICALLY similar chunks.
Both are combined in hybrid_retriever.py using Reciprocal Rank Fusion.

Persistence:
  The BM25 index is built once and saved to bm25_index/ as a .pkl file.
  Loading from disk is instant vs rebuilding which scans all chunks.
"""

import os                                # File path operations
import re                                # Regex for tokenization
import time                              # Timing index build
import pickle                            # Serialize index to/from disk
from pathlib import Path                 # Cross-platform paths
from typing import List, Tuple, Optional # Type hints

from rank_bm25 import BM25Okapi          # The BM25 implementation
from langchain_core.documents import Document
from loguru import logger

from src.config import BM25_INDEX_PATH   # Where to save index files


# ══════════════════════════════════════════════════════════════════
# TOKENIZATION
# BM25 works on TOKENS (individual words), not raw text
# We must tokenize both the corpus AND queries the same way
# ══════════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    """
    Converts a text string into a list of lowercase tokens.

    Tokenization steps:
      1. Lowercase everything (Revenue = revenue = REVENUE)
      2. Replace punctuation with spaces (don't → don t → ["don", "t"])
      3. Split on whitespace → list of words
      4. Remove tokens shorter than 2 chars (removes "a", "I", stray chars)
      5. Remove common stop words (the, is, at, which, on...)

    WHY remove stop words?
    "What is the revenue?" → without removal: ["what","is","the","revenue"]
    "is" and "the" appear in EVERY document → IDF ≈ 0 → they add noise
    With removal: ["what", "revenue"] → cleaner, more precise matching

    WHY lowercase?
    "Revenue" and "revenue" and "REVENUE" should all match the same chunks.
    BM25 is exact-match — without lowercasing, case differences = no match.

    Args:
        text: any string (chunk content or query)

    Returns:
        List of clean lowercase tokens

    Example:
        tokenize("The Q3 Revenue was $4.2M!")
        → ["q3", "revenue", "was", "4.2m"]
    """
    if not text:
        return []

    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Replace punctuation with spaces
    # [^a-z0-9\s] matches anything that is NOT a letter, number, or whitespace
    # We keep numbers because "SKU-4821" and "clause 14.3" matter
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Step 3: Split on whitespace → raw token list
    raw_tokens = text.split()

    # Step 4 & 5: Filter short tokens and stop words
    # Common English stop words that add noise without meaning
    STOP_WORDS = {
        "the", "is", "at", "which", "on", "a", "an", "and", "or",
        "but", "in", "with", "to", "of", "for", "as", "by", "from",
        "be", "was", "are", "were", "been", "has", "have", "had",
        "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "this", "that", "these", "those", "it",
        "its", "we", "our", "you", "your", "they", "their", "he",
        "she", "his", "her", "not", "no", "so", "if", "then",
    }

    tokens = [
        token for token in raw_tokens
        if len(token) >= 2              # remove single chars
        and token not in STOP_WORDS     # remove stop words
    ]

    return tokens


# ══════════════════════════════════════════════════════════════════
# BM25 RETRIEVER CLASS
# ══════════════════════════════════════════════════════════════════

class BM25Retriever:
    """
    Keyword-based retriever using the BM25Okapi algorithm.

    Lifecycle:
      1. __init__()     : store chunks, set up paths
      2. build_index()  : tokenize all chunks, build BM25Okapi object
      3. save_index()   : pickle index to disk for persistence
      4. load_index()   : restore index from disk (skips rebuild)
      5. retrieve()     : score all chunks against a query, return top-K

    Usage:
        retriever = BM25Retriever(chunks)
        retriever.build_index()
        retriever.save_index("sample.pdf")

        # Later (after app restart):
        retriever = BM25Retriever(chunks)
        retriever.load_index("sample.pdf")

        results = retriever.retrieve("quarterly revenue", top_k=20)
        # Returns: [(Document, score), (Document, score), ...]
    """

    def __init__(self, chunks: List[Document]):
        """
        Initializes the BM25Retriever with document chunks.

        Args:
            chunks: list of Document objects from chunker.py
                    These are the chunks we will search over.

        Raises:
            ValueError: if chunks list is empty
        """
        if not chunks:
            raise ValueError(
                "BM25Retriever requires at least one chunk.\n"
                "Ensure split_documents() returned results."
            )

        # Store the original chunks — we return these as search results
        self.chunks: List[Document] = chunks

        # The BM25Okapi index object (None until build_index() is called)
        self.index: Optional[BM25Okapi] = None

        # Tokenized corpus — stored so we can inspect/debug tokenization
        self.tokenized_corpus: Optional[List[List[str]]] = None

        # Ensure the BM25 index storage directory exists
        Path(BM25_INDEX_PATH).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"BM25Retriever initialized | "
            f"Chunks: {len(chunks)}"
        )

    def _get_index_path(self, source_name: str) -> Path:
        """
        Returns the file path where this document's BM25 index is saved.

        Args:
            source_name: filename like "report.pdf"

        Returns:
            Path like bm25_index/report_pdf_bm25.pkl
        """
        # Clean filename: replace dots and spaces with underscores
        clean_name = source_name.replace(".", "_").replace(" ", "_")
        return Path(BM25_INDEX_PATH) / f"{clean_name}_bm25.pkl"

    def build_index(self) -> None:
        """
        Builds the BM25Okapi index from the stored chunks.

        Process:
          1. Tokenize every chunk's text
          2. Pass tokenized corpus to BM25Okapi constructor
          3. BM25Okapi calculates IDF scores, document lengths, avgdl

        This must be called BEFORE retrieve() if no saved index exists.

        Time complexity: O(N × L) where N=chunks, L=avg chunk length
        Typical time: ~0.5s for 200 chunks, ~2s for 1000 chunks

        Raises:
            RuntimeError: if tokenization or index build fails
        """
        try:
            logger.info(
                f"Building BM25 index for {len(self.chunks)} chunks..."
            )
            start_time = time.time()

            # Step 1: Tokenize every chunk
            # Result: list of lists — one token list per chunk
            # Example: [["revenue", "q3", "million"], ["costs", "stable"], ...]
            self.tokenized_corpus = [
                tokenize(chunk.page_content)
                for chunk in self.chunks
            ]

            # Log tokenization stats for debugging
            token_counts = [len(tokens) for tokens in self.tokenized_corpus]
            avg_tokens   = sum(token_counts) / len(token_counts)
            logger.info(
                f"  Tokenization complete | "
                f"Avg tokens per chunk: {avg_tokens:.1f}"
            )

            # Step 2: Build BM25Okapi index
            # BM25Okapi is the standard Okapi BM25 variant
            # k1=1.5, b=0.75 are the standard defaults from the original paper
            self.index = BM25Okapi(
                self.tokenized_corpus,
                k1=1.5,    # term frequency saturation
                b=0.75,    # length normalization
            )

            elapsed = time.time() - start_time
            logger.info(
                f"BM25 index built in {elapsed:.2f}s | "
                f"Corpus size: {len(self.chunks)} chunks | "
                f"Avg document length: {self.index.avgdl:.1f} tokens"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to build BM25 index: {e}"
            ) from e

    def save_index(self, source_name: str) -> None:
        """
        Saves the BM25 index to disk using pickle serialization.

        What gets saved:
          - self.index           : the BM25Okapi object (IDF scores, avgdl, etc.)
          - self.tokenized_corpus: pre-tokenized chunk texts
          - chunk metadata       : source, page, chunk_id for each chunk

        What does NOT get saved:
          - The actual chunk text (we keep self.chunks in memory)
          - The embedding model (saved separately in ChromaDB)

        Args:
            source_name: filename used to name the .pkl file

        Raises:
            RuntimeError: if index hasn't been built or save fails
        """
        try:
            if self.index is None:
                raise RuntimeError(
                    "Cannot save index — build_index() must be called first."
                )

            index_path = self._get_index_path(source_name)

            # Build the data package to serialize
            # We save everything needed to restore full functionality
            save_data = {
                "index"            : self.index,
                "tokenized_corpus" : self.tokenized_corpus,
                # Save chunk texts and metadata (not the full Document objects
                # because they may not serialize cleanly across Python versions)
                "chunk_texts"      : [c.page_content for c in self.chunks],
                "chunk_metadatas"  : [c.metadata for c in self.chunks],
                "num_chunks"       : len(self.chunks),
            }

            # Write serialized data to .pkl file
            # "wb" = write binary mode (pickle produces binary data)
            with open(index_path, "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_size_kb = index_path.stat().st_size / 1024
            logger.info(
                f"BM25 index saved to '{index_path}' | "
                f"Size: {file_size_kb:.1f} KB"
            )

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to save BM25 index to disk: {e}"
            ) from e

    def load_index(self, source_name: str) -> bool:
        """
        Loads a previously saved BM25 index from disk.

        Returns True if loaded successfully, False if no saved index exists.
        This allows the caller to decide: load from disk OR rebuild.

        Args:
            source_name: filename used to find the .pkl file

        Returns:
            True  : index loaded from disk successfully
            False : no saved index found (need to call build_index())

        Raises:
            RuntimeError: file exists but is corrupted
        """
        try:
            index_path = self._get_index_path(source_name)

            # Check if a saved index exists for this document
            if not index_path.exists():
                logger.info(
                    f"No saved BM25 index found for '{source_name}'. "
                    f"Will build from scratch."
                )
                return False

            logger.info(f"Loading BM25 index from '{index_path}'...")
            start_time = time.time()

            # Read and deserialize the pickle file
            # "rb" = read binary mode
            with open(index_path, "rb") as f:
                save_data = pickle.load(f)

            # Restore the index and tokenized corpus
            self.index             = save_data["index"]
            self.tokenized_corpus  = save_data["tokenized_corpus"]

            # Restore chunks from saved texts and metadatas
            chunk_texts     = save_data["chunk_texts"]
            chunk_metadatas = save_data["chunk_metadatas"]

            # Rebuild Document objects from saved components
            self.chunks = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(chunk_texts, chunk_metadatas)
            ]

            elapsed = time.time() - start_time
            logger.info(
                f"BM25 index loaded in {elapsed:.3f}s | "
                f"Chunks: {len(self.chunks)}"
            )
            return True

        except (KeyError, AttributeError) as e:
            raise RuntimeError(
                f"BM25 index file for '{source_name}' is corrupted: {e}\n"
                f"Delete the file and rebuild: "
                f"rm {self._get_index_path(source_name)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load BM25 index: {e}"
            ) from e

    def index_exists_on_disk(self, source_name: str) -> bool:
        """
        Checks if a saved BM25 index exists on disk for a document.
        Used to decide whether to load or rebuild without loading.

        Args:
            source_name: filename to check

        Returns:
            True if .pkl file exists, False otherwise
        """
        return self._get_index_path(source_name).exists()

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves the top-K chunks most relevant to the query using BM25.

        Process:
          1. Tokenize the query (same way we tokenized the corpus)
          2. BM25 scores every chunk against the tokenized query
          3. Sort by score descending
          4. Return top_k (Document, score) tuples

        IMPORTANT: Query must be tokenized the SAME WAY as the corpus.
        If corpus uses lowercase + no punctuation, query must too.
        Our tokenize() function handles both consistently.

        Args:
            query : user's question as a string
            top_k : how many results to return (default: 20)

        Returns:
            List of (Document, float) tuples sorted by score descending
            Score is the raw BM25 score (not normalized to 0-1)
            A score of 0.0 means no query terms found in that chunk

        Raises:
            RuntimeError: index not built or loaded yet
            ValueError  : empty query
        """
        try:
            # Guard: index must exist
            if self.index is None:
                raise RuntimeError(
                    "BM25 index not ready.\n"
                    "Call build_index() or load_index() first."
                )

            # Guard: query must not be empty
            if not query or not query.strip():
                raise ValueError(
                    "Query cannot be empty."
                )

            logger.info(
                f"BM25 retrieval | "
                f"Query: '{query[:60]}' | "
                f"Top-K: {top_k}"
            )

            # Step 1: Tokenize query the same way as the corpus
            query_tokens = tokenize(query)

            if not query_tokens:
                logger.warning(
                    f"Query '{query}' produced no tokens after filtering.\n"
                    f"All words may be stop words or too short.\n"
                    f"Returning empty results."
                )
                return []

            logger.debug(f"Query tokens: {query_tokens}")

            # Step 2: Score every chunk against the query
            # get_scores() returns a numpy array of scores, one per chunk
            # Index i in scores corresponds to self.chunks[i]
            scores = self.index.get_scores(query_tokens)

            # Step 3: Pair each chunk with its score
            chunk_score_pairs = list(zip(self.chunks, scores))

            # Step 4: Sort by score descending (highest BM25 score first)
            chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # Step 5: Take top_k results
            # Filter out zero-score chunks (no query terms found at all)
            top_results = [
                (doc, float(score))
                for doc, score in chunk_score_pairs[:top_k]
                if score > 0.0   # exclude chunks with zero relevance
            ]

            # Log results summary
            if top_results:
                top_score   = top_results[0][1]
                nonzero     = len(top_results)
                logger.info(
                    f"BM25 results: {nonzero} non-zero scored chunks | "
                    f"Top score: {top_score:.4f}"
                )
            else:
                logger.warning(
                    f"BM25 found no matching chunks for query: '{query}'\n"
                    f"Query tokens {query_tokens} matched nothing in corpus.\n"
                    f"Vector search will still run in hybrid retrieval."
                )

            return top_results

        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError(
                f"BM25 retrieval failed: {e}"
            ) from e

    def get_index_stats(self) -> dict:
        """
        Returns statistics about the current BM25 index.
        Used for logging and Streamlit sidebar display.

        Returns:
            Dict with corpus_size, avg_doc_length, is_built
        """
        if self.index is None:
            return {
                "is_built"       : False,
                "corpus_size"    : 0,
                "avg_doc_length" : 0.0,
            }

        return {
            "is_built"       : True,
            "corpus_size"    : len(self.chunks),
            "avg_doc_length" : round(self.index.avgdl, 1),
            "total_tokens"   : sum(
                len(tokens) for tokens in (self.tokenized_corpus or [])
            ),
        }


# ══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# Used by rag_pipeline.py to set up BM25 in one call
# ══════════════════════════════════════════════════════════════════

def build_or_load_bm25(
    chunks: List[Document],
    source_name: str,
) -> BM25Retriever:
    """
    Convenience function: loads existing BM25 index or builds a new one.

    This is the function rag_pipeline.py calls — it handles the
    load-vs-build decision automatically:
      - If saved index exists → load it (fast, ~0.01s)
      - If no saved index    → build it and save (slower, ~0.5-2s)

    Args:
        chunks      : document chunks from split_documents()
        source_name : filename for index naming and lookup

    Returns:
        Ready-to-use BM25Retriever with index built or loaded

    Raises:
        RuntimeError: if build or load fails
    """
    retriever = BM25Retriever(chunks)

    # Try loading from disk first (faster than rebuilding)
    if retriever.index_exists_on_disk(source_name):
        success = retriever.load_index(source_name) 
        if success:
            logger.info(f"BM25 index loaded from disk for '{source_name}'")
            return retriever
        else:
            logger.warning(
                f"BM25 load failed for '{source_name}', rebuilding..."
            )

    # Build fresh index and save for next time
    retriever.build_index()
    retriever.save_index(source_name)

    return retriever