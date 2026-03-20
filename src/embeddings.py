"""
embeddings.py — Embedding Model Module
========================================
Converts text into numerical vectors (embeddings).
Supports two providers:
  - "local"  : all-MiniLM-L6-v2 via HuggingFace (FREE, 384 dims)
  - "openai" : text-embedding-3-small via OpenAI API (paid, 1536 dims)

The embedding model is used in TWO places:
  1. At index time: embed every document chunk → store in ChromaDB
  2. At query time: embed the user's question → search ChromaDB
Both must use the SAME model — otherwise the vectors are incompatible.
"""

import time  # For timing how long embedding takes
from typing import List, Optional  # Type hints

from loguru import logger  # Colored logging

from src.config import (
    EMBEDDING_PROVIDER,  # "local" or "openai"
    OPENAI_API_KEY,  # needed if provider is "openai"
    OPENAI_EMBEDDING_MODEL,  # "text-embedding-3-small"
    LOCAL_EMBEDDING_MODEL,  # "all-MiniLM-L6-v2"
)


# ══════════════════════════════════════════════════════════════════
# EMBEDDING MODEL FACTORY
# Returns the correct embedding model based on config
# ══════════════════════════════════════════════════════════════════


def get_embedding_model(provider: Optional[str] = None):
    """
    Returns the configured embedding model instance.

    This is a factory function — it creates and returns the right
    embedding object based on EMBEDDING_PROVIDER in .env.

    WHY a factory function?
    - The rest of the code never needs to know WHICH model is used
    - Switching from local to OpenAI = change one line in .env
    - The returned object has the same interface regardless of provider

    Args:
        provider: "local" or "openai". If None, reads from config.

    Returns:
        LangChain embedding model object with .embed_documents()
        and .embed_query() methods

    Raises:
        ValueError  : unknown provider string
        RuntimeError: model fails to load
    """
    # Use config value if no override provided
    selected_provider = provider or EMBEDDING_PROVIDER

    logger.info(f"Loading embedding model — provider: '{selected_provider}'")

    # ── Local embeddings (FREE, no API key needed) ──────────────────
    if selected_provider == "local":
        return _load_local_embeddings()

    # ── OpenAI embeddings (paid, high quality) ──────────────────────
    elif selected_provider == "openai":
        return _load_openai_embeddings()

    else:
        raise ValueError(
            f"Unknown embedding provider: '{selected_provider}'\n"
            f"Valid options: 'local', 'openai'\n"
            f"Check EMBEDDING_PROVIDER in your .env file."
        )


def _load_local_embeddings():
    """
    Loads the local HuggingFace embedding model.

    Model: all-MiniLM-L6-v2
    - Size: ~90 MB download (cached after first use)
    - Dimensions: 384
    - Speed: ~1000 sentences/second on CPU
    - Quality: Excellent for English text retrieval tasks

    First run: downloads model to ~/.cache/huggingface/
    Subsequent runs: loads from cache instantly

    Returns:
        HuggingFaceEmbeddings instance

    Raises:
        RuntimeError: if model download or load fails
    """
    try:
        # Import here (not at top) so OpenAI users don't need sentence-transformers
        from langchain_community.embeddings import HuggingFaceEmbeddings

        start_time = time.time()

        logger.info(
            f"Loading local embedding model: '{LOCAL_EMBEDDING_MODEL}'\n"
            f"  First run: downloads ~90MB to ~/.cache/huggingface/\n"
            f"  Subsequent runs: loads from cache in ~2 seconds"
        )

        # model_kwargs: passed directly to the HuggingFace model
        # device="cpu" — run on CPU (change to "cuda" if you have a GPU)
        model = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            # encode_kwargs: passed to model.encode()
            # normalize_embeddings=True: vectors have length 1.0
            # This makes cosine similarity = dot product (faster computation)
            encode_kwargs={"normalize_embeddings": True},
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Local embedding model loaded in {elapsed:.1f}s | "
            f"Dimensions: 384 | Device: CPU"
        )
        return model

    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed.\n"
            "Fix: pip install sentence-transformers"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load local embedding model '{LOCAL_EMBEDDING_MODEL}': {e}\n"
            f"Check your internet connection for first-time model download."
        ) from e


def _load_openai_embeddings():
    """
    Loads the OpenAI embedding model via API.

    Model: text-embedding-3-small
    - Dimensions: 1536
    - Cost: $0.02 per million tokens
    - Quality: State of the art
    - Requires: OPENAI_API_KEY in .env

    Returns:
        OpenAIEmbeddings instance

    Raises:
        ValueError  : API key missing
        RuntimeError: API connection fails
    """
    try:
        from langchain_openai import OpenAIEmbeddings

        # Guard: API key must be set
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set in .env\n"
                "Get your key at: https://platform.openai.com/api-keys\n"
                "Or switch to free local embeddings: EMBEDDING_PROVIDER=local"
            )

        logger.info(
            f"Loading OpenAI embedding model: '{OPENAI_EMBEDDING_MODEL}' | "
            f"Dimensions: 1536"
        )

        model = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )

        logger.info("OpenAI embedding model ready")
        return model

    except ImportError:
        raise RuntimeError(
            "langchain-openai is not installed.\n" "Fix: pip install langchain-openai"
        )
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load OpenAI embedding model: {e}") from e


# ══════════════════════════════════════════════════════════════════
# EMBEDDING UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════


def embed_texts(texts: List[str], embedding_model) -> List[List[float]]:
    """
    Embeds a list of text strings into vectors.

    Used during document indexing to embed all chunks.
    Handles batching automatically (the model handles this internally).

    Args:
        texts          : list of strings to embed
        embedding_model: model returned by get_embedding_model()

    Returns:
        List of vectors — one vector per input text
        Each vector is a list of floats (384 or 1536 numbers)

    Raises:
        ValueError  : empty texts list
        RuntimeError: embedding API call fails
    """
    try:
        if not texts:
            raise ValueError(
                "Cannot embed an empty list of texts.\n"
                "Ensure split_documents() returned chunks before embedding."
            )

        logger.info(f"Embedding {len(texts)} texts...")
        start_time = time.time()

        # embed_documents() handles batching internally
        # For OpenAI: sends in batches of 2048 to respect API limits
        # For local: processes all at once on CPU
        vectors = embedding_model.embed_documents(texts)

        elapsed = time.time() - start_time
        dims = len(vectors[0]) if vectors else 0

        logger.info(
            f"Embedded {len(vectors)} texts in {elapsed:.2f}s | "
            f"Dimensions: {dims} | "
            f"Avg: {elapsed/len(texts)*1000:.1f}ms per text"
        )
        return vectors

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to embed texts: {e}\n"
            f"If using OpenAI, check your API key and rate limits."
        ) from e


def embed_query(query: str, embedding_model) -> List[float]:
    """
    Embeds a single query string for search.

    WHY a separate function from embed_texts()?
    Some models (like OpenAI) use DIFFERENT internal processing
    for queries vs documents. embed_query() signals "this is a
    search query" so the model can optimize accordingly.

    Args:
        query          : the user's question as a string
        embedding_model: model returned by get_embedding_model()

    Returns:
        Single vector as a list of floats

    Raises:
        ValueError  : empty query string
        RuntimeError: embedding fails
    """
    try:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty.\n" "Please provide a question or search term."
            )

        # embed_query() is optimized for single-text search queries
        vector = embedding_model.embed_query(query.strip())

        logger.debug(
            f"Query embedded | "
            f"Dimensions: {len(vector)} | "
            f"Query: '{query[:50]}...'"
        )
        return vector

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to embed query '{query[:50]}': {e}") from e


def compute_similarity(text1: str, text2: str, embedding_model) -> float:
    """
    Computes cosine similarity between two texts.
    Returns a score from -1.0 (opposite) to 1.0 (identical).

    Useful for debugging: "how similar is my query to this chunk?"
    Not used in the main pipeline — used for testing and exploration.

    Args:
        text1          : first text string
        text2          : second text string
        embedding_model: model returned by get_embedding_model()

    Returns:
        Float between -1.0 and 1.0

    Example:
        similarity = compute_similarity(
            "What was the Q3 revenue?",
            "Q3 total revenue reached $4.2 million.",
            model
        )
        # Returns something like 0.87 (very similar)
    """
    try:
        import numpy as np  # numpy for the dot product calculation

        # Embed both texts as query vectors
        vec1 = np.array(embedding_model.embed_query(text1))
        vec2 = np.array(embedding_model.embed_query(text2))

        # Cosine similarity = dot product of normalized vectors
        # Since we use normalize_embeddings=True, vectors already have length 1
        # So cosine similarity = simple dot product
        similarity = float(np.dot(vec1, vec2))

        logger.debug(
            f"Similarity: {similarity:.4f}\n"
            f"  Text1: '{text1[:60]}'\n"
            f"  Text2: '{text2[:60]}'"
        )
        return similarity

    except Exception as e:
        raise RuntimeError(f"Failed to compute similarity: {e}") from e
