"""
vector_store.py — ChromaDB Vector Store Module
================================================
Manages storing and querying document embeddings using ChromaDB.

ChromaDB runs LOCALLY — no server, no cloud, no account needed.
All data is saved in the chroma_db/ folder on your disk.
Data persists between app restarts automatically.

Key concepts:
  Collection : like a database table — one per document set
  Embedding  : the vector representation of a text chunk
  Metadata   : source, page, chunk_id stored alongside each vector
  Query      : embed a question → find nearest vectors → return chunks
"""

import time  # Timing operations
from typing import List, Tuple, Optional, Dict, Any  # Type hints

import chromadb  # The vector database
from chromadb.config import Settings  # ChromaDB configuration options
from langchain_core.documents import Document
from loguru import logger

from src.config import CHROMA_DB_PATH  # Path where ChromaDB saves files
from src.embeddings import embed_texts, embed_query


# ══════════════════════════════════════════════════════════════════
# CHROMADB CLIENT
# ══════════════════════════════════════════════════════════════════


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Creates a persistent ChromaDB client.

    PersistentClient saves all data to disk at CHROMA_DB_PATH.
    The folder is created automatically if it doesn't exist.
    Data survives app restarts — no need to re-embed documents.

    Returns:
        chromadb.PersistentClient instance

    Raises:
        RuntimeError: if ChromaDB cannot initialize at the given path
    """
    try:
        logger.info(f"Initializing ChromaDB at: '{CHROMA_DB_PATH}'")

        # PersistentClient: saves data to disk (unlike in-memory EphemeralClient)
        # path: the folder where ChromaDB stores its files
        client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(
                # anonymized_telemetry: ChromaDB sends usage stats by default
                # We disable this for privacy
                anonymized_telemetry=False,
            ),
        )

        logger.info("ChromaDB client ready")
        return client

    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize ChromaDB at '{CHROMA_DB_PATH}': {e}\n"
            f"Check that the path is writable and disk has enough space."
        ) from e


# ══════════════════════════════════════════════════════════════════
# COLLECTION MANAGEMENT
# A Collection = one set of embedded documents
# We use one collection per uploaded document
# ══════════════════════════════════════════════════════════════════


def get_collection_name(source_name: str) -> str:
    """
    Converts a filename into a valid ChromaDB collection name.

    ChromaDB collection names must:
    - Be 3-63 characters long
    - Contain only letters, numbers, underscores, hyphens
    - Start and end with a letter or number

    Args:
        source_name: filename like "Annual Report Q3 2024.pdf"

    Returns:
        Valid collection name like "Annual_Report_Q3_2024_pdf"
    """
    import re

    # Replace spaces and dots with underscores
    name = source_name.replace(" ", "_").replace(".", "_")

    # Remove any characters that aren't letters, numbers, underscores, hyphens
    name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)

    # Truncate to 63 characters (ChromaDB limit)
    name = name[:63]

    # Ensure it starts with a letter or number (not underscore)
    if name and not name[0].isalnum():
        name = "doc_" + name

    # Ensure minimum length of 3
    if len(name) < 3:
        name = name + "_collection"

    return name


def collection_exists(client: chromadb.PersistentClient, source_name: str) -> bool:
    """
    Checks if a collection already exists for a given document.
    Used to avoid re-embedding documents that were already processed.

    Args:
        client     : ChromaDB client from get_chroma_client()
        source_name: filename of the document

    Returns:
        True if collection exists, False otherwise
    """
    try:
        collection_name = get_collection_name(source_name)

        # list_collections() returns all existing collection names
        existing = [col.name for col in client.list_collections()]

        exists = collection_name in existing
        logger.info(f"Collection '{collection_name}' exists: {exists}")
        return exists

    except Exception as e:
        logger.warning(f"Could not check collection existence: {e}")
        return False


def create_or_load_collection(
    client: chromadb.PersistentClient,
    source_name: str,
) -> chromadb.Collection:
    """
    Creates a new collection or loads an existing one.

    get_or_create_collection() is idempotent:
    - If collection exists: loads it (no data loss)
    - If collection doesn't exist: creates it fresh

    Args:
        client     : ChromaDB client
        source_name: filename used to name the collection

    Returns:
        ChromaDB Collection object

    Raises:
        RuntimeError: if collection cannot be created or loaded
    """
    try:
        collection_name = get_collection_name(source_name)

        logger.info(f"Getting/creating collection: '{collection_name}'")

        # get_or_create_collection: safe to call multiple times
        # metadata: stores info about this collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "source": source_name,
                "description": f"Embeddings for document: {source_name}",
            },
        )

        count = collection.count()
        logger.info(
            f"Collection '{collection_name}' ready | " f"Existing documents: {count}"
        )
        return collection

    except Exception as e:
        raise RuntimeError(
            f"Failed to create/load collection for '{source_name}': {e}"
        ) from e


def delete_collection(
    client: chromadb.PersistentClient,
    source_name: str,
) -> bool:
    """
    Deletes a collection and all its embeddings.
    Called when a user clicks "Delete Document" in the UI.

    Args:
        client     : ChromaDB client
        source_name: filename of the document to delete

    Returns:
        True if deleted, False if collection didn't exist
    """
    try:
        collection_name = get_collection_name(source_name)

        if not collection_exists(client, source_name):
            logger.warning(
                f"Collection '{collection_name}' doesn't exist — nothing to delete"
            )
            return False

        client.delete_collection(name=collection_name)
        logger.info(f"Deleted collection '{collection_name}'")
        return True

    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        return False


def get_collection_info(
    client: chromadb.PersistentClient,
    source_name: str,
) -> Dict[str, Any]:
    """
    Returns metadata about a collection.
    Used by the Streamlit sidebar to show indexing stats.

    Args:
        client     : ChromaDB client
        source_name: filename

    Returns:
        Dict with collection_name, document_count, exists
    """
    try:
        collection_name = get_collection_name(source_name)
        exists = collection_exists(client, source_name)

        if not exists:
            return {
                "collection_name": collection_name,
                "document_count": 0,
                "exists": False,
            }

        collection = client.get_collection(collection_name)
        count = collection.count()

        return {
            "collection_name": collection_name,
            "document_count": count,
            "exists": True,
        }

    except Exception as e:
        logger.warning(f"Could not get collection info: {e}")
        return {"exists": False, "document_count": 0}


# ══════════════════════════════════════════════════════════════════
# ADDING DOCUMENTS TO THE STORE
# ══════════════════════════════════════════════════════════════════


def add_documents_to_store(
    chunks: List[Document],
    source_name: str,
    embedding_model,
    client: Optional[chromadb.PersistentClient] = None,
) -> chromadb.Collection:
    """
    Embeds document chunks and stores them in ChromaDB.

    Full process:
      1. Create ChromaDB client (if not provided)
      2. Create or load collection for this document
      3. Extract text content from all chunks
      4. Embed all texts → list of vectors
      5. Build metadata dicts for each chunk
      6. Add everything to ChromaDB collection
      7. Return collection for immediate querying

    Args:
        chunks         : list of Document chunks from chunker.py
        source_name    : filename (used to name the collection)
        embedding_model: model from get_embedding_model()
        client         : existing ChromaDB client (or None to create new)

    Returns:
        ChromaDB Collection with all chunks embedded and stored

    Raises:
        ValueError  : empty chunks list
        RuntimeError: embedding or storage fails
    """
    try:
        if not chunks:
            raise ValueError(
                "Cannot add empty chunks to vector store.\n"
                "Ensure split_documents() returned results."
            )

        logger.info(
            f"Adding {len(chunks)} chunks to vector store | "
            f"Document: '{source_name}'"
        )
        start_time = time.time()

        # Create client if not provided
        if client is None:
            client = get_chroma_client()

        # Get or create collection for this document
        collection = create_or_load_collection(client, source_name)

        # ── Extract data from chunks ────────────────────────────────

        # The actual text content of each chunk
        texts = [chunk.page_content for chunk in chunks]

        # Unique IDs for each chunk — ChromaDB requires unique string IDs
        ids = [
            chunk.metadata.get("chunk_id", f"chunk_{i}")
            for i, chunk in enumerate(chunks)
        ]

        # Metadata for each chunk — stored alongside the vector
        # ChromaDB only accepts: str, int, float, bool values in metadata
        # Lists and nested dicts are NOT supported — we convert them
        metadatas = []
        for chunk in chunks:
            meta = {}
            for key, val in chunk.metadata.items():
                # Convert lists to comma-separated strings
                if isinstance(val, list):
                    meta[key] = ", ".join(str(v) for v in val)
                # Convert all values to str/int/float/bool only
                elif isinstance(val, (str, int, float, bool)):
                    meta[key] = val
                else:
                    meta[key] = str(val)
            metadatas.append(meta)

        # ── Embed all texts ─────────────────────────────────────────
        logger.info("Embedding chunks...")
        vectors = embed_texts(texts, embedding_model)

        # ── Add to ChromaDB ─────────────────────────────────────────
        # upsert: insert OR update if ID already exists
        # This prevents duplicates if the same document is uploaded twice
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

        elapsed = time.time() - start_time
        final_count = collection.count()

        logger.info(
            f"Vector store updated | "
            f"Chunks added: {len(chunks)} | "
            f"Total in collection: {final_count} | "
            f"Time: {elapsed:.2f}s"
        )
        return collection

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to add documents to vector store: {e}") from e


# ══════════════════════════════════════════════════════════════════
# QUERYING THE STORE
# ══════════════════════════════════════════════════════════════════


def query_vector_store(
    query: str,
    source_name: str,
    embedding_model,
    top_k: int = 20,
    client: Optional[chromadb.PersistentClient] = None,
) -> List[Tuple[Document, float]]:
    """
    Searches the vector store for chunks most similar to the query.

    Process:
      1. Embed the query string → query vector
      2. ChromaDB finds the top_k nearest vectors (HNSW search)
      3. Returns chunks with their similarity scores

    Args:
        query          : user's question as a string
        source_name    : which document to search in
        embedding_model: same model used during indexing
        top_k          : how many results to return (default: 20)
        client         : existing ChromaDB client (or None)

    Returns:
        List of (Document, score) tuples sorted by score descending
        Score is cosine similarity: 0.0 (unrelated) to 1.0 (identical)

    Raises:
        ValueError  : collection doesn't exist for this document
        RuntimeError: query fails
    """
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # Create client if not provided
        if client is None:
            client = get_chroma_client()

        # Guard: ensure collection exists
        if not collection_exists(client, source_name):
            raise ValueError(
                f"No vector store found for '{source_name}'.\n"
                f"The document needs to be processed first.\n"
                f"Call add_documents_to_store() before querying."
            )

        collection = client.get_collection(get_collection_name(source_name))

        logger.info(
            f"Querying vector store | "
            f"Document: '{source_name}' | "
            f"Top-K: {top_k}"
        )

        # Embed the query using the same model used during indexing
        query_vector = embed_query(query, embedding_model)

        # Query ChromaDB for nearest neighbors
        # n_results: how many to return
        # include: what data to include in response
        results = collection.query(
            query_embeddings=[query_vector],  # wrap in list (batch query API)
            n_results=min(top_k, collection.count()),  # can't return more than exists
            include=["documents", "metadatas", "distances"],
        )

        # ── Parse ChromaDB response ─────────────────────────────────
        # ChromaDB returns results in this structure:
        # results["documents"][0] = list of text strings
        # results["metadatas"][0] = list of metadata dicts
        # results["distances"][0] = list of distances (lower = more similar)
        # The [0] index is because we sent a single query (batch of 1)

        documents_list = results["documents"][0]
        metadatas_list = results["metadatas"][0]
        distances_list = results["distances"][0]

        # Convert distances to similarity scores
        # ChromaDB returns L2 distances (lower = closer)
        # We convert to similarity: similarity = 1 - (distance / 2)
        # This gives us 1.0 for identical, 0.0 for very different
        output = []
        for text, meta, distance in zip(documents_list, metadatas_list, distances_list):
            # Convert metadata values back (ChromaDB returns strings for lists)
            similarity_score = max(0.0, 1.0 - (distance / 2.0))

            doc = Document(
                page_content=text,
                metadata=meta,
            )
            output.append((doc, similarity_score))

        # Sort by similarity score descending (highest first)
        output.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"Vector search complete | "
            f"Results: {len(output)} | "
            f"Top score: {output[0][1]:.4f}"
            if output
            else "No results"
        )
        return output

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Vector store query failed: {e}") from e
