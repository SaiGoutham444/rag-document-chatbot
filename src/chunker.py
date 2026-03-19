"""
chunker.py — Text Chunking Module
===================================
Splits large Documents into smaller overlapping chunks for retrieval.

WHY we chunk:
  LLMs have token limits. We chunk so we retrieve only the
  5 most relevant pieces (~625 tokens) instead of the whole
  document (potentially 50,000+ tokens).

WHY overlap:
  Prevents facts from being split across chunk boundaries.
  A 50-char overlap ensures a sentence started in chunk N
  also appears at the start of chunk N+1.

Default settings (tuned for RAG quality):
  chunk_size    = 500 chars  ≈ 125 tokens ≈ 3-4 sentences
  chunk_overlap = 50 chars   ≈ half a sentence
"""

import hashlib                                    # For deterministic chunk ID generation
from typing import List, Dict, Any                # Type hints

from langchain_core.documents import Document     # Standard LangChain Document class
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger                         # Colored logging

from src.config import (
    DEFAULT_CHUNK_SIZE,      # 500 characters
    DEFAULT_CHUNK_OVERLAP,   # 50 characters
    CHUNK_SEPARATORS,        # ["\n\n", "\n", ". ", " ", ""]
)


# ══════════════════════════════════════════════════════════════════
# CHUNK ID GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_chunk_id(
    source: str,
    page: int,
    chunk_index: int,
    content: str,
) -> str:
    """
    Generates a deterministic unique ID for a chunk.

    WHY deterministic (not random UUID)?
    If we re-upload the same document, the same chunk gets
    the same ID every time. This means:
    - Vector store updates instead of duplicating
    - Citations remain stable across re-uploads
    - We can detect "already processed" documents

    Format: "filename_p{page}_c{index}_{8-char-hash}"
    Example: "report.pdf_p5_c12_a1b2c3d4"

    Args:
        source      : filename e.g. "annual_report.pdf"
        page        : 1-indexed page number
        chunk_index : position of this chunk in the full list
        content     : the chunk's text (first 50 chars used in hash)

    Returns:
        String ID that is unique and stable across runs
    """
    # Build the raw string that uniquely identifies this chunk
    # We use source + page + index + first 50 chars of content
    raw = f"{source}_page{page}_chunk{chunk_index}_{content[:50]}"

    # MD5 gives a short, consistent hash (not for security — just uniqueness)
    # hexdigest()[:8] gives us 8 hex characters = 4 billion possible values
    hash_suffix = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]

    # Clean source name: replace spaces and slashes (break file systems)
    clean_source = source.replace(" ", "_").replace("/", "-").replace("\\", "-")

    # Final format: readable + unique
    return f"{clean_source}_p{page}_c{chunk_index}_{hash_suffix}"


# ══════════════════════════════════════════════════════════════════
# METADATA ENRICHMENT
# ══════════════════════════════════════════════════════════════════

def add_chunk_metadata(chunks: List[Document]) -> List[Document]:
    """
    Adds chunk-specific metadata fields to every chunk.

    RecursiveCharacterTextSplitter copies the parent Document's
    metadata (source, page, file_type) to every child chunk.
    This function ADDS additional chunk-level fields on top.

    Fields added:
        chunk_id     : unique stable ID for citations and deduplication
        chunk_index  : position 0, 1, 2, ... across ALL chunks
        total_chunks : total chunk count (context: "chunk 5 of 47")
        char_count   : character length of this chunk's content
        word_count   : approximate word count

    Args:
        chunks: raw chunks from RecursiveCharacterTextSplitter

    Returns:
        Same chunks with enriched metadata dictionaries
    """
    # Calculate total count once — used in every chunk's metadata
    total_chunks = len(chunks)

    enriched = []

    for index, chunk in enumerate(chunks):
        # Read inherited metadata from parent document
        source = chunk.metadata.get("source", "unknown")
        page   = chunk.metadata.get("page", 1)

        # Generate stable unique ID for this chunk
        chunk_id = generate_chunk_id(
            source=source,
            page=page,
            chunk_index=index,
            content=chunk.page_content,
        )

        # Build new metadata: spread existing fields, then add new ones
        # The ** (spread) operator copies all existing key-value pairs
        new_metadata = {
            **chunk.metadata,               # source, page, file_type, etc. (inherited)
            "chunk_id"    : chunk_id,       # "report.pdf_p5_c12_a1b2c3d4"
            "chunk_index" : index,          # 0, 1, 2, 3, ...
            "total_chunks": total_chunks,   # 47 (same for all chunks from this doc)
            "char_count"  : len(chunk.page_content),          # 487
            "word_count"  : len(chunk.page_content.split()),  # 91
        }

        enriched.append(Document(
            page_content=chunk.page_content,
            metadata=new_metadata,
        ))

    return enriched


# ══════════════════════════════════════════════════════════════════
# CHUNK VALIDATION
# ══════════════════════════════════════════════════════════════════

def validate_chunks(chunks: List[Document]) -> List[Document]:
    """
    Filters out invalid chunks and logs quality statistics.

    What counts as invalid?
    - Empty content (whitespace only)
    - Extremely short chunks (< 20 chars) — usually page numbers,
      section headers like "Chapter 1", or stray punctuation.
      These add noise without useful information for retrieval.

    Args:
        chunks: list of chunks to validate

    Returns:
        Filtered list with only meaningful chunks

    Raises:
        ValueError: if NO valid chunks remain after filtering
    """
    valid_chunks   = []
    invalid_count  = 0

    for chunk in chunks:
        content = chunk.page_content.strip()

        # Filter 1: must have actual content
        if not content:
            invalid_count += 1
            continue

        # Filter 2: must have at least 20 characters of content
        # This removes stray page numbers, headers, lone punctuation
        if len(content) < 20:
            invalid_count += 1
            logger.debug(
                f"Filtered short chunk ({len(content)} chars): "
                f"'{content[:50]}'"
            )
            continue

        valid_chunks.append(chunk)

    # Log how many were filtered
    if invalid_count > 0:
        logger.warning(
            f"Filtered out {invalid_count} chunks "
            f"(too short or empty) from {len(chunks)} total"
        )

    # Guard: we must have at least one valid chunk
    if not valid_chunks:
        raise ValueError(
            "No valid chunks remain after filtering.\n"
            "All chunks were empty or shorter than 20 characters.\n"
            "Check that your document contains readable text content."
        )

    return valid_chunks


# ══════════════════════════════════════════════════════════════════
# CHUNK STATISTICS
# ══════════════════════════════════════════════════════════════════

def get_chunk_statistics(chunks: List[Document]) -> Dict[str, Any]:
    """
    Computes statistics about a list of chunks.
    Used for logging, debugging, and Streamlit sidebar display.

    Args:
        chunks: list of Document chunks (after splitting)

    Returns:
        Dict with:
            total_chunks : int   — count of all chunks
            min_chars    : int   — smallest chunk size
            max_chars    : int   — largest chunk size
            avg_chars    : float — average chunk size
            total_chars  : int   — total characters across all chunks
            total_words  : int   — total words across all chunks
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "min_chars"   : 0,
            "max_chars"   : 0,
            "avg_chars"   : 0.0,
            "total_chars" : 0,
            "total_words" : 0,
        }

    # Calculate character count for every chunk
    char_counts = [len(c.page_content) for c in chunks]

    # Sum all words across all chunks
    total_words = sum(len(c.page_content.split()) for c in chunks)

    return {
        "total_chunks": len(chunks),
        "min_chars"   : min(char_counts),
        "max_chars"   : max(char_counts),
        "avg_chars"   : sum(char_counts) / len(char_counts),
        "total_chars" : sum(char_counts),
        "total_words" : total_words,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN SPLITTING FUNCTION — the only one other modules call
# ══════════════════════════════════════════════════════════════════

def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits a list of Documents into smaller overlapping chunks.

    Full pipeline:
      1. Validate inputs (not empty, overlap < size)
      2. Create RecursiveCharacterTextSplitter
      3. Split all documents (metadata is automatically inherited)
      4. Filter out empty/too-short chunks
      5. Enrich every chunk with chunk_id, chunk_index, etc.
      6. Log statistics

    Args:
        documents    : List of Documents from document_loader.load_document()
        chunk_size   : target character count per chunk (default: 500)
        chunk_overlap: overlap characters between adjacent chunks (default: 50)

    Returns:
        List of enriched Document chunks ready for embedding

    Raises:
        ValueError  : empty documents list, bad size/overlap values
        RuntimeError: splitting fails unexpectedly
    """
    try:
        # ── Guard: must have documents to split ────────────────────
        if not documents:
            raise ValueError(
                "Cannot split an empty document list.\n"
                "Ensure load_document() returned results before calling split_documents()."
            )

        # ── Guard: overlap must be smaller than chunk size ─────────
        # If overlap >= chunk_size, the splitter creates infinite loops
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size}).\n"
                f"Recommended: overlap = ~10% of chunk_size. "
                f"Example: chunk_size=500, chunk_overlap=50."
            )

        # ── Guard: chunk_size must be positive ─────────────────────
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be a positive integer. Got: {chunk_size}"
            )

        logger.info(
            f"Splitting {len(documents)} document sections | "
            f"chunk_size={chunk_size} | chunk_overlap={chunk_overlap}"
        )

        # ── Create the text splitter ────────────────────────────────
        # RecursiveCharacterTextSplitter tries each separator in order:
        # "\n\n" (paragraph) → "\n" (line) → ". " (sentence) → " " (word) → ""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=CHUNK_SEPARATORS,  # from config.py
            length_function=len,          # measure size in characters (not tokens)
            is_separator_regex=False,     # treat separators as literal strings
        )

        # ── Split all documents ─────────────────────────────────────
        # split_documents() processes each Document and copies its
        # metadata to every child chunk automatically
        raw_chunks = splitter.split_documents(documents)

        logger.info(f"  Raw chunks created: {len(raw_chunks)}")

        # ── Filter invalid chunks ───────────────────────────────────
        valid_chunks = validate_chunks(raw_chunks)

        logger.info(f"  Valid chunks after filtering: {len(valid_chunks)}")

        # ── Enrich with chunk-specific metadata ─────────────────────
        final_chunks = add_chunk_metadata(valid_chunks)

        # ── Log statistics ──────────────────────────────────────────
        stats = get_chunk_statistics(final_chunks)
        logger.info(
            f"  Chunking complete:\n"
            f"    Total chunks : {stats['total_chunks']}\n"
            f"    Avg size     : {stats['avg_chars']:.0f} chars\n"
            f"    Min size     : {stats['min_chars']} chars\n"
            f"    Max size     : {stats['max_chars']} chars\n"
            f"    Total words  : {stats['total_words']:,}"
        )

        return final_chunks

    except ValueError:
        raise   # Re-raise clean errors unchanged

    except Exception as e:
        raise RuntimeError(
            f"Unexpected error during text splitting: {str(e)}"
        ) from e
