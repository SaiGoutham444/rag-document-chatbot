"""
document_loader.py
==================
Loads PDF, DOCX, TXT, CSV, and HTML files into standardized
LangChain Document objects with full source metadata.
"""

import os
import re
from pathlib import Path
from typing import List, Dict

import pandas as pd
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document
from loguru import logger

from src.config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB


def validate_file(file_path: str) -> Path:
    """
    Validates a file before loading it.
    Checks: not empty string, exists, is a file, not empty bytes,
    not too large, extension is supported.

    Args:
        file_path: string path to the file

    Returns:
        Path object if valid

    Raises:
        FileNotFoundError: file does not exist
        ValueError: empty, too large, or unsupported type
    """
    # Guard: path string must not be empty
    if not file_path or not file_path.strip():
        raise ValueError("file_path cannot be empty.")

    # Convert string to Path object
    path = Path(file_path)

    # Check the file exists on disk
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: '{file_path}'\n"
            f"Check the path and make sure the file was uploaded."
        )

    # Check it is a file not a directory
    if not path.is_file():
        raise ValueError(
            f"'{file_path}' is a directory, not a file.\n"
            f"Provide the full path to a document."
        )

    # Check file is not empty (0 bytes)
    file_size_bytes = path.stat().st_size
    if file_size_bytes == 0:
        raise ValueError(
            f"'{path.name}' is empty (0 bytes).\n"
            f"Please upload a file that contains content."
        )

    # Check file is not too large
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"'{path.name}' is {file_size_mb:.1f} MB — too large.\n"
            f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
        )

    # Check extension is supported
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(SUPPORTED_EXTENSIONS.keys())
        raise ValueError(
            f"Unsupported file type: '{extension}'\n"
            f"Supported types: {supported_list}"
        )

    logger.info(
        f"File validated: '{path.name}' | "
        f"{file_size_mb:.2f} MB | "
        f"Type: {SUPPORTED_EXTENSIONS[extension]}"
    )
    return path


def load_pdf(file_path: Path) -> List[Document]:
    """
    Loads a PDF and returns one Document per page.
    Page numbers are 1-indexed in metadata for human-readable citations.

    Args:
        file_path: validated Path to a .pdf file

    Returns:
        List[Document] one per non-empty page

    Raises:
        ValueError: all pages empty (scanned/image PDF)
        RuntimeError: corrupted or password-protected file
    """
    try:
        logger.info(f"Loading PDF: '{file_path.name}'")

        # PyPDFLoader splits by page, returns 0-indexed page numbers
        loader = PyPDFLoader(str(file_path))
        raw_pages = loader.load()

        if not raw_pages:
            raise ValueError(
                f"'{file_path.name}' appears to have no pages."
            )

        total_pages = len(raw_pages)
        file_size_kb = round(file_path.stat().st_size / 1024, 1)

        enriched_docs = []
        empty_page_count = 0

        for raw_doc in raw_pages:
            content = raw_doc.page_content.strip()

            # Skip pages with no text (scanned/image pages)
            if not content:
                empty_page_count += 1
                continue

            # Convert 0-indexed page to 1-indexed for citations
            page_number = raw_doc.metadata.get("page", 0) + 1

            enriched_docs.append(Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "page": page_number,
                    "file_type": "pdf",
                    "total_pages": total_pages,
                    "file_size_kb": file_size_kb,
                }
            ))

        if empty_page_count > 0:
            logger.warning(
                f"{empty_page_count}/{total_pages} pages were empty and skipped."
            )

        if not enriched_docs:
            raise ValueError(
                f"No text extracted from '{file_path.name}'.\n"
                f"This PDF may contain only scanned images.\n"
                f"Fix: Open in Google Drive > Open with Google Docs to OCR it."
            )

        logger.info(f"Loaded {len(enriched_docs)} pages from '{file_path.name}'")
        return enriched_docs

    except (ValueError, FileNotFoundError):
        raise

    except Exception as e:
        raise RuntimeError(
            f"Failed to load PDF '{file_path.name}': {str(e)}\n"
            f"Possible causes: corrupted file or password-protected PDF."
        ) from e


def load_docx(file_path: Path) -> List[Document]:
    """
    Loads a Word .docx file as a single Document.
    Word has no fixed page boundaries in its data model,
    so we load the entire content and let the chunker split it.

    Args:
        file_path: validated Path to a .docx file

    Returns:
        List[Document] with one element

    Raises:
        ValueError: empty file
        RuntimeError: corrupted or old .doc format
    """
    try:
        logger.info(f"Loading DOCX: '{file_path.name}'")

        loader = Docx2txtLoader(str(file_path))
        raw_docs = loader.load()

        if not raw_docs or not raw_docs[0].page_content.strip():
            raise ValueError(
                f"'{file_path.name}' appears to be empty or image-only."
            )

        file_size_kb = round(file_path.stat().st_size / 1024, 1)
        content = raw_docs[0].page_content.strip()
        word_count = len(content.split())

        doc = Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "page": 1,
                "file_type": "docx",
                "file_size_kb": file_size_kb,
                "word_count": word_count,
            }
        )

        logger.info(
            f"Loaded DOCX: '{file_path.name}' | "
            f"{word_count:,} words | {file_size_kb} KB"
        )
        return [doc]

    except (ValueError, FileNotFoundError):
        raise

    except Exception as e:
        raise RuntimeError(
            f"Failed to load DOCX '{file_path.name}': {str(e)}\n"
            f"Only .docx (Word 2007+) is supported. Resave .doc files as .docx."
        ) from e


def load_txt(file_path: Path) -> List[Document]:
    """
    Loads a plain text file as a single Document.
    Tries UTF-8 first, falls back to latin-1 for legacy files.

    Args:
        file_path: validated Path to a .txt file

    Returns:
        List[Document] with one element

    Raises:
        ValueError: empty file
        RuntimeError: cannot decode with any known encoding
    """
    try:
        logger.info(f"Loading TXT: '{file_path.name}'")

        content = None

        # Try UTF-8 first (modern standard)
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            raw = loader.load()
            content = raw[0].page_content if raw else ""
        except UnicodeDecodeError:
            logger.warning(
                f"UTF-8 failed for '{file_path.name}', retrying with latin-1..."
            )

        # Fall back to latin-1 for legacy files
        if not content:
            try:
                loader = TextLoader(str(file_path), encoding="latin-1")
                raw = loader.load()
                content = raw[0].page_content if raw else ""
            except UnicodeDecodeError:
                raise RuntimeError(
                    f"Cannot decode '{file_path.name}' with UTF-8 or latin-1.\n"
                    f"Try resaving the file as UTF-8 in your text editor."
                )

        if not content or not content.strip():
            raise ValueError(
                f"'{file_path.name}' is empty or contains only whitespace."
            )

        file_size_kb = round(file_path.stat().st_size / 1024, 1)
        line_count = content.count("\n") + 1

        doc = Document(
            page_content=content.strip(),
            metadata={
                "source": file_path.name,
                "page": 1,
                "file_type": "txt",
                "file_size_kb": file_size_kb,
                "line_count": line_count,
            }
        )

        logger.info(
            f"Loaded TXT: '{file_path.name}' | "
            f"{line_count:,} lines | {file_size_kb} KB"
        )
        return [doc]

    except (ValueError, RuntimeError, FileNotFoundError):
        raise

    except Exception as e:
        raise RuntimeError(
            f"Failed to load TXT '{file_path.name}': {str(e)}"
        ) from e


def load_csv(file_path: Path) -> List[Document]:
    """
    Loads a CSV file — one Document per data row.
    Each row becomes: "ColumnA: val1 | ColumnB: val2 | ..."
    This format works well for both keyword and semantic search.

    Args:
        file_path: validated Path to a .csv file

    Returns:
        List[Document] one per data row

    Raises:
        ValueError: empty file or no valid rows
        RuntimeError: not valid CSV format
    """
    try:
        logger.info(f"Loading CSV: '{file_path.name}'")

        df = pd.read_csv(str(file_path), encoding_errors="replace")

        if df.empty:
            raise ValueError(
                f"'{file_path.name}' contains no data rows."
            )

        file_size_kb = round(file_path.stat().st_size / 1024, 1)
        column_names = list(df.columns)

        docs = []
        for row_index, row in df.iterrows():
            # Build "Column: Value | Column: Value" string
            row_parts = []
            for col, val in row.items():
                if pd.notna(val) and str(val).strip():
                    row_parts.append(f"{col}: {str(val).strip()}")

            # Skip fully empty rows
            if not row_parts:
                continue

            row_text = " | ".join(row_parts)

            docs.append(Document(
                page_content=row_text,
                metadata={
                    "source": file_path.name,
                    "page": 1,
                    "file_type": "csv",
                    "row_number": int(row_index) + 2,
                    "columns": column_names,
                    "file_size_kb": file_size_kb,
                }
            ))

        if not docs:
            raise ValueError(
                f"No valid data rows found in '{file_path.name}'."
            )

        logger.info(
            f"Loaded CSV: '{file_path.name}' | "
            f"{len(docs):,} rows | {len(column_names)} columns"
        )
        return docs

    except (ValueError, FileNotFoundError):
        raise

    except Exception as e:
        raise RuntimeError(
            f"Failed to load CSV '{file_path.name}': {str(e)}"
        ) from e


def load_html(file_path: Path) -> List[Document]:
    """
    Loads an HTML file, strips all tags, returns clean text.
    Primary: UnstructuredHTMLLoader
    Fallback: regex-based tag stripping

    Args:
        file_path: validated Path to a .html or .htm file

    Returns:
        List[Document] with one element of clean text

    Raises:
        ValueError: no text after tag removal
        RuntimeError: cannot parse file
    """
    try:
        logger.info(f"Loading HTML: '{file_path.name}'")

        content = None

        # Primary: UnstructuredHTMLLoader understands DOM structure
        try:
            loader = UnstructuredHTMLLoader(str(file_path))
            raw_docs = loader.load()
            if raw_docs and raw_docs[0].page_content.strip():
                content = raw_docs[0].page_content.strip()
        except Exception as primary_error:
            logger.warning(
                f"UnstructuredHTMLLoader failed: {primary_error}. "
                f"Trying regex fallback..."
            )

        # Fallback: manual regex stripping
        if not content:
            content = _html_regex_fallback(file_path)

        if not content or not content.strip():
            raise ValueError(
                f"No text found in '{file_path.name}' after removing HTML tags."
            )

        file_size_kb = round(file_path.stat().st_size / 1024, 1)
        word_count = len(content.split())

        doc = Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "page": 1,
                "file_type": "html",
                "file_size_kb": file_size_kb,
                "word_count": word_count,
            }
        )

        logger.info(
            f"Loaded HTML: '{file_path.name}' | "
            f"{word_count:,} words after tag removal"
        )
        return [doc]

    except (ValueError, FileNotFoundError):
        raise

    except Exception as e:
        raise RuntimeError(
            f"Failed to load HTML '{file_path.name}': {str(e)}"
        ) from e


def _html_regex_fallback(file_path: Path) -> str:
    """
    Fallback HTML text extractor using regex.
    Used when UnstructuredHTMLLoader fails on malformed HTML.

    Args:
        file_path: Path to the HTML file

    Returns:
        String with HTML tags removed and whitespace normalized
    """
    raw = file_path.read_text(encoding="utf-8", errors="replace")

    # Remove script blocks
    raw = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)

    # Remove style blocks
    raw = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)

    # Remove all HTML tags
    raw = re.sub(r"<[^>]+>", " ", raw)

    # Decode common HTML entities
    entity_map = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&nbsp;": " ", "&#39;": "'", "&quot;": '"',
    }
    for entity, replacement in entity_map.items():
        raw = raw.replace(entity, replacement)

    # Normalize whitespace
    raw = re.sub(r"\s+", " ", raw).strip()

    return raw


def load_document(file_path: str) -> List[Document]:
    """
    Master loader — auto-detects file type and routes to correct loader.
    This is the only function other modules need to import.

    Args:
        file_path: path to any supported document as a string

    Returns:
        List[Document] with standardized metadata on every document

    Raises:
        FileNotFoundError: file does not exist
        ValueError: unsupported type, empty, or too large
        RuntimeError: corrupted or unreadable file
    """
    # Validate first — fail fast with clear errors
    path = validate_file(file_path)

    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(path)

    elif extension == ".docx":
        return load_docx(path)

    elif extension == ".txt":
        return load_txt(path)

    elif extension == ".csv":
        return load_csv(path)

    elif extension in (".html", ".htm"):
        return load_html(path)

    else:
        raise ValueError(f"No loader implemented for extension: '{extension}'")


def get_document_info(docs: List[Document]) -> Dict:
    """
    Returns a summary dictionary about a loaded document list.
    Used by the Streamlit sidebar to show upload statistics.

    Args:
        docs: list returned by load_document()

    Returns:
        Dict with source_name, file_type, total_sections,
        total_words, total_chars, file_size_kb, total_pages
    """
    if not docs:
        return {}

    first = docs[0]
    total_words = sum(len(d.page_content.split()) for d in docs)
    total_chars = sum(len(d.page_content) for d in docs)

    return {
        "source_name": first.metadata.get("source", "Unknown"),
        "file_type": first.metadata.get("file_type", "unknown"),
        "total_sections": len(docs),
        "total_words": total_words,
        "total_chars": total_chars,
        "file_size_kb": first.metadata.get("file_size_kb", 0),
        "total_pages": first.metadata.get("total_pages", len(docs)),
    }