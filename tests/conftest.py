"""
conftest.py — Shared pytest Fixtures
======================================
Fixtures defined here are automatically available to ALL test files.
No imports needed — pytest injects them by name.

Fixtures provided:
  sample_pdf_path       : path to test PDF
  sample_txt_path       : path to test TXT file
  sample_docs           : pre-built list of Document objects
  sample_chunks         : pre-built list of chunked Documents
  mock_llm              : fake LLM that returns deterministic responses
  mock_embedding_model  : fake embedder that returns random vectors
  temp_chroma_dir       : temporary ChromaDB directory (auto-cleaned)
  citation_enforcer     : real CitationEnforcer instance
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from src.citation_enforcer import CitationEnforcer
from src.chunker import split_documents


# ══════════════════════════════════════════════════════════════════
# FILE FIXTURES
# ══════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_pdf_path(tmp_path) -> str:
    """
    Creates a minimal test PDF file and returns its path.
    Uses tmp_path (pytest built-in) which auto-cleans after test.
    """
    # Try to use the real sample PDF if it exists
    real_pdf = Path("data/sample_docs/resume.pdf")
    if real_pdf.exists():
        return str(real_pdf)

    # Otherwise create a minimal text file as fallback
    # (avoids needing a real PDF for basic tests)
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text(
        "This is a test document about quarterly revenue.\n"
        "The Q3 revenue was $4.2 million.\n"
        "Operating costs were $2.1 million.\n"
        "Net profit margin improved to 34 percent.\n"
    )
    return str(txt_file)


@pytest.fixture
def sample_txt_path(tmp_path) -> str:
    """
    Creates a plain text test file and returns its path.
    """
    txt_file = tmp_path / "test_document.txt"
    txt_file.write_text(
        "Introduction\n\n"
        "This document covers financial results for Q3 2024.\n"
        "The company reported strong growth across all segments.\n\n"
        "Revenue\n\n"
        "Total revenue reached $4.2 million, a 23% increase.\n"
        "Enterprise sales drove the majority of growth.\n\n"
        "Costs\n\n"
        "Operating costs remained stable at $2.1 million.\n"
        "Cost reduction was driven by automation initiatives.\n\n"
        "Profit\n\n"
        "Net profit margin improved to 34 percent.\n"
        "This represents a 5 percent improvement from Q2.\n"
    )
    return str(txt_file)


@pytest.fixture
def sample_csv_path(tmp_path) -> str:
    """
    Creates a test CSV file and returns its path.
    """
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(
        "Name,Department,Salary,Skills\n"
        "Alice,Engineering,95000,Python SQL Git\n"
        "Bob,Marketing,75000,Excel PowerPoint\n"
        "Charlie,Engineering,105000,Java Python Docker\n"
        "Diana,HR,65000,Communication Excel\n"
    )
    return str(csv_file)


# ══════════════════════════════════════════════════════════════════
# DOCUMENT FIXTURES
# ══════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_docs() -> List[Document]:
    """
    Returns a pre-built list of Document objects for testing.
    Simulates what load_document() would return for a PDF.
    """
    return [
        Document(
            page_content=(
                "The company achieved record revenue in Q3 2024. "
                "Total sales reached $4.2 million dollars representing "
                "a 23 percent increase year over year. The growth was "
                "driven primarily by enterprise sales in the technology sector."
            ),
            metadata={
                "source": "test_report.pdf",
                "page": 1,
                "file_type": "pdf",
                "total_pages": 3,
                "file_size_kb": 245.3,
            },
        ),
        Document(
            page_content=(
                "Operating costs remained stable at $2.1 million. "
                "The cost reduction was driven by automation of manual processes. "
                "Headcount remained flat at 142 employees across all departments. "
                "R&D spending increased by 12 percent to support new product lines."
            ),
            metadata={
                "source": "test_report.pdf",
                "page": 2,
                "file_type": "pdf",
                "total_pages": 3,
                "file_size_kb": 245.3,
            },
        ),
        Document(
            page_content=(
                "Net profit margin improved to 34 percent this quarter. "
                "This represents a 5 percent improvement compared to Q2. "
                "The board approved a dividend increase of 8 percent. "
                "Full year guidance has been raised to $16 million in revenue."
            ),
            metadata={
                "source": "test_report.pdf",
                "page": 3,
                "file_type": "pdf",
                "total_pages": 3,
                "file_size_kb": 245.3,
            },
        ),
    ]


@pytest.fixture
def sample_chunks(sample_docs) -> List[Document]:
    """
    Returns chunked version of sample_docs.
    Uses real split_documents() with small chunk size for testing.
    """
    return split_documents(
        documents=sample_docs,
        chunk_size=200,
        chunk_overlap=20,
    )


# ══════════════════════════════════════════════════════════════════
# MOCK FIXTURES
# ══════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_llm():
    """
    Returns a fake LLM that returns deterministic responses.
    Never calls a real API — free, fast, offline.

    The mock response contains proper citation tags so
    citation enforcement tests work correctly.
    """
    mock = MagicMock()

    # Simulate .invoke() returning an AIMessage-like object
    mock_response = MagicMock()
    mock_response.content = (
        "The Q3 revenue was $4.2 million "
        "[SOURCE: test_report.pdf | PAGE: 1 | CHUNK: test_report_pdf_p1_c0_abc12345]. "
        "Operating costs were $2.1 million "
        "[SOURCE: test_report.pdf | PAGE: 2 | CHUNK: test_report_pdf_p2_c0_def67890]."
    )
    mock.invoke.return_value = mock_response

    # Simulate .stream() yielding tokens
    tokens = ["The ", "Q3 ", "revenue ", "was ", "$4.2 ", "million."]
    mock.stream.return_value = iter([MagicMock(content=token) for token in tokens])

    return mock


@pytest.fixture
def mock_embedding_model():
    """
    Returns a fake embedding model that produces random vectors.
    Avoids downloading the 90MB HuggingFace model during tests.
    Vectors are random but consistent shape (384 dimensions).
    """
    mock = MagicMock()

    # embed_documents: returns list of 384-dim random vectors
    def fake_embed_documents(texts):
        return [np.random.rand(384).tolist() for _ in texts]

    # embed_query: returns single 384-dim random vector
    def fake_embed_query(text):
        return np.random.rand(384).tolist()

    mock.embed_documents = fake_embed_documents
    mock.embed_query = fake_embed_query

    return mock


# ══════════════════════════════════════════════════════════════════
# INFRASTRUCTURE FIXTURES
# ══════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_chroma_dir(tmp_path) -> str:
    """
    Creates a temporary ChromaDB directory.
    Automatically deleted after each test.
    Prevents tests from polluting the real chroma_db/ folder.
    """
    chroma_dir = tmp_path / "test_chroma_db"
    chroma_dir.mkdir()
    return str(chroma_dir)


@pytest.fixture
def temp_bm25_dir(tmp_path) -> str:
    """
    Creates a temporary BM25 index directory.
    Automatically deleted after each test.
    """
    bm25_dir = tmp_path / "test_bm25_index"
    bm25_dir.mkdir()
    return str(bm25_dir)


@pytest.fixture
def citation_enforcer() -> CitationEnforcer:
    """
    Returns a real CitationEnforcer instance for testing.
    Uses default min_citation_score from config.
    """
    return CitationEnforcer()


@pytest.fixture
def sample_citation_answer() -> str:
    """
    Returns a sample LLM answer with citation tags.
    Used for testing citation parsing and validation.
    """
    return (
        "The Q3 revenue was $4.2 million "
        "[SOURCE: test_report.pdf | PAGE: 1 | CHUNK: test_report_pdf_p1_c0_abc12345]. "
        "Operating costs were $2.1 million "
        "[SOURCE: test_report.pdf | PAGE: 2 | CHUNK: test_report_pdf_p2_c0_def67890]. "
        "The profit margin improved to 34 percent "
        "[SOURCE: test_report.pdf | PAGE: 3 | CHUNK: test_report_pdf_p3_c0_ghi11111]."
    )


@pytest.fixture
def sample_answer_no_citations() -> str:
    """
    Returns a sample answer WITHOUT citation tags.
    Used to test citation enforcement failure detection.
    """
    return (
        "The Q3 revenue was $4.2 million. "
        "Operating costs were $2.1 million. "
        "The profit margin improved to 34 percent."
    )
