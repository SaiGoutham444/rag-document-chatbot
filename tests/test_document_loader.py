"""
test_document_loader.py — Unit Tests for Document Loader
=========================================================
Tests every loader function and the master load_document() router.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.document_loader import (
    load_document,
    load_txt,
    load_csv,
    validate_file,
    get_document_info,
)


class TestValidateFile:
    """Tests for the validate_file() function."""

    def test_raises_on_missing_file(self, tmp_path):
        """validate_file() must raise FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError):
            validate_file(str(tmp_path / "nonexistent.pdf"))

    def test_raises_on_unsupported_extension(self, tmp_path):
        """validate_file() must raise ValueError for unsupported extensions."""
        bad_file = tmp_path / "script.py"
        bad_file.write_text("print('hello')")
        with pytest.raises(ValueError, match="Unsupported file type"):
            validate_file(str(bad_file))

    def test_raises_on_empty_file(self, tmp_path):
        """validate_file() must raise ValueError for empty files."""
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            validate_file(str(empty))

    def test_raises_on_empty_path_string(self):
        """validate_file() must raise ValueError for empty string input."""
        with pytest.raises(ValueError):
            validate_file("")

    def test_returns_path_object_for_valid_file(self, sample_txt_path):
        """validate_file() must return a Path object for valid files."""
        result = validate_file(sample_txt_path)
        assert isinstance(result, Path)

    def test_valid_file_path_matches_input(self, sample_txt_path):
        """Returned Path must point to the same file as input."""
        result = validate_file(sample_txt_path)
        assert result.exists()


class TestLoadTxt:
    """Tests for the load_txt() function."""

    def test_returns_list_of_documents(self, sample_txt_path):
        """load_txt() must return a non-empty list of Document objects."""
        docs = load_document(sample_txt_path)
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_document_has_page_content(self, sample_txt_path):
        """Each Document must have non-empty page_content."""
        docs = load_document(sample_txt_path)
        for doc in docs:
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0

    def test_metadata_includes_source(self, sample_txt_path):
        """Each Document must have 'source' in metadata."""
        docs = load_document(sample_txt_path)
        for doc in docs:
            assert "source" in doc.metadata
            assert doc.metadata["source"].endswith(".txt")

    def test_metadata_includes_file_type(self, sample_txt_path):
        """Each Document must have file_type='txt' in metadata."""
        docs = load_document(sample_txt_path)
        assert docs[0].metadata["file_type"] == "txt"

    def test_metadata_includes_page_number(self, sample_txt_path):
        """Each Document must have 'page' in metadata."""
        docs = load_document(sample_txt_path)
        for doc in docs:
            assert "page" in doc.metadata
            assert isinstance(doc.metadata["page"], int)

    def test_content_matches_file_content(self, sample_txt_path, tmp_path):
        """Document content must match what's in the file."""
        test_content = "Hello this is a test document with unique content xyz123."
        txt_file = tmp_path / "unique.txt"
        txt_file.write_text(test_content)
        docs = load_document(str(txt_file))
        assert "unique content xyz123" in docs[0].page_content


class TestLoadCsv:
    """Tests for the load_csv() function."""

    def test_returns_one_document_per_row(self, sample_csv_path):
        """load_csv() must return one Document per data row."""
        docs = load_document(sample_csv_path)
        # sample_csv has 4 data rows (excluding header)
        assert len(docs) == 4

    def test_row_content_includes_column_names(self, sample_csv_path):
        """Each Document must contain column names in its content."""
        docs = load_document(sample_csv_path)
        # Every row should contain column names like "Name:", "Department:"
        for doc in docs:
            assert "Name:" in doc.page_content or "Department:" in doc.page_content

    def test_metadata_has_row_number(self, sample_csv_path):
        """Each Document must have 'row_number' in metadata."""
        docs = load_document(sample_csv_path)
        for doc in docs:
            assert "row_number" in doc.metadata

    def test_metadata_has_columns_list(self, sample_csv_path):
        """Each Document must have 'columns' list in metadata."""
        docs = load_document(sample_csv_path)
        assert "columns" in docs[0].metadata
        assert isinstance(docs[0].metadata["columns"], list)


class TestLoadDocument:
    """Tests for the master load_document() router."""

    def test_routes_txt_correctly(self, sample_txt_path):
        """load_document() must correctly route .txt files."""
        docs = load_document(sample_txt_path)
        assert docs[0].metadata["file_type"] == "txt"

    def test_routes_csv_correctly(self, sample_csv_path):
        """load_document() must correctly route .csv files."""
        docs = load_document(sample_csv_path)
        assert docs[0].metadata["file_type"] == "csv"

    def test_raises_for_nonexistent_file(self):
        """load_document() must raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_document("does_not_exist.pdf")

    def test_raises_for_unsupported_type(self, tmp_path):
        """load_document() must raise ValueError for unsupported types."""
        bad = tmp_path / "file.xyz"
        bad.write_text("content")
        with pytest.raises(ValueError):
            load_document(str(bad))


class TestGetDocumentInfo:
    """Tests for the get_document_info() utility."""

    def test_returns_dict(self, sample_docs):
        """get_document_info() must return a dictionary."""
        info = get_document_info(sample_docs)
        assert isinstance(info, dict)

    def test_contains_required_keys(self, sample_docs):
        """get_document_info() must contain all required keys."""
        info = get_document_info(sample_docs)
        required = ["source_name", "file_type", "total_sections",
                    "total_words", "total_chars"]
        for key in required:
            assert key in info, f"Missing key: {key}"

    def test_returns_empty_dict_for_empty_list(self):
        """get_document_info() must return {} for empty input."""
        info = get_document_info([])
        assert info == {}

    def test_word_count_is_positive(self, sample_docs):
        """total_words must be > 0 for non-empty documents."""
        info = get_document_info(sample_docs)
        assert info["total_words"] > 0