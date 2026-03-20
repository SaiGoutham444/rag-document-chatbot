"""
test_citation_enforcer.py — Unit Tests for Citation Enforcer
============================================================
"""

import pytest
from langchain_core.documents import Document

from src.citation_enforcer import CitationEnforcer, CitationRef


class TestBuildContextWithIds:
    """Tests for build_context_with_ids()."""

    def test_contains_chunk_headers(self, citation_enforcer, sample_chunks):
        """Context must contain [CHUNK N] headers."""
        context = citation_enforcer.build_context_with_ids(sample_chunks)
        assert "[CHUNK 1" in context
        assert "SOURCE:" in context
        assert "PAGE:"   in context

    def test_contains_chunk_text(self, citation_enforcer, sample_chunks):
        """Context must contain the actual chunk text."""
        context = citation_enforcer.build_context_with_ids(sample_chunks)
        first_chunk_text = sample_chunks[0].page_content[:30]
        assert first_chunk_text in context

    def test_raises_on_empty_chunks(self, citation_enforcer):
        """build_context_with_ids() must raise ValueError for empty input."""
        with pytest.raises(ValueError):
            citation_enforcer.build_context_with_ids([])

    def test_chunk_count_matches(self, citation_enforcer, sample_chunks):
        """Context must contain a header for every chunk."""
        context = citation_enforcer.build_context_with_ids(sample_chunks)
        for i in range(1, len(sample_chunks) + 1):
            assert f"[CHUNK {i}" in context


class TestBuildCitationPrompt:
    """Tests for build_citation_prompt()."""

    def test_contains_query(self, citation_enforcer, sample_chunks):
        """Prompt must contain the user's query."""
        context = citation_enforcer.build_context_with_ids(sample_chunks)
        prompt  = citation_enforcer.build_citation_prompt(
            "What was the revenue?", context
        )
        assert "What was the revenue?" in prompt

    def test_contains_citation_rules(self, citation_enforcer, sample_chunks):
        """Prompt must contain citation rule instructions."""
        context = citation_enforcer.build_context_with_ids(sample_chunks)
        prompt  = citation_enforcer.build_citation_prompt("test query", context)
        assert "CITATION RULES" in prompt
        assert "MANDATORY"      in prompt

    def test_contains_context(self, citation_enforcer, sample_chunks):
        """Prompt must contain the context chunks."""
        context = citation_enforcer.build_context_with_ids(sample_chunks)
        prompt  = citation_enforcer.build_citation_prompt("test query", context)
        assert context in prompt

    def test_raises_on_empty_query(self, citation_enforcer, sample_chunks):
        """build_citation_prompt() must raise ValueError for empty query."""
        context = citation_enforcer.build_context_with_ids(sample_chunks)
        with pytest.raises(ValueError):
            citation_enforcer.build_citation_prompt("", context)


class TestParseCitations:
    """Tests for parse_citations()."""

    def test_extracts_all_citations(
        self, citation_enforcer, sample_citation_answer
    ):
        """parse_citations() must find all citation tags."""
        citations = citation_enforcer.parse_citations(sample_citation_answer)
        assert len(citations) == 3

    def test_extracts_correct_filename(
        self, citation_enforcer, sample_citation_answer
    ):
        """Each CitationRef must have correct filename."""
        citations = citation_enforcer.parse_citations(sample_citation_answer)
        for c in citations:
            assert c.filename == "test_report.pdf"

    def test_extracts_correct_page(
        self, citation_enforcer, sample_citation_answer
    ):
        """CitationRef page numbers must be correct."""
        citations = citation_enforcer.parse_citations(sample_citation_answer)
        pages = [c.page for c in citations]
        assert "1" in pages
        assert "2" in pages

    def test_extracts_chunk_id(
        self, citation_enforcer, sample_citation_answer
    ):
        """Each CitationRef must have a non-empty chunk_id."""
        citations = citation_enforcer.parse_citations(sample_citation_answer)
        for c in citations:
            assert c.chunk_id
            assert len(c.chunk_id) > 0

    def test_returns_empty_for_no_citations(self, citation_enforcer):
        """parse_citations() must return [] when no tags found."""
        citations = citation_enforcer.parse_citations(
            "This answer has no citation tags at all."
        )
        assert citations == []

    def test_raises_on_empty_answer(self, citation_enforcer):
        """parse_citations() must raise ValueError for empty answer."""
        with pytest.raises(ValueError):
            citation_enforcer.parse_citations("")


class TestValidateCitations:
    """Tests for validate_citations()."""

    def test_valid_citation_passes(
        self, citation_enforcer, sample_chunks, sample_citation_answer
    ):
        """Citations matching real chunk_ids must be marked valid."""
        # Use real chunk_ids from sample_chunks
        real_id = sample_chunks[0].metadata["chunk_id"]
        answer  = f"Some claim [SOURCE: test.pdf | PAGE: 1 | CHUNK: {real_id}]"

        citations = citation_enforcer.parse_citations(answer)
        report    = citation_enforcer.validate_citations(citations, sample_chunks)

        assert len(report.valid_citations) == 1
        assert len(report.invalid_citations) == 0

    def test_hallucinated_citation_flagged(
        self, citation_enforcer, sample_chunks
    ):
        """Citations with fake chunk_ids must be marked invalid."""
        answer = (
            "Some claim "
            "[SOURCE: test.pdf | PAGE: 1 | CHUNK: completely_fake_id_xyz]"
        )
        citations = citation_enforcer.parse_citations(answer)
        report    = citation_enforcer.validate_citations(citations, sample_chunks)

        assert len(report.invalid_citations) == 1
        assert len(report.warnings) > 0

    def test_coverage_score_calculation(
        self, citation_enforcer, sample_chunks
    ):
        """Coverage score = valid / total citations."""
        real_id = sample_chunks[0].metadata["chunk_id"]
        answer  = (
            f"Claim 1 [SOURCE: test.pdf | PAGE: 1 | CHUNK: {real_id}]. "
            f"Claim 2 [SOURCE: test.pdf | PAGE: 1 | CHUNK: fake_id_xyz]."
        )
        citations = citation_enforcer.parse_citations(answer)
        report    = citation_enforcer.validate_citations(citations, sample_chunks)

        # 1 valid, 1 invalid → score = 0.5
        assert report.coverage_score == pytest.approx(0.5)

    def test_no_citations_gives_zero_score(
        self, citation_enforcer, sample_chunks
    ):
        """Answer with no citation tags must score 0.0."""
        citations = citation_enforcer.parse_citations(
            "This answer has no citations at all."
        )
        report = citation_enforcer.validate_citations(citations, sample_chunks)
        assert report.coverage_score == 0.0


class TestEnforceCitations:
    """Tests for the master enforce_citations() function."""

    def test_cannot_find_response_is_valid(
        self, citation_enforcer, sample_chunks
    ):
        """'Cannot find' response must be marked valid with score 1.0."""
        enforced = citation_enforcer.enforce_citations(
            answer           = "I cannot find this information in the provided document.",
            retrieved_chunks = sample_chunks,
        )
        assert enforced.is_valid       == True
        assert enforced.report.coverage_score == 1.0
        assert enforced.citations      == []

    def test_valid_answer_returns_is_valid_true(
        self, citation_enforcer, sample_chunks
    ):
        """Answer with valid citations must return is_valid=True."""
        real_id = sample_chunks[0].metadata["chunk_id"]
        answer  = (
            f"The revenue was $4.2M "
            f"[SOURCE: test.pdf | PAGE: 1 | CHUNK: {real_id}]."
        )
        enforced = citation_enforcer.enforce_citations(answer, sample_chunks)
        assert enforced.is_valid == True

    def test_returns_enforced_answer_object(
        self, citation_enforcer, sample_chunks, sample_citation_answer
    ):
        """enforce_citations() must return an EnforcedAnswer object."""
        from src.citation_enforcer import EnforcedAnswer
        enforced = citation_enforcer.enforce_citations(
            sample_citation_answer, sample_chunks
        )
        assert isinstance(enforced, EnforcedAnswer)
        assert isinstance(enforced.answer,    str)
        assert isinstance(enforced.citations, list)
        assert isinstance(enforced.is_valid,  bool)