"""
citation_enforcer.py — Citation Enforcement Module
====================================================
Ensures every LLM answer is grounded in retrieved chunks
with verifiable citations.

3-layer system:
  Layer 1: Prompt engineering — formats context with chunk IDs,
           instructs LLM to cite every factual claim
  Layer 2: Output validation — parses citations from LLM response,
           verifies they reference actual retrieved chunks
  Layer 3: Report generation — coverage score, valid/invalid citations

WHY this matters:
  Without citations, LLM answers cannot be verified.
  With citations, every claim traces to a specific chunk.
  Users can click the citation and read the source text.
  Invalid citations (hallucinated chunk IDs) are flagged.
"""

import re  # Regex for parsing citations
from dataclasses import dataclass, field  # Clean data containers
from typing import List  # Type hints

from langchain_core.documents import Document
from loguru import logger

from src.config import MIN_CITATION_SCORE  # Threshold from config


# ══════════════════════════════════════════════════════════════════
# DATA CLASSES
# Clean containers for citation data passed between functions
# ══════════════════════════════════════════════════════════════════


@dataclass
class CitationRef:
    """
    Represents a single citation extracted from the LLM response.

    Example citation tag in LLM output:
      [SOURCE: report.pdf | PAGE: 3 | CHUNK: report_pdf_p3_c2_a1b2c3d4]

    Fields:
      chunk_id  : the chunk_id string (links to retrieved chunk)
      filename  : source document name
      page      : page number as string
      raw_tag   : the original citation string as written by LLM
      is_valid  : True if chunk_id matches a retrieved chunk
    """

    chunk_id: str
    filename: str
    page: str
    raw_tag: str
    is_valid: bool = False  # set by validate_citations()


@dataclass
class CitationReport:
    """
    Full validation report for one LLM response.

    Fields:
      all_citations    : every CitationRef extracted from the answer
      valid_citations  : citations that match actual retrieved chunks
      invalid_citations: citations with chunk IDs not in retrieved set
      coverage_score   : valid / total (0.0 to 1.0)
      total_claims     : approximate number of factual sentences
      warnings         : list of warning messages
    """

    all_citations: List[CitationRef] = field(default_factory=list)
    valid_citations: List[CitationRef] = field(default_factory=list)
    invalid_citations: List[CitationRef] = field(default_factory=list)
    coverage_score: float = 0.0
    total_claims: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class EnforcedAnswer:
    """
    Final output of the citation enforcement process.

    Fields:
      answer        : the LLM's response text (unchanged)
      citations     : list of CitationRef objects from the answer
      report        : full CitationReport with validation details
      is_valid      : True if coverage_score >= MIN_CITATION_SCORE
      source_chunks : the actual Document objects that were cited
    """

    answer: str
    citations: List[CitationRef]
    report: CitationReport
    is_valid: bool
    source_chunks: List[Document] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# CITATION ENFORCER CLASS
# ══════════════════════════════════════════════════════════════════


class CitationEnforcer:
    """
    Manages the full citation enforcement pipeline.

    Usage:
        enforcer = CitationEnforcer()

        # Build the prompt (before calling LLM)
        context = enforcer.build_context_with_ids(reranked_chunks)
        prompt  = enforcer.build_citation_prompt(query, context)

        # After LLM responds:
        enforced = enforcer.enforce_citations(
            answer           = llm_response,
            retrieved_chunks = reranked_chunks,
        )

        print(enforced.answer)         # the answer text
        print(enforced.is_valid)       # True if well-cited
        print(enforced.citations)      # list of CitationRef objects
        print(enforced.report.coverage_score)  # 0.0 to 1.0
    """

    def __init__(self, min_citation_score: float = MIN_CITATION_SCORE):
        """
        Args:
            min_citation_score: minimum coverage to consider answer valid
                                 Default from config: 0.7
        """
        self.min_citation_score = min_citation_score

        logger.info(
            f"CitationEnforcer initialized | "
            f"Min citation score: {min_citation_score}"
        )

    # ──────────────────────────────────────────────────────────────
    # LAYER 1: CONTEXT FORMATTING + PROMPT BUILDING
    # ──────────────────────────────────────────────────────────────

    def build_context_with_ids(
        self,
        chunks: List[Document],
    ) -> str:
        """
        Formats retrieved chunks into a labeled context string for the prompt.

        Each chunk gets a clear header showing its ID, source, and page.
        This is what the LLM reads as "the document context."

        The [CHUNK N] labels are what the LLM uses in citations.
        We map those back to actual chunks during validation.

        Example output:
            [CHUNK 1 | SOURCE: report.pdf | PAGE: 3 | ID: report_pdf_p3_c2_a1b2]
            The Q3 revenue reached $4.2 million representing a 23% increase...
            ────────────────────────────────────────────────────────────────────

            [CHUNK 2 | SOURCE: report.pdf | PAGE: 4 | ID: report_pdf_p4_c1_b3c4]
            Operating costs remained stable at $2.1 million. The cost reduction...
            ────────────────────────────────────────────────────────────────────

        Args:
            chunks: reranked Document chunks (top-5 from reranker)

        Returns:
            Formatted string ready to insert into the LLM prompt

        Raises:
            ValueError: if chunks list is empty
        """
        if not chunks:
            raise ValueError(
                "Cannot build context from empty chunks list.\n"
                "Ensure reranker returned results before building context."
            )

        context_parts = []

        for i, chunk in enumerate(chunks, start=1):
            # Extract metadata for the header label
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "?")
            chunk_id = chunk.metadata.get("chunk_id", f"chunk_{i}")

            # Build the chunk header — this is what the LLM will cite
            header = (
                f"[CHUNK {i} | "
                f"SOURCE: {source} | "
                f"PAGE: {page} | "
                f"ID: {chunk_id}]"
            )

            # Combine header + content + separator
            chunk_block = f"{header}\n" f"{chunk.page_content.strip()}\n" f"{'─' * 68}"

            context_parts.append(chunk_block)

        # Join all chunks with blank line between them
        full_context = "\n\n".join(context_parts)

        logger.debug(
            f"Context built | "
            f"{len(chunks)} chunks | "
            f"{len(full_context)} characters"
        )
        return full_context

    def build_citation_prompt(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Builds the complete prompt that forces the LLM to cite sources.

        This prompt has 4 parts:
          1. Role definition — who the LLM is
          2. Citation rules — the exact format required (MANDATORY)
          3. Context — the labeled document chunks
          4. Question — the user's actual question

        Every word is deliberate:
          - "MANDATORY" in caps gets LLM attention
          - "immediately after" prevents end-of-paragraph citations
          - Exact fallback phrase is detectable programmatically
          - "NEVER invent" explicitly forbids hallucination

        Args:
            query  : user's question string
            context: formatted context from build_context_with_ids()

        Returns:
            Complete prompt string ready to send to the LLM

        Raises:
            ValueError: empty query or context
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not context or not context.strip():
            raise ValueError("Context cannot be empty.")

        # The complete prompt template
        # Every section has a clear purpose
        prompt = f"""You are a precise document Q&A assistant.
You answer questions ONLY using the provided document chunks below.
You NEVER use outside knowledge or make assumptions beyond what the chunks say.

════════════════════════════════════════════════════
CITATION RULES — MANDATORY — READ CAREFULLY
════════════════════════════════════════════════════

RULE 1: EVERY sentence must end with a citation tag. NO EXCEPTIONS.
        Count your sentences. Count your citation tags.
        The numbers MUST match exactly.
        
        WRONG (missing citation on sentence 2):
        "They use SQL [SOURCE: f.pdf | PAGE: 1 | CHUNK: abc]. They use Git."
        
        RIGHT (every sentence cited):
        "They use SQL [SOURCE: f.pdf | PAGE: 1 | CHUNK: abc]. 
         They use Git [SOURCE: f.pdf | PAGE: 1 | CHUNK: abc]."

RULE 2: Use EXACTLY this citation format after EVERY sentence:
        [SOURCE: {{filename}} | PAGE: {{page_number}} | CHUNK: {{chunk_id}}]

RULE 3: If multiple chunks support a claim, cite ALL of them.

RULE 4: If the answer is NOT found in the chunks, respond EXACTLY:
        "I cannot find this information in the provided document."

RULE 5: NEVER invent chunk IDs, page numbers, or filenames.
        Only use IDs that appear in the chunk headers below.

RULE 6: Keep answers to 2-3 sentences maximum. Be concise and direct.
        Fewer sentences = easier to cite every one correctly.

════════════════════════════════════════════════════
PROVIDED DOCUMENT CHUNKS
════════════════════════════════════════════════════

{context}

════════════════════════════════════════════════════
QUESTION
════════════════════════════════════════════════════

{query}

Answer in 2-3 sentences. EVERY sentence must have a citation tag:"""

        logger.debug(
            f"Citation prompt built | "
            f"{len(prompt)} characters | "
            f"Query: '{query[:50]}'"
        )
        return prompt

    # ──────────────────────────────────────────────────────────────
    # LAYER 2: PARSING CITATIONS FROM LLM RESPONSE
    # ──────────────────────────────────────────────────────────────

    def parse_citations(self, answer: str) -> List[CitationRef]:
        """
        Extracts all [SOURCE: ...] citation tags from the LLM response.

        Uses regex to find every citation tag in the answer.
        Does NOT validate whether citations are real — just extracts them.
        Validation happens in validate_citations().

        Citation format we parse:
          [SOURCE: report.pdf | PAGE: 3 | CHUNK: report_pdf_p3_c2_a1b2]

        Args:
            answer: the raw LLM response string

        Returns:
            List of CitationRef objects (one per citation found)
            Empty list if no citations found

        Raises:
            ValueError: empty answer string
        """
        if not answer or not answer.strip():
            raise ValueError("Cannot parse citations from empty answer.")

        # Regex pattern explanation:
        # \[SOURCE:\s*   → literal "[SOURCE:" + optional spaces
        # ([^\|]+)       → capture group 1: filename (anything except |)
        # \|\s*PAGE:\s*  → literal "| PAGE:" + spaces
        # ([^\|]+)       → capture group 2: page number
        # \|\s*CHUNK:\s* → literal "| CHUNK:" + spaces
        # ([^\]]+)       → capture group 3: chunk ID (anything except ])
        # \]             → closing bracket
        pattern = re.compile(
            r"\[SOURCE:\s*([^\|]+)\|\s*PAGE:\s*([^\|]+)\|\s*CHUNK:\s*([^\]]+)\]",
            re.IGNORECASE,  # case-insensitive matching
        )

        citations = []
        seen_tags = set()  # track duplicates

        for match in pattern.finditer(answer):
            raw_tag = match.group(0).strip()  # full [SOURCE: ...] string
            filename = match.group(1).strip()  # "report.pdf"
            page = match.group(2).strip()  # "3"
            chunk_id = match.group(3).strip()  # "report_pdf_p3_c2_a1b2"

            # Skip exact duplicate citations (same tag appearing twice)
            if raw_tag in seen_tags:
                continue
            seen_tags.add(raw_tag)

            citations.append(
                CitationRef(
                    chunk_id=chunk_id,
                    filename=filename,
                    page=page,
                    raw_tag=raw_tag,
                    is_valid=False,  # set during validation
                )
            )

        logger.debug(
            f"Parsed {len(citations)} citations from answer " f"({len(answer)} chars)"
        )
        return citations

    # ──────────────────────────────────────────────────────────────
    # LAYER 2: VALIDATING CITATIONS
    # ──────────────────────────────────────────────────────────────

    def validate_citations(
        self,
        citations: List[CitationRef],
        retrieved_chunks: List[Document],
    ) -> CitationReport:
        """
        Validates that each citation references an actual retrieved chunk.

        A citation is VALID if its chunk_id matches the chunk_id
        in the metadata of one of our retrieved chunks.

        A citation is INVALID (hallucinated) if the LLM invented
        a chunk_id that doesn't exist in our retrieved set.

        Computes coverage_score = valid_citations / total_citations
        (Not total_claims — that would require NLP claim detection)

        Args:
            citations       : list from parse_citations()
            retrieved_chunks: the actual chunks we retrieved and showed LLM

        Returns:
            CitationReport with valid/invalid breakdown and coverage score
        """
        # Build a set of valid chunk IDs from retrieved chunks
        # Using a set for O(1) lookup instead of O(N) list search
        valid_chunk_ids = {
            chunk.metadata.get("chunk_id", "") for chunk in retrieved_chunks
        }

        report = CitationReport()
        report.all_citations = citations
        report.total_claims = len(citations)

        for citation in citations:
            # Check if cited chunk_id exists in our retrieved set
            if citation.chunk_id in valid_chunk_ids:
                citation.is_valid = True
                report.valid_citations.append(citation)
            else:
                citation.is_valid = False
                report.invalid_citations.append(citation)
                report.warnings.append(
                    f"Hallucinated citation: chunk_id '{citation.chunk_id}' "
                    f"was not in retrieved chunks. "
                    f"LLM may have invented this source."
                )
                logger.warning(
                    f"Invalid citation detected: '{citation.chunk_id}' "
                    f"not in retrieved chunk IDs: {valid_chunk_ids}"
                )

        # Calculate coverage score
        if len(citations) > 0:
            report.coverage_score = len(report.valid_citations) / len(citations)
        else:
            # No citations found at all
            report.coverage_score = 0.0
            report.warnings.append(
                "No citations found in the answer. "
                "The LLM did not follow citation rules."
            )

        logger.info(
            f"Citation validation complete | "
            f"Total: {len(citations)} | "
            f"Valid: {len(report.valid_citations)} | "
            f"Invalid: {len(report.invalid_citations)} | "
            f"Score: {report.coverage_score:.2f}"
        )
        return report

    # ──────────────────────────────────────────────────────────────
    # LAYER 3: LINKING CITATIONS TO SOURCE CHUNKS
    # ──────────────────────────────────────────────────────────────

    def link_citations_to_chunks(
        self,
        citations: List[CitationRef],
        retrieved_chunks: List[Document],
    ) -> List[Document]:
        """
        Finds the actual Document objects for each valid citation.

        Used by the Streamlit UI to show the source chunk text
        next to each citation in the answer.

        Args:
            citations       : validated CitationRef list
            retrieved_chunks: the chunks we retrieved

        Returns:
            List of Document objects that were cited (valid only)
            Preserves order of first appearance in citations
        """
        # Build lookup: chunk_id → Document
        chunk_lookup = {
            chunk.metadata.get("chunk_id", ""): chunk for chunk in retrieved_chunks
        }

        linked_chunks = []
        seen_ids = set()

        for citation in citations:
            if citation.is_valid and citation.chunk_id not in seen_ids:
                chunk = chunk_lookup.get(citation.chunk_id)
                if chunk:
                    linked_chunks.append(chunk)
                    seen_ids.add(citation.chunk_id)

        logger.debug(
            f"Linked {len(linked_chunks)} unique source chunks "
            f"to {len(citations)} citations"
        )
        return linked_chunks

    def count_factual_claims(self, answer: str) -> int:
        """
        Estimates the number of factual claims in an answer.

        A simple heuristic: count sentences that are NOT:
          - Questions (end with ?)
          - The "cannot find" fallback phrase
          - Very short (< 20 chars, likely transitional)

        This is used as the denominator for coverage_score
        when we want to measure "what % of claims have citations"
        rather than "what % of citations are valid."

        Args:
            answer: LLM response text

        Returns:
            Integer count of estimated factual claims
        """
        # Check for the "cannot find" fallback — 0 claims
        if "cannot find this information" in answer.lower():
            return 0

        # Split into sentences on period, exclamation, or newline
        sentences = re.split(r"[.!\n]+", answer)

        # Count sentences that look like factual statements
        factual_count = 0
        for sentence in sentences:
            sentence = sentence.strip()

            # Skip empty sentences
            if not sentence:
                continue

            # Skip questions
            if sentence.endswith("?"):
                continue

            # Skip very short sentences (transitions, headers)
            if len(sentence) < 20:
                continue

            factual_count += 1

        return max(factual_count, 1)  # at least 1 to avoid division by zero

    # ──────────────────────────────────────────────────────────────
    # MAIN ENFORCEMENT FUNCTION
    # The single function rag_pipeline.py calls
    # ──────────────────────────────────────────────────────────────

    def enforce_citations(
        self,
        answer: str,
        retrieved_chunks: List[Document],
    ) -> EnforcedAnswer:
        """
        Full citation enforcement pipeline for one LLM response.

        Steps:
          1. Parse all [SOURCE: ...] tags from the answer
          2. Validate each citation against retrieved chunks
          3. Link valid citations to actual Document objects
          4. Compute coverage score and is_valid flag
          5. Return EnforcedAnswer with everything packaged

        Args:
            answer          : raw LLM response string
            retrieved_chunks: the chunks we passed to the LLM

        Returns:
            EnforcedAnswer dataclass with:
              .answer        → original answer text (unchanged)
              .citations     → list of CitationRef objects
              .report        → CitationReport with scores
              .is_valid      → True if coverage >= min_citation_score
              .source_chunks → Document objects for cited chunks

        Raises:
            ValueError: empty answer or chunks
        """
        try:
            if not answer or not answer.strip():
                raise ValueError("Answer cannot be empty.")

            if not retrieved_chunks:
                raise ValueError(
                    "retrieved_chunks cannot be empty.\n"
                    "Cannot validate citations without the source chunks."
                )

            logger.info(
                f"Enforcing citations | "
                f"Answer length: {len(answer)} chars | "
                f"Retrieved chunks: {len(retrieved_chunks)}"
            )

            # ── Step 1: Handle "cannot find" response ───────────────
            # If LLM correctly said it can't answer, that's valid
            if "cannot find this information" in answer.lower():
                logger.info(
                    "LLM correctly responded with 'cannot find' — "
                    "no citations needed"
                )
                return EnforcedAnswer(
                    answer=answer,
                    citations=[],
                    report=CitationReport(
                        coverage_score=1.0,  # perfect — correct behavior
                        warnings=["LLM correctly declined to answer"],
                    ),
                    is_valid=True,
                    source_chunks=[],
                )

            # ── Step 2: Parse citations from answer ─────────────────
            citations = self.parse_citations(answer)

            # ── Step 3: Validate citations ───────────────────────────
            report = self.validate_citations(citations, retrieved_chunks)

            # ── Step 4: Link citations to source chunks ──────────────
            source_chunks = self.link_citations_to_chunks(citations, retrieved_chunks)

            # ── Step 5: Determine if answer meets quality threshold ──
            is_valid = report.coverage_score >= self.min_citation_score

            if not is_valid:
                logger.warning(
                    f"Citation score {report.coverage_score:.2f} below "
                    f"threshold {self.min_citation_score}. "
                    f"Answer flagged as potentially unreliable."
                )

            enforced = EnforcedAnswer(
                answer=answer,
                citations=citations,
                report=report,
                is_valid=is_valid,
                source_chunks=source_chunks,
            )

            logger.info(
                f"Citation enforcement complete | "
                f"Citations: {len(citations)} | "
                f"Valid: {len(report.valid_citations)} | "
                f"Score: {report.coverage_score:.2f} | "
                f"Is valid: {is_valid}"
            )
            return enforced

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Citation enforcement failed: {e}") from e
