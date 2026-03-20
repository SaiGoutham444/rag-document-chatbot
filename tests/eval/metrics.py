"""
metrics.py — RAG Evaluation Metrics
=====================================
Computes the 5 key metrics for evaluating RAG pipeline quality.

All metrics return float values between 0.0 and 1.0.
Higher is always better.

Metrics:
  1. faithfulness        — are claims grounded in context?
  2. answer_relevancy    — does answer address the question?
  3. context_recall      — did retriever find all relevant chunks?
  4. context_precision   — are retrieved chunks relevant?
  5. citation_coverage   — what % of claims have citations?
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger


# ══════════════════════════════════════════════════════════════════
# RESULT DATA CLASS
# ══════════════════════════════════════════════════════════════════


@dataclass
class EvalResult:
    """
    Stores evaluation scores for one question-answer pair.

    All scores: 0.0 (worst) → 1.0 (best)
    overall_score: weighted average of all metrics
    """

    question: str
    ground_truth: str
    generated_answer: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    citation_coverage: float = 0.0
    overall_score: float = 0.0
    category: str = ""
    error: Optional[str] = None


@dataclass
class EvalReport:
    """
    Aggregated evaluation report across all questions.
    Contains per-question results and overall averages.
    """

    results: List[EvalResult] = field(default_factory=list)
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_recall: float = 0.0
    avg_context_precision: float = 0.0
    avg_citation_coverage: float = 0.0
    avg_overall_score: float = 0.0
    total_questions: int = 0
    passed_questions: int = 0
    quality_gate_passed: bool = False


# ══════════════════════════════════════════════════════════════════
# INDIVIDUAL METRICS
# ══════════════════════════════════════════════════════════════════


def compute_faithfulness(
    answer: str,
    context_chunks: List[Document],
) -> float:
    """
    Measures how grounded the answer is in the retrieved context.

    Approach (heuristic — real RAGAS uses LLM for this):
      1. Split answer into sentences (claims)
      2. For each claim, check if key words appear in context
      3. Score = claims_supported / total_claims

    A claim is "supported" if >50% of its non-stop-word tokens
    appear somewhere in the retrieved context text.

    This is a HEURISTIC approximation.
    Production systems use an LLM-as-judge for better accuracy.

    Args:
        answer        : generated answer string
        context_chunks: retrieved Document chunks shown to LLM

    Returns:
        Float 0.0-1.0
    """
    if not answer or not context_chunks:
        return 0.0

    # Build the full context text for checking
    context_text = " ".join(chunk.page_content.lower() for chunk in context_chunks)

    # Split answer into sentences/claims
    sentences = [
        s.strip()
        for s in re.split(r"[.!?\n]+", answer)
        if len(s.strip()) > 20  # skip very short fragments
    ]

    if not sentences:
        return 0.0

    # Stop words to ignore during token matching
    STOP_WORDS = {
        "the",
        "is",
        "at",
        "which",
        "on",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "with",
        "to",
        "of",
        "for",
        "as",
        "by",
        "from",
        "be",
        "was",
        "are",
        "were",
        "this",
        "that",
        "it",
        "we",
        "they",
        "have",
        "has",
        "had",
    }

    supported_count = 0

    for sentence in sentences:
        # Remove citation tags before checking
        clean = re.sub(r"\[SOURCE:[^\]]+\]", "", sentence)
        clean = clean.lower()

        # Get meaningful tokens from this claim
        tokens = [
            t
            for t in re.findall(r"\b[a-z0-9]+\b", clean)
            if t not in STOP_WORDS and len(t) > 2
        ]

        if not tokens:
            supported_count += 1  # empty claim = skip
            continue

        # Check what % of tokens appear in context
        found = sum(1 for t in tokens if t in context_text)
        support_ratio = found / len(tokens)

        if support_ratio >= 0.5:  # at least half the tokens in context
            supported_count += 1

    score = supported_count / len(sentences)
    logger.debug(
        f"Faithfulness: {score:.3f} "
        f"({supported_count}/{len(sentences)} claims supported)"
    )
    return round(score, 4)


def compute_answer_relevancy(
    question: str,
    answer: str,
    embedding_model=None,
) -> float:
    """
    Measures how relevant the answer is to the question.

    Approach:
      If embedding_model provided: use cosine similarity
      Otherwise: use keyword overlap heuristic

    A high score means the answer directly addresses the question.
    A low score means the answer is off-topic.

    Args:
        question       : the user's original question
        answer         : generated answer string
        embedding_model: optional — for semantic similarity scoring

    Returns:
        Float 0.0-1.0
    """
    if not question or not answer:
        return 0.0

    # Check for "cannot find" response — it's relevant (correctly refused)
    if "cannot find this information" in answer.lower():
        return 1.0

    if embedding_model is not None:
        # Semantic similarity using embeddings
        try:
            import numpy as np

            q_vec = np.array(embedding_model.embed_query(question))
            a_vec = np.array(embedding_model.embed_query(answer))

            # Cosine similarity
            similarity = float(
                np.dot(q_vec, a_vec)
                / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec) + 1e-8)
            )
            score = max(0.0, min(1.0, similarity))
            logger.debug(f"Answer relevancy (semantic): {score:.3f}")
            return round(score, 4)

        except Exception as e:
            logger.warning(f"Semantic relevancy failed: {e}. Using keyword fallback.")

    # Keyword overlap heuristic (fallback)
    # Extract key question terms
    question_words = set(
        w.lower() for w in re.findall(r"\b[a-z]+\b", question.lower()) if len(w) > 3
    )
    answer_words = set(
        w.lower() for w in re.findall(r"\b[a-z]+\b", answer.lower()) if len(w) > 3
    )

    if not question_words:
        return 0.5

    # Jaccard similarity between question and answer keywords
    overlap = question_words & answer_words
    union = question_words | answer_words
    score = len(overlap) / len(union) if union else 0.0

    # Scale to [0.3, 1.0] range — pure keyword match underestimates relevancy
    score = 0.3 + (score * 0.7)
    score = max(0.0, min(1.0, score))

    logger.debug(f"Answer relevancy (keyword): {score:.3f}")
    return round(score, 4)


def compute_context_recall(
    question: str,
    ground_truth: str,
    context_chunks: List[Document],
) -> float:
    """
    Measures whether retrieved chunks contain the information
    needed to answer the question (per the ground truth).

    Approach:
      1. Extract key facts from ground truth answer
      2. Check if each fact appears in retrieved context
      3. Score = facts_found_in_context / total_facts

    Args:
        question      : user's question
        ground_truth  : the correct answer we expect
        context_chunks: chunks retrieved by the pipeline

    Returns:
        Float 0.0-1.0
    """
    if not ground_truth or not context_chunks:
        return 0.0

    # Build context text
    context_text = " ".join(chunk.page_content.lower() for chunk in context_chunks)

    # Extract key tokens from ground truth
    STOP_WORDS = {
        "the",
        "is",
        "at",
        "which",
        "on",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "with",
        "to",
        "of",
        "for",
        "as",
        "by",
        "from",
        "be",
        "was",
        "are",
        "were",
        "this",
        "that",
        "it",
        "candidate",
        "has",
        "have",
    }

    gt_tokens = [
        t
        for t in re.findall(r"\b[a-z0-9]+\b", ground_truth.lower())
        if t not in STOP_WORDS and len(t) > 2
    ]

    if not gt_tokens:
        return 0.5  # can't evaluate without tokens

    # Check how many ground truth tokens appear in context
    found = sum(1 for t in gt_tokens if t in context_text)
    score = found / len(gt_tokens)

    logger.debug(
        f"Context recall: {score:.3f} "
        f"({found}/{len(gt_tokens)} GT tokens in context)"
    )
    return round(score, 4)


def compute_context_precision(
    question: str,
    context_chunks: List[Document],
    ground_truth: str,
) -> float:
    """
    Measures what fraction of retrieved chunks are actually relevant.

    Approach:
      For each retrieved chunk, check if it contains tokens
      from the ground truth answer.
      Score = relevant_chunks / total_retrieved_chunks

    Args:
        question      : user's question
        context_chunks: chunks retrieved by pipeline
        ground_truth  : correct answer for reference

    Returns:
        Float 0.0-1.0
    """
    if not context_chunks or not ground_truth:
        return 0.0

    # Key tokens from ground truth
    STOP_WORDS = {
        "the",
        "is",
        "a",
        "an",
        "and",
        "or",
        "in",
        "with",
        "to",
        "of",
        "for",
        "as",
        "by",
        "from",
        "be",
        "was",
    }

    gt_tokens = set(
        t
        for t in re.findall(r"\b[a-z0-9]+\b", ground_truth.lower())
        if t not in STOP_WORDS and len(t) > 2
    )

    if not gt_tokens:
        return 0.5

    relevant_count = 0

    for chunk in context_chunks:
        chunk_text = chunk.page_content.lower()
        chunk_tokens = set(re.findall(r"\b[a-z0-9]+\b", chunk_text))

        # Chunk is "relevant" if it shares >20% tokens with ground truth
        overlap = gt_tokens & chunk_tokens
        if len(overlap) / len(gt_tokens) >= 0.20:
            relevant_count += 1

    score = relevant_count / len(context_chunks)
    logger.debug(
        f"Context precision: {score:.3f} "
        f"({relevant_count}/{len(context_chunks)} chunks relevant)"
    )
    return round(score, 4)


def compute_citation_coverage(answer: str) -> float:
    if not answer:
        return 0.0

    if "cannot find this information" in answer.lower():
        return 1.0

    # Remove citation tags temporarily to count clean sentences
    clean_answer = re.sub(r"\[SOURCE:[^\]]+\]", "", answer)

    # Split into sentences
    sentences = [
        s.strip()
        for s in re.split(r"[.!\n]+", clean_answer)
        if len(s.strip()) > 15  # lowered from 20 to 15
    ]

    if not sentences:
        return 1.0  # no sentences to check = fine

    # Citation tag pattern in ORIGINAL answer
    citation_pattern = re.compile(r"\[SOURCE:", re.IGNORECASE)

    # Count total citation tags in the full answer
    total_citations = len(citation_pattern.findall(answer))

    # Score = min(citations / sentences, 1.0)
    # If citations >= sentences → perfect score
    score = min(total_citations / len(sentences), 1.0) if sentences else 0.0

    logger.debug(
        f"Citation coverage: {score:.3f} "
        f"({total_citations} citations / {len(sentences)} sentences)"
    )
    return round(score, 4)


def compute_overall_score(
    faithfulness: float,
    answer_relevancy: float,
    context_recall: float,
    context_precision: float,
    citation_coverage: float,
) -> float:
    """
    Computes weighted overall score from all 5 metrics.

    Weights reflect relative importance for RAG quality:
      faithfulness      30% — most critical: no hallucinations
      answer_relevancy  25% — answer must address the question
      context_recall    20% — retriever must find relevant chunks
      context_precision 15% — retrieved chunks should be relevant
      citation_coverage 10% — citations for verifiability

    Args:
        All 5 metric scores as floats 0.0-1.0

    Returns:
        Weighted average float 0.0-1.0
    """
    weights = {
        "faithfulness": 0.30,
        "answer_relevancy": 0.25,
        "context_recall": 0.20,
        "context_precision": 0.15,
        "citation_coverage": 0.10,
    }

    overall = (
        faithfulness * weights["faithfulness"]
        + answer_relevancy * weights["answer_relevancy"]
        + context_recall * weights["context_recall"]
        + context_precision * weights["context_precision"]
        + citation_coverage * weights["citation_coverage"]
    )

    return round(overall, 4)
