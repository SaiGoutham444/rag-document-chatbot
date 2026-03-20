"""
run_evals.py — RAG Evaluation Runner
======================================
Runs the full evaluation suite against the ground truth dataset.

Usage:
  python tests/eval/run_evals.py              ← runs all evals
  python tests/eval/run_evals.py --fast       ← runs first 3 questions only

Exit codes:
  0 = quality gate PASSED (all metrics above threshold)
  1 = quality gate FAILED (one or more metrics below threshold)

Used by CI/CD pipeline to block PRs that degrade RAG quality.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from loguru import logger

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import EVAL_THRESHOLDS
from tests.eval.metrics import (
    EvalResult,
    EvalReport,
    compute_faithfulness,
    compute_answer_relevancy,
    compute_context_recall,
    compute_context_precision,
    compute_citation_coverage,
    compute_overall_score,
)


# ══════════════════════════════════════════════════════════════════
# DATASET LOADING
# ══════════════════════════════════════════════════════════════════


def load_eval_dataset(path: str = "tests/eval/eval_dataset.json") -> List[Dict]:
    """
    Loads the ground truth evaluation dataset from JSON.

    Args:
        path: path to eval_dataset.json

    Returns:
        List of dicts with question, ground_truth, source_document, etc.

    Raises:
        FileNotFoundError: if dataset file doesn't exist
        ValueError       : if dataset is empty or malformed
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found: '{path}'\n"
            f"Expected at: tests/eval/eval_dataset.json"
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        raise ValueError("Eval dataset is empty.")

    required_keys = ["question", "ground_truth", "source_document"]
    for i, item in enumerate(dataset):
        for key in required_keys:
            if key not in item:
                raise ValueError(f"Item {i} missing required key: '{key}'")

    logger.info(f"Loaded {len(dataset)} eval questions from '{path}'")
    return dataset


# ══════════════════════════════════════════════════════════════════
# SINGLE QUESTION EVALUATION
# ══════════════════════════════════════════════════════════════════


def run_single_eval(
    pipeline: RAGPipeline,
    eval_case: Dict,
) -> EvalResult:
    """
    Runs the full pipeline on one question and computes all metrics.

    Process:
      1. Call pipeline.query(question)
      2. Compute all 5 metrics against ground truth
      3. Return EvalResult with all scores

    Args:
        pipeline : initialized RAGPipeline with document loaded
        eval_case: one item from eval_dataset.json

    Returns:
        EvalResult with all metric scores
    """
    question = eval_case["question"]
    ground_truth = eval_case["ground_truth"]
    category = eval_case.get("category", "general")

    logger.info(f"Evaluating: '{question[:60]}'")

    result = EvalResult(
        question=question,
        ground_truth=ground_truth,
        generated_answer="",
        category=category,
    )

    try:
        # ── Run the pipeline ─────────────────────────────────────────
        start = time.time()
        response = pipeline.query(question)
        elapsed = time.time() - start

        result.generated_answer = response.answer

        # ── Compute metrics ──────────────────────────────────────────
        result.faithfulness = compute_faithfulness(
            answer=response.answer,
            context_chunks=response.source_chunks,
        )

        result.answer_relevancy = compute_answer_relevancy(
            question=question,
            answer=response.answer,
            embedding_model=pipeline.embedding_model,
        )

        result.context_recall = compute_context_recall(
            question=question,
            ground_truth=ground_truth,
            context_chunks=response.source_chunks,
        )

        result.context_precision = compute_context_precision(
            question=question,
            context_chunks=response.source_chunks,
            ground_truth=ground_truth,
        )

        result.citation_coverage = compute_citation_coverage(
            answer=response.answer,
        )

        result.overall_score = compute_overall_score(
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_recall=result.context_recall,
            context_precision=result.context_precision,
            citation_coverage=result.citation_coverage,
        )

        logger.info(
            f"  Overall: {result.overall_score:.3f} | "
            f"Faith: {result.faithfulness:.3f} | "
            f"Rel: {result.answer_relevancy:.3f} | "
            f"Recall: {result.context_recall:.3f} | "
            f"Prec: {result.context_precision:.3f} | "
            f"Cit: {result.citation_coverage:.3f} | "
            f"Time: {elapsed:.1f}s"
        )

    except Exception as e:
        result.error = str(e)
        logger.error(f"Eval failed for '{question[:50]}': {e}")

    return result


# ══════════════════════════════════════════════════════════════════
# FULL EVALUATION RUN
# ══════════════════════════════════════════════════════════════════


def run_full_eval(
    pipeline: RAGPipeline,
    dataset: List[Dict],
) -> EvalReport:
    """
    Runs evaluation on all questions in the dataset.

    Args:
        pipeline: initialized RAGPipeline with document loaded
        dataset : list of eval cases from load_eval_dataset()

    Returns:
        EvalReport with per-question results and averages
    """
    report = EvalReport()
    results = []

    logger.info(f"Starting full evaluation | {len(dataset)} questions")
    eval_start = time.time()

    for i, eval_case in enumerate(dataset, start=1):
        logger.info(f"Question {i}/{len(dataset)}")
        result = run_single_eval(pipeline, eval_case)
        results.append(result)

    # ── Aggregate results ────────────────────────────────────────
    valid_results = [r for r in results if r.error is None]

    report.results = results
    report.total_questions = len(results)

    if valid_results:
        report.avg_faithfulness = sum(r.faithfulness for r in valid_results) / len(
            valid_results
        )
        report.avg_answer_relevancy = sum(
            r.answer_relevancy for r in valid_results
        ) / len(valid_results)
        report.avg_context_recall = sum(r.context_recall for r in valid_results) / len(
            valid_results
        )
        report.avg_context_precision = sum(
            r.context_precision for r in valid_results
        ) / len(valid_results)
        report.avg_citation_coverage = sum(
            r.citation_coverage for r in valid_results
        ) / len(valid_results)
        report.avg_overall_score = sum(r.overall_score for r in valid_results) / len(
            valid_results
        )

    # ── Quality gate check ───────────────────────────────────────
    report.quality_gate_passed = check_quality_gate(report)

    # ── Count passed questions ───────────────────────────────────
    report.passed_questions = sum(1 for r in valid_results if r.overall_score >= 0.60)

    elapsed = time.time() - eval_start
    logger.info(
        f"Evaluation complete in {elapsed:.1f}s | "
        f"Quality gate: {'PASSED' if report.quality_gate_passed else 'FAILED'}"
    )

    return report


# ══════════════════════════════════════════════════════════════════
# QUALITY GATE
# ══════════════════════════════════════════════════════════════════


def check_quality_gate(report: EvalReport) -> bool:
    """
    Checks if all metric averages meet the minimum thresholds.
    Returns True if ALL metrics pass, False if ANY metric fails.

    Thresholds from config.py EVAL_THRESHOLDS:
      faithfulness      >= 0.70
      answer_relevancy  >= 0.70
      context_recall    >= 0.65
      context_precision >= 0.65
      citation_coverage >= 0.80

    Args:
        report: completed EvalReport

    Returns:
        True if quality gate passes, False if it fails
    """
    checks = {
        "faithfulness": report.avg_faithfulness,
        "answer_relevancy": report.avg_answer_relevancy,
        "context_recall": report.avg_context_recall,
        "context_precision": report.avg_context_precision,
        "citation_coverage": report.avg_citation_coverage,
    }

    all_passed = True

    for metric, score in checks.items():
        threshold = EVAL_THRESHOLDS.get(metric, 0.70)
        passed = score >= threshold
        status = "✅ PASS" if passed else "❌ FAIL"

        logger.info(
            f"  {status} | {metric:20s}: " f"{score:.3f} (threshold: {threshold:.2f})"
        )

        if not passed:
            all_passed = False

    return all_passed


# ══════════════════════════════════════════════════════════════════
# REPORT SAVING
# ══════════════════════════════════════════════════════════════════


def save_eval_report(report: EvalReport, path: str = "eval_report.json"):
    """
    Saves the evaluation report to a JSON file.
    Used by CI/CD to store results as build artifacts.

    Args:
        report: completed EvalReport
        path  : output file path
    """
    report_dict = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": report.total_questions,
        "passed_questions": report.passed_questions,
        "quality_gate_passed": report.quality_gate_passed,
        "averages": {
            "faithfulness": round(report.avg_faithfulness, 4),
            "answer_relevancy": round(report.avg_answer_relevancy, 4),
            "context_recall": round(report.avg_context_recall, 4),
            "context_precision": round(report.avg_context_precision, 4),
            "citation_coverage": round(report.avg_citation_coverage, 4),
            "overall_score": round(report.avg_overall_score, 4),
        },
        "thresholds": EVAL_THRESHOLDS,
        "per_question": [
            {
                "question": r.question,
                "category": r.category,
                "faithfulness": r.faithfulness,
                "answer_relevancy": r.answer_relevancy,
                "context_recall": r.context_recall,
                "context_precision": r.context_precision,
                "citation_coverage": r.citation_coverage,
                "overall_score": r.overall_score,
                "error": r.error,
            }
            for r in report.results
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Eval report saved to '{path}'")


# ══════════════════════════════════════════════════════════════════
# PRETTY PRINT
# ══════════════════════════════════════════════════════════════════


def print_eval_summary(report: EvalReport):
    """
    Prints a formatted evaluation summary table to the terminal.

    Args:
        report: completed EvalReport
    """
    print("\n")
    print("=" * 65)
    print("  RAG EVALUATION REPORT")
    print("=" * 65)

    # Per-question results
    print(f"\n{'Question':<45} {'Overall':>8} {'Faith':>7} {'Rel':>6}")
    print("-" * 65)

    for r in report.results:
        status = "✅" if r.error is None else "❌"
        overall = f"{r.overall_score:.3f}" if r.error is None else "ERROR"
        faith = f"{r.faithfulness:.3f}" if r.error is None else "  -  "
        rel = f"{r.answer_relevancy:.3f}" if r.error is None else "  -  "
        q_short = r.question[:43] + ".." if len(r.question) > 43 else r.question

        print(f"{status} {q_short:<44} {overall:>7} {faith:>7} {rel:>6}")

    # Averages
    print("=" * 65)
    print(f"\n  AVERAGES ACROSS {report.total_questions} QUESTIONS:")
    print(f"  {'Faithfulness':<22}: {report.avg_faithfulness:.4f}")
    print(f"  {'Answer Relevancy':<22}: {report.avg_answer_relevancy:.4f}")
    print(f"  {'Context Recall':<22}: {report.avg_context_recall:.4f}")
    print(f"  {'Context Precision':<22}: {report.avg_context_precision:.4f}")
    print(f"  {'Citation Coverage':<22}: {report.avg_citation_coverage:.4f}")
    print(f"  {'Overall Score':<22}: {report.avg_overall_score:.4f}")

    # Quality gate result
    print("\n" + "=" * 65)
    if report.quality_gate_passed:
        print("  ✅ QUALITY GATE: PASSED — PR can be merged")
    else:
        print("  ❌ QUALITY GATE: FAILED — PR is BLOCKED")
        print("     Fix the failing metrics before merging.")
    print("=" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════


def main():
    """
    Main evaluation runner.
    Called by CI/CD and by developers running 'make eval'.
    """
    # ── Parse arguments ──────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only first 3 questions (faster, for development)",
    )
    parser.add_argument(
        "--doc",
        type=str,
        default="data/sample_docs/resume.pdf",
        help="Path to document to evaluate against",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_report.json",
        help="Path to save the eval report JSON",
    )
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("RAG EVALUATION PIPELINE")
    logger.info("=" * 65)

    # ── Load dataset ─────────────────────────────────────────────
    dataset = load_eval_dataset()

    if args.fast:
        dataset = dataset[:3]
        logger.info(f"Fast mode: using first {len(dataset)} questions")

    # ── Initialize pipeline ──────────────────────────────────────
    logger.info("Initializing pipeline...")
    pipeline = RAGPipeline()

    # ── Process document ─────────────────────────────────────────
    logger.info(f"Processing document: '{args.doc}'")
    pipeline.process_document(args.doc)

    # ── Run evaluation ───────────────────────────────────────────
    report = run_full_eval(pipeline, dataset)

    # ── Print summary ────────────────────────────────────────────
    print_eval_summary(report)

    # ── Save report ──────────────────────────────────────────────
    save_eval_report(report, args.output)

    # ── Exit with correct code for CI ───────────────────────────
    # exit(0) = success → CI pipeline continues
    # exit(1) = failure → CI pipeline BLOCKS the PR
    if report.quality_gate_passed:
        logger.info("Quality gate PASSED — exiting with code 0")
        sys.exit(0)
    else:
        logger.error("Quality gate FAILED — exiting with code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
