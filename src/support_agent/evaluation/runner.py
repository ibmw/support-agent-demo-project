"""
Batch evaluation runner for the support agent.

Provides functionality to evaluate the agent against a set of test queries
and export results in CSV or JSON format.
"""

import csv
import json
import random
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from ..agent.crew import SupportCrew
from ..config import QUESTIONS_CSV
from ..exceptions import AgentError
from ..logging import get_logger
from .metrics import calculate_summary, format_summary_report
from .schemas import BatchConfig, EvalSummary, EvaluationResult

logger = get_logger(__name__, component="evaluation")


def load_questions(csv_path: Path | None = None) -> list[str]:
    """
    Load evaluation questions from CSV file.

    Args:
        csv_path: Path to questions CSV (defaults to data/questions.csv)

    Returns:
        List of question strings
    """
    path = csv_path or QUESTIONS_CSV

    questions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both 'customer' and 'question' column names
            question = row.get("customer") or row.get("question", "")
            if question.strip():
                questions.append(question.strip())

    logger.info("Loaded questions", count=len(questions), path=str(path))
    return questions


def sample_questions(
    questions: list[str],
    sample_size: int | None = None,
    random_seed: int | None = None,
) -> list[str]:
    """
    Sample questions for evaluation.

    Args:
        questions: Full list of questions
        sample_size: Number to sample (None = all)
        random_seed: Seed for reproducibility

    Returns:
        Sampled (or full) list of questions
    """
    if sample_size is None or sample_size >= len(questions):
        return questions

    if random_seed is not None:
        random.seed(random_seed)

    sampled = random.sample(questions, sample_size)
    logger.info(
        "Sampled questions",
        sample_size=len(sampled),
        total=len(questions),
        seed=random_seed,
    )
    return sampled


def evaluate_single(
    question: str,
    crew: SupportCrew,
) -> EvaluationResult:
    """
    Evaluate a single question.

    Args:
        question: The question to evaluate
        crew: The SupportCrew instance to use

    Returns:
        EvaluationResult with response or error
    """
    start_time = time.time()

    try:
        response = crew.process_query(question)
        latency_ms = (time.time() - start_time) * 1000

        return EvaluationResult(
            question=question,
            response=response,
            latency_ms=latency_ms,
            retrieved_chunks=len(response.sources),
        )

    except AgentError as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.warning(
            "Evaluation failed",
            question_preview=question[:50],
            error=str(e),
        )
        return EvaluationResult(
            question=question,
            response=None,
            latency_ms=latency_ms,
            error=str(e),
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            "Unexpected evaluation error",
            question_preview=question[:50],
            error=str(e),
            error_type=type(e).__name__,
        )
        return EvaluationResult(
            question=question,
            response=None,
            latency_ms=latency_ms,
            error=f"{type(e).__name__}: {e}",
        )


def run_batch_evaluation(
    questions: list[str],
    crew: SupportCrew | None = None,
    config: BatchConfig | None = None,
    progress_callback: Callable[[int, int, EvaluationResult], None] | None = None,
) -> tuple[list[EvaluationResult], EvalSummary]:
    """
    Run batch evaluation on a list of questions.

    Args:
        questions: List of questions to evaluate
        crew: SupportCrew instance (creates one if not provided)
        config: Batch configuration
        progress_callback: Optional callback for progress updates
            Called with (current_index, total, result) after each evaluation

    Returns:
        Tuple of (results list, summary)
    """
    config = config or BatchConfig()

    # Sample if needed
    eval_questions = sample_questions(
        questions,
        sample_size=config.sample_size,
        random_seed=config.random_seed,
    )

    # Create crew if not provided
    if crew is None:
        crew = SupportCrew()

    start_time = datetime.now(UTC)
    results: list[EvaluationResult] = []
    total = len(eval_questions)

    logger.info(
        "Starting batch evaluation",
        total_questions=total,
        eval_tag=config.eval_tag,
    )

    for i, question in enumerate(eval_questions):
        result = evaluate_single(question, crew)
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total, result)

        # Log progress every 10%
        if (i + 1) % max(1, total // 10) == 0 or i + 1 == total:
            successful = sum(1 for r in results if r.success)
            logger.info(
                "Evaluation progress",
                completed=i + 1,
                total=total,
                successful=successful,
                failed=i + 1 - successful,
            )

    end_time = datetime.now(UTC)
    summary = calculate_summary(results, start_time, end_time, config.eval_tag)

    logger.info(
        "Batch evaluation complete",
        total=summary.total_queries,
        successful=summary.successful,
        failed=summary.failed,
        duration_seconds=round(summary.duration_seconds, 2),
    )

    return results, summary


def iter_batch_evaluation(
    questions: list[str],
    crew: SupportCrew | None = None,
    config: BatchConfig | None = None,
) -> Iterator[tuple[int, int, EvaluationResult]]:
    """
    Iterate through batch evaluation, yielding results as they complete.

    Useful for streaming progress updates.

    Args:
        questions: List of questions to evaluate
        crew: SupportCrew instance (creates one if not provided)
        config: Batch configuration

    Yields:
        Tuple of (current_index, total, result) for each evaluation
    """
    config = config or BatchConfig()

    eval_questions = sample_questions(
        questions,
        sample_size=config.sample_size,
        random_seed=config.random_seed,
    )

    if crew is None:
        crew = SupportCrew()

    total = len(eval_questions)

    for i, question in enumerate(eval_questions):
        result = evaluate_single(question, crew)
        yield i + 1, total, result


def export_results_csv(
    results: list[EvaluationResult],
    output_path: Path,
) -> None:
    """
    Export evaluation results to CSV.

    Args:
        results: List of evaluation results
        output_path: Path for output CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "question",
                "action",
                "message",
                "sources",
                "confidence",
                "latency_ms",
                "retrieved_chunks",
                "success",
                "error",
                "timestamp",
            ]
        )

        # Data rows
        for result in results:
            response = result.response
            writer.writerow(
                [
                    result.question,
                    response.action if response else "",
                    response.message if response else "",
                    "|".join(response.sources) if response else "",
                    response.confidence if response else "",
                    round(result.latency_ms, 2),
                    result.retrieved_chunks,
                    result.success,
                    result.error or "",
                    result.timestamp.isoformat(),
                ]
            )

    logger.info("Exported results to CSV", path=str(output_path), count=len(results))


def export_results_json(
    results: list[EvaluationResult],
    output_path: Path,
    summary: EvalSummary | None = None,
) -> None:
    """
    Export evaluation results to JSON.

    Args:
        results: List of evaluation results
        output_path: Path for output JSON file
        summary: Optional summary to include
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "results": [r.model_dump(mode="json") for r in results],
    }

    if summary:
        data["summary"] = summary.model_dump(mode="json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Exported results to JSON", path=str(output_path), count=len(results))


def export_results(
    results: list[EvaluationResult],
    output_path: Path,
    format: str = "csv",
    summary: EvalSummary | None = None,
) -> None:
    """
    Export evaluation results to file.

    Args:
        results: List of evaluation results
        output_path: Path for output file
        format: Output format ('csv' or 'json')
        summary: Optional summary (included in JSON only)
    """
    if format == "json":
        export_results_json(results, output_path, summary)
    else:
        export_results_csv(results, output_path)


class EvaluationRunner:
    """
    High-level evaluation runner with configuration.

    Example:
        runner = EvaluationRunner(sample_size=100, eval_tag="experiment-1")
        results, summary = runner.run()
        runner.export(results, summary, Path("results/eval.csv"))
        print(runner.format_report(summary))
    """

    def __init__(
        self,
        questions_path: Path | None = None,
        sample_size: int | None = None,
        random_seed: int | None = None,
        eval_tag: str | None = None,
        output_format: str = "csv",
    ):
        """
        Initialize the evaluation runner.

        Args:
            questions_path: Path to questions CSV
            sample_size: Number of questions to sample (None = all)
            random_seed: Random seed for reproducibility
            eval_tag: Tag for tracking this evaluation run
            output_format: Default output format ('csv' or 'json')
        """
        self.questions_path = questions_path
        self.config = BatchConfig(
            sample_size=sample_size,
            random_seed=random_seed,
            eval_tag=eval_tag,
            output_format=output_format,
        )
        self._questions: list[str] | None = None
        self._crew: SupportCrew | None = None

    @property
    def questions(self) -> list[str]:
        """Load questions lazily."""
        if self._questions is None:
            self._questions = load_questions(self.questions_path)
        return self._questions

    @property
    def crew(self) -> SupportCrew:
        """Create crew lazily."""
        if self._crew is None:
            self._crew = SupportCrew()
        return self._crew

    def run(
        self,
        progress_callback: Callable[[int, int, EvaluationResult], None] | None = None,
    ) -> tuple[list[EvaluationResult], EvalSummary]:
        """
        Run the evaluation.

        Args:
            progress_callback: Optional progress callback

        Returns:
            Tuple of (results, summary)
        """
        return run_batch_evaluation(
            questions=self.questions,
            crew=self.crew,
            config=self.config,
            progress_callback=progress_callback,
        )

    def export(
        self,
        results: list[EvaluationResult],
        summary: EvalSummary,
        output_path: Path,
    ) -> None:
        """Export results to file."""
        export_results(
            results=results,
            output_path=output_path,
            format=self.config.output_format,
            summary=summary,
        )

    def format_report(self, summary: EvalSummary) -> str:
        """Format summary as human-readable report."""
        return format_summary_report(summary)
