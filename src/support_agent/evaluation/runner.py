"""
Batch evaluation runner for the support agent.

Provides functionality to evaluate the agent against a set of test queries
and export results in CSV or JSON format. Supports both single-turn and
multi-turn conversation evaluations.
"""

import csv
import json
import random
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from ..agent.crew import SupportCrew
from ..config import QUESTIONS_CSV
from ..exceptions import AgentError
from ..logging import get_logger
from ..memory import reset_session_manager
from .metrics import (
    calculate_conversation_summary,
    calculate_summary,
    format_conversation_summary_report,
    format_summary_report,
)
from .schemas import (
    BatchConfig,
    ConversationBatchConfig,
    ConversationEvalSummary,
    ConversationResult,
    ConversationScenario,
    EvalSummary,
    EvaluationResult,
    TurnResult,
)

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


# =============================================================================
# Conversation Evaluation
# =============================================================================


def load_scenarios(json_path: Path) -> list[ConversationScenario]:
    """
    Load conversation scenarios from a JSON file.

    Args:
        json_path: Path to scenarios JSON file

    Returns:
        List of ConversationScenario objects
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list format and {"scenarios": [...]} format
    scenarios_data = data if isinstance(data, list) else data.get("scenarios", [])

    scenarios = [ConversationScenario(**s) for s in scenarios_data]
    logger.info("Loaded conversation scenarios", count=len(scenarios), path=str(json_path))
    return scenarios


def filter_scenarios(
    scenarios: list[ConversationScenario],
    scenario_ids: list[str] | None = None,
    categories: list[str] | None = None,
) -> list[ConversationScenario]:
    """
    Filter scenarios by IDs or categories.

    Args:
        scenarios: Full list of scenarios
        scenario_ids: Optional list of specific IDs to include
        categories: Optional list of categories to include

    Returns:
        Filtered list of scenarios
    """
    filtered = scenarios

    if scenario_ids:
        id_set = set(scenario_ids)
        filtered = [s for s in filtered if s.id in id_set]

    if categories:
        cat_set = set(categories)
        filtered = [s for s in filtered if s.category in cat_set]

    if scenario_ids or categories:
        logger.info(
            "Filtered scenarios",
            original_count=len(scenarios),
            filtered_count=len(filtered),
            scenario_ids=scenario_ids,
            categories=categories,
        )

    return filtered


def evaluate_conversation(
    scenario: ConversationScenario,
    crew: SupportCrew,
) -> ConversationResult:
    """
    Evaluate a single conversation scenario.

    Args:
        scenario: The conversation scenario to evaluate
        crew: The SupportCrew instance to use

    Returns:
        ConversationResult with turn-by-turn results
    """
    start_time = time.time()
    turns: list[TurnResult] = []
    session_id: UUID | None = None
    total_error: str | None = None

    log = logger.bind(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        expected_turns=len(scenario.user_messages),
    )
    log.info("Evaluating conversation scenario")

    try:
        for i, user_message in enumerate(scenario.user_messages):
            turn_start = time.time()

            try:
                response, session_id = crew.process_conversation(
                    query=user_message,
                    session_id=session_id,
                )
                turn_latency_ms = (time.time() - turn_start) * 1000

                turn = TurnResult(
                    turn_index=i,
                    user_message=user_message,
                    response=response,
                    latency_ms=turn_latency_ms,
                )
                turns.append(turn)

                log.debug(
                    "Turn completed",
                    turn_index=i,
                    action=response.action,
                    latency_ms=round(turn_latency_ms, 2),
                )

                # Check if we should stop early (CLOSE or HANDOVER typically end conversation)
                if response.action in ("CLOSE", "HANDOVER") and i < len(scenario.user_messages) - 1:
                    log.info(
                        "Conversation ended early",
                        turn_index=i,
                        action=response.action,
                        remaining_turns=len(scenario.user_messages) - i - 1,
                    )
                    break

            except (AgentError, Exception) as e:
                turn_latency_ms = (time.time() - turn_start) * 1000
                error_msg = f"{type(e).__name__}: {e}"

                turn = TurnResult(
                    turn_index=i,
                    user_message=user_message,
                    response=None,
                    latency_ms=turn_latency_ms,
                    error=error_msg,
                )
                turns.append(turn)

                log.warning(
                    "Turn failed",
                    turn_index=i,
                    error=error_msg,
                )
                # Stop on error
                break

    except Exception as e:
        total_error = f"{type(e).__name__}: {e}"
        log.error("Conversation evaluation failed", error=total_error)

    total_latency_ms = (time.time() - start_time) * 1000

    result = ConversationResult(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        turns=turns,
        total_latency_ms=total_latency_ms,
        error=total_error,
    )

    log.info(
        "Conversation evaluation complete",
        success=result.success,
        final_action=result.final_action,
        turn_count=result.turn_count,
        total_latency_ms=round(total_latency_ms, 2),
    )

    return result


def run_conversation_evaluation(
    scenarios: list[ConversationScenario],
    crew: SupportCrew | None = None,
    config: ConversationBatchConfig | None = None,
    progress_callback: Callable[[int, int, ConversationResult], None] | None = None,
    reset_memory_per_scenario: bool = True,
) -> tuple[list[ConversationResult], ConversationEvalSummary]:
    """
    Run batch evaluation on conversation scenarios.

    Args:
        scenarios: List of scenarios to evaluate
        crew: SupportCrew instance (creates one if not provided)
        config: Batch configuration
        progress_callback: Optional callback for progress updates
        reset_memory_per_scenario: Whether to reset session memory between scenarios

    Returns:
        Tuple of (results list, summary)
    """
    config = config or ConversationBatchConfig()

    # Filter scenarios if needed
    eval_scenarios = filter_scenarios(
        scenarios,
        scenario_ids=config.scenario_ids,
        categories=config.categories,
    )

    if crew is None:
        crew = SupportCrew()

    start_time = datetime.now(UTC)
    results: list[ConversationResult] = []
    total = len(eval_scenarios)

    logger.info(
        "Starting conversation batch evaluation",
        total_scenarios=total,
        eval_tag=config.eval_tag,
    )

    for i, scenario in enumerate(eval_scenarios):
        # Reset session memory to ensure clean state
        if reset_memory_per_scenario:
            reset_session_manager()

        result = evaluate_conversation(scenario, crew)
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total, result)

        # Log progress every 10%
        if (i + 1) % max(1, total // 10) == 0 or i + 1 == total:
            successful = sum(1 for r in results if r.success)
            logger.info(
                "Conversation evaluation progress",
                completed=i + 1,
                total=total,
                successful=successful,
                failed=i + 1 - successful,
            )

    end_time = datetime.now(UTC)
    summary = calculate_conversation_summary(
        results, scenarios, start_time, end_time, config.eval_tag
    )

    logger.info(
        "Conversation batch evaluation complete",
        total=summary.total_scenarios,
        successful=summary.successful,
        failed=summary.failed,
        duration_seconds=round(summary.duration_seconds, 2),
        resolution_rate=f"{summary.metrics.resolution_rate:.1%}",
    )

    return results, summary


def iter_conversation_evaluation(
    scenarios: list[ConversationScenario],
    crew: SupportCrew | None = None,
    config: ConversationBatchConfig | None = None,
    reset_memory_per_scenario: bool = True,
) -> Iterator[tuple[int, int, ConversationResult]]:
    """
    Iterate through conversation evaluation, yielding results as they complete.

    Args:
        scenarios: List of scenarios to evaluate
        crew: SupportCrew instance (creates one if not provided)
        config: Batch configuration
        reset_memory_per_scenario: Whether to reset session memory between scenarios

    Yields:
        Tuple of (current_index, total, result) for each evaluation
    """
    config = config or ConversationBatchConfig()

    eval_scenarios = filter_scenarios(
        scenarios,
        scenario_ids=config.scenario_ids,
        categories=config.categories,
    )

    if crew is None:
        crew = SupportCrew()

    total = len(eval_scenarios)

    for i, scenario in enumerate(eval_scenarios):
        if reset_memory_per_scenario:
            reset_session_manager()

        result = evaluate_conversation(scenario, crew)
        yield i + 1, total, result


def export_conversation_results_json(
    results: list[ConversationResult],
    output_path: Path,
    summary: ConversationEvalSummary | None = None,
    scenarios: list[ConversationScenario] | None = None,
) -> None:
    """
    Export conversation evaluation results to JSON.

    Args:
        results: List of conversation results
        output_path: Path for output JSON file
        summary: Optional summary to include
        scenarios: Optional scenarios for golden comparison data
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {
        "results": [r.model_dump(mode="json") for r in results],
    }

    if summary:
        data["summary"] = summary.model_dump(mode="json")

    # Add golden comparisons if scenarios provided
    if scenarios:
        from .metrics import compare_to_golden

        scenario_map = {s.id: s for s in scenarios}
        comparisons = []
        for r in results:
            scenario = scenario_map.get(r.scenario_id)
            if scenario and scenario.expected_final_action:
                comparison = compare_to_golden(r, scenario)
                comparisons.append(comparison.model_dump(mode="json"))
        if comparisons:
            data["golden_comparisons"] = comparisons

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(
        "Exported conversation results to JSON",
        path=str(output_path),
        count=len(results),
    )


def export_conversation_results_csv(
    results: list[ConversationResult],
    output_path: Path,
) -> None:
    """
    Export conversation evaluation results to CSV.

    Creates one row per turn for detailed analysis.

    Args:
        results: List of conversation results
        output_path: Path for output CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "scenario_id",
            "scenario_name",
            "turn_index",
            "user_message",
            "action",
            "message",
            "sources",
            "latency_ms",
            "success",
            "error",
            "final_action",
            "total_turns",
            "total_latency_ms",
            "timestamp",
        ])

        # Data rows (one per turn)
        for result in results:
            for turn in result.turns:
                response = turn.response
                writer.writerow([
                    result.scenario_id,
                    result.scenario_name,
                    turn.turn_index,
                    turn.user_message,
                    response.action if response else "",
                    response.message if response else "",
                    "|".join(response.sources) if response else "",
                    round(turn.latency_ms, 2),
                    turn.success,
                    turn.error or "",
                    result.final_action or "",
                    result.turn_count,
                    round(result.total_latency_ms, 2),
                    result.timestamp.isoformat(),
                ])

    logger.info(
        "Exported conversation results to CSV",
        path=str(output_path),
        scenarios=len(results),
        total_turns=sum(r.turn_count for r in results),
    )


def export_conversation_results(
    results: list[ConversationResult],
    output_path: Path,
    format: str = "json",
    summary: ConversationEvalSummary | None = None,
    scenarios: list[ConversationScenario] | None = None,
) -> None:
    """
    Export conversation evaluation results to file.

    Args:
        results: List of conversation results
        output_path: Path for output file
        format: Output format ('csv' or 'json')
        summary: Optional summary (included in JSON only)
        scenarios: Optional scenarios for golden comparison (JSON only)
    """
    if format == "json":
        export_conversation_results_json(results, output_path, summary, scenarios)
    else:
        export_conversation_results_csv(results, output_path)


class ConversationEvaluationRunner:
    """
    High-level conversation evaluation runner with configuration.

    Example:
        runner = ConversationEvaluationRunner(
            scenarios_path=Path("data/conversations/scenarios.json"),
            eval_tag="experiment-1",
        )
        results, summary = runner.run()
        runner.export(results, summary, Path("results/conv_eval.json"))
        print(runner.format_report(summary))
    """

    def __init__(
        self,
        scenarios_path: Path,
        scenario_ids: list[str] | None = None,
        categories: list[str] | None = None,
        eval_tag: str | None = None,
        output_format: str = "json",
        reset_memory_per_scenario: bool = True,
    ):
        """
        Initialize the conversation evaluation runner.

        Args:
            scenarios_path: Path to scenarios JSON file
            scenario_ids: Specific scenario IDs to evaluate (None = all)
            categories: Filter by categories (None = all)
            eval_tag: Tag for tracking this evaluation run
            output_format: Default output format ('csv' or 'json')
            reset_memory_per_scenario: Reset session memory between scenarios
        """
        self.scenarios_path = scenarios_path
        self.reset_memory_per_scenario = reset_memory_per_scenario
        self.config = ConversationBatchConfig(
            scenario_ids=scenario_ids,
            categories=categories,
            eval_tag=eval_tag,
            output_format=output_format,
        )
        self._scenarios: list[ConversationScenario] | None = None
        self._crew: SupportCrew | None = None

    @property
    def scenarios(self) -> list[ConversationScenario]:
        """Load scenarios lazily."""
        if self._scenarios is None:
            self._scenarios = load_scenarios(self.scenarios_path)
        return self._scenarios

    @property
    def crew(self) -> SupportCrew:
        """Create crew lazily."""
        if self._crew is None:
            self._crew = SupportCrew()
        return self._crew

    def run(
        self,
        progress_callback: Callable[[int, int, ConversationResult], None] | None = None,
    ) -> tuple[list[ConversationResult], ConversationEvalSummary]:
        """
        Run the conversation evaluation.

        Args:
            progress_callback: Optional progress callback

        Returns:
            Tuple of (results, summary)
        """
        return run_conversation_evaluation(
            scenarios=self.scenarios,
            crew=self.crew,
            config=self.config,
            progress_callback=progress_callback,
            reset_memory_per_scenario=self.reset_memory_per_scenario,
        )

    def export(
        self,
        results: list[ConversationResult],
        summary: ConversationEvalSummary,
        output_path: Path,
    ) -> None:
        """Export results to file."""
        export_conversation_results(
            results=results,
            output_path=output_path,
            format=self.config.output_format,
            summary=summary,
            scenarios=self.scenarios,
        )

    def format_report(self, summary: ConversationEvalSummary) -> str:
        """Format summary as human-readable report."""
        return format_conversation_summary_report(summary)
