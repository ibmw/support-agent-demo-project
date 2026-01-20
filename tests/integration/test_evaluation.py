"""Integration tests for evaluation runner."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from support_agent.agent.crew import SupportCrew, reset_crew
from support_agent.agent.schemas import AgentResponse
from support_agent.evaluation import (
    BatchConfig,
    EvaluationResult,
    EvaluationRunner,
    evaluate_single,
    export_results,
    export_results_csv,
    export_results_json,
    iter_batch_evaluation,
    load_questions,
    run_batch_evaluation,
    sample_questions,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset crew singleton before each test."""
    reset_crew()
    yield
    reset_crew()


@pytest.fixture
def mock_crew_result():
    """Factory fixture for creating mock crew results."""

    def _create_result(
        action: str = "CLOSE",
        message: str = "Here's the answer.",
        sources: list[str] | None = None,
        confidence: float = 0.9,
    ) -> MagicMock:
        response = AgentResponse(
            action=action,
            message=message,
            sources=sources or ["Source"],
            confidence=confidence,
        )
        mock_result = MagicMock()
        mock_result.pydantic = response
        mock_result.tasks_output = []
        mock_result.token_usage = None
        return mock_result

    return _create_result


@pytest.fixture
def mock_crewai(mock_crew_result):
    """Mock CrewAI components."""
    with (
        patch("support_agent.agent.crew.Agent") as mock_agent,
        patch("support_agent.agent.crew.Crew") as mock_crew_cls,
        patch("support_agent.agent.crew.Task") as mock_task,
    ):
        mock_crew_cls.return_value.kickoff.return_value = mock_crew_result()
        yield {
            "agent": mock_agent,
            "crew_cls": mock_crew_cls,
            "task": mock_task,
            "set_result": lambda r: setattr(
                mock_crew_cls.return_value.kickoff, "return_value", r
            ),
        }


@pytest.fixture
def sample_questions_csv(tmp_path) -> Path:
    """Create a temporary questions CSV file."""
    csv_path = tmp_path / "questions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "customer"])
        writer.writerow(["0", "How do I reset my password?"])
        writer.writerow(["1", "What's the pricing?"])
        writer.writerow(["2", "I need a refund"])
        writer.writerow(["3", "How do integrations work?"])
        writer.writerow(["4", "Contact support"])
    return csv_path


@pytest.fixture
def sample_results() -> list[EvaluationResult]:
    """Sample evaluation results for export tests."""
    return [
        EvaluationResult(
            question="Question 1",
            response=AgentResponse(
                action="CLOSE",
                message="Answer 1",
                sources=["Source A", "Source B"],
                confidence=0.9,
            ),
            latency_ms=150.0,
            retrieved_chunks=2,
        ),
        EvaluationResult(
            question="Question 2",
            response=None,
            latency_ms=50.0,
            error="API Error",
        ),
    ]


# ============================================================================
# load_questions Tests
# ============================================================================


class TestLoadQuestions:
    """Tests for load_questions function."""

    def test_loads_questions_from_csv(self, sample_questions_csv):
        """Loads questions from CSV file."""
        questions = load_questions(sample_questions_csv)

        assert len(questions) == 5
        assert "How do I reset my password?" in questions
        assert "What's the pricing?" in questions

    def test_strips_whitespace(self, tmp_path):
        """Strips whitespace from questions."""
        csv_path = tmp_path / "questions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["", "customer"])
            writer.writerow(["0", "  Question with spaces  "])
        questions = load_questions(csv_path)

        assert questions[0] == "Question with spaces"

    def test_skips_empty_questions(self, tmp_path):
        """Skips empty or whitespace-only questions."""
        csv_path = tmp_path / "questions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["", "customer"])
            writer.writerow(["0", "Valid question"])
            writer.writerow(["1", ""])
            writer.writerow(["2", "   "])
            writer.writerow(["3", "Another valid question"])

        questions = load_questions(csv_path)
        assert len(questions) == 2


# ============================================================================
# sample_questions Tests
# ============================================================================


class TestSampleQuestions:
    """Tests for sample_questions function."""

    def test_returns_all_when_no_sample_size(self):
        """Returns all questions when sample_size is None."""
        questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        sampled = sample_questions(questions)

        assert sampled == questions

    def test_returns_all_when_sample_exceeds_total(self):
        """Returns all questions when sample_size > total."""
        questions = ["Q1", "Q2", "Q3"]
        sampled = sample_questions(questions, sample_size=10)

        assert sampled == questions

    def test_samples_correct_number(self):
        """Returns correct number of samples."""
        questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        sampled = sample_questions(questions, sample_size=3)

        assert len(sampled) == 3
        assert all(q in questions for q in sampled)

    def test_reproducible_with_seed(self):
        """Same seed produces same sample."""
        questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]

        sample1 = sample_questions(questions, sample_size=3, random_seed=42)
        sample2 = sample_questions(questions, sample_size=3, random_seed=42)

        assert sample1 == sample2

    def test_different_seeds_different_samples(self):
        """Different seeds produce different samples (usually)."""
        questions = list(range(100))  # Large enough to almost always differ

        sample1 = sample_questions(questions, sample_size=10, random_seed=42)
        sample2 = sample_questions(questions, sample_size=10, random_seed=123)

        # Very unlikely to be equal with different seeds
        assert sample1 != sample2


# ============================================================================
# evaluate_single Tests
# ============================================================================


class TestEvaluateSingle:
    """Tests for evaluate_single function."""

    def test_successful_evaluation(self, mock_crewai, mock_crew_result):
        """Returns successful result with response."""
        crew = SupportCrew()
        result = evaluate_single("How do I reset my password?", crew)

        assert result.success is True
        assert result.response is not None
        assert result.response.action == "CLOSE"
        assert result.latency_ms > 0

    def test_captures_error(self, mock_crewai):
        """Captures error when evaluation fails."""
        mock_crewai["crew_cls"].return_value.kickoff.side_effect = Exception(
            "LLM failed"
        )

        crew = SupportCrew()
        result = evaluate_single("Test question", crew)

        assert result.success is False
        assert result.error is not None
        assert "LLM failed" in result.error

    def test_includes_question(self, mock_crewai):
        """Result includes the original question."""
        crew = SupportCrew()
        question = "What's the pricing?"
        result = evaluate_single(question, crew)

        assert result.question == question


# ============================================================================
# run_batch_evaluation Tests
# ============================================================================


class TestRunBatchEvaluation:
    """Tests for run_batch_evaluation function."""

    def test_evaluates_all_questions(self, mock_crewai):
        """Evaluates all provided questions."""
        questions = ["Q1", "Q2", "Q3"]
        results, summary = run_batch_evaluation(questions)

        assert len(results) == 3
        assert summary.total_queries == 3

    def test_respects_sample_size(self, mock_crewai):
        """Respects sample_size configuration."""
        questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        config = BatchConfig(sample_size=2, random_seed=42)

        results, summary = run_batch_evaluation(questions, config=config)

        assert len(results) == 2
        assert summary.total_queries == 2

    def test_calls_progress_callback(self, mock_crewai):
        """Calls progress callback for each evaluation."""
        questions = ["Q1", "Q2", "Q3"]
        progress_calls = []

        def callback(current, total, result):
            progress_calls.append((current, total, result.question))

        run_batch_evaluation(questions, progress_callback=callback)

        assert len(progress_calls) == 3
        assert progress_calls[0][0] == 1
        assert progress_calls[2][0] == 3

    def test_calculates_summary(self, mock_crewai):
        """Calculates correct summary statistics."""
        questions = ["Q1", "Q2"]
        results, summary = run_batch_evaluation(questions)

        assert summary.successful == 2
        assert summary.failed == 0
        assert summary.action_distribution.close == 2
        assert summary.duration_seconds > 0

    def test_includes_eval_tag(self, mock_crewai):
        """Includes eval_tag in summary."""
        config = BatchConfig(eval_tag="test-experiment")
        results, summary = run_batch_evaluation(["Q1"], config=config)

        assert summary.eval_tag == "test-experiment"


# ============================================================================
# iter_batch_evaluation Tests
# ============================================================================


class TestIterBatchEvaluation:
    """Tests for iter_batch_evaluation function."""

    def test_yields_results(self, mock_crewai):
        """Yields results as they complete."""
        questions = ["Q1", "Q2", "Q3"]
        results = list(iter_batch_evaluation(questions))

        assert len(results) == 3
        # Each result is (current, total, EvaluationResult)
        assert results[0][0] == 1
        assert results[0][1] == 3
        assert results[2][0] == 3

    def test_respects_config(self, mock_crewai):
        """Respects batch configuration."""
        questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        config = BatchConfig(sample_size=2, random_seed=42)

        results = list(iter_batch_evaluation(questions, config=config))

        assert len(results) == 2


# ============================================================================
# Export Tests
# ============================================================================


class TestExportResultsCsv:
    """Tests for CSV export."""

    def test_exports_to_csv(self, sample_results, tmp_path):
        """Exports results to CSV file."""
        output_path = tmp_path / "results.csv"
        export_results_csv(sample_results, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["question"] == "Question 1"
        assert rows[0]["action"] == "CLOSE"
        assert rows[1]["error"] == "API Error"

    def test_creates_parent_directories(self, sample_results, tmp_path):
        """Creates parent directories if they don't exist."""
        output_path = tmp_path / "subdir" / "nested" / "results.csv"
        export_results_csv(sample_results, output_path)

        assert output_path.exists()

    def test_handles_multiple_sources(self, tmp_path):
        """Handles multiple sources with pipe separator."""
        results = [
            EvaluationResult(
                question="Q1",
                response=AgentResponse(
                    action="CLOSE",
                    message="Answer",
                    sources=["Source A", "Source B", "Source C"],
                ),
                latency_ms=100.0,
            ),
        ]
        output_path = tmp_path / "results.csv"
        export_results_csv(results, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["sources"] == "Source A|Source B|Source C"


class TestExportResultsJson:
    """Tests for JSON export."""

    def test_exports_to_json(self, sample_results, tmp_path):
        """Exports results to JSON file."""
        output_path = tmp_path / "results.json"
        export_results_json(sample_results, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert "results" in data
        assert len(data["results"]) == 2

    def test_includes_summary(self, sample_results, tmp_path):
        """Includes summary when provided."""
        from datetime import UTC, datetime

        from support_agent.evaluation import calculate_summary

        start = datetime.now(UTC)
        end = datetime.now(UTC)
        summary = calculate_summary(sample_results, start, end, eval_tag="test")

        output_path = tmp_path / "results.json"
        export_results_json(sample_results, output_path, summary=summary)

        with open(output_path) as f:
            data = json.load(f)

        assert "summary" in data
        assert data["summary"]["eval_tag"] == "test"


class TestExportResults:
    """Tests for export_results dispatcher."""

    def test_exports_csv_by_default(self, sample_results, tmp_path):
        """Exports CSV by default."""
        output_path = tmp_path / "results.csv"
        export_results(sample_results, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            assert "question" in f.readline()  # CSV header

    def test_exports_json_when_specified(self, sample_results, tmp_path):
        """Exports JSON when format='json'."""
        output_path = tmp_path / "results.json"
        export_results(sample_results, output_path, format="json")

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert "results" in data


# ============================================================================
# EvaluationRunner Tests
# ============================================================================


class TestEvaluationRunner:
    """Tests for EvaluationRunner class."""

    def test_initializes_with_config(self):
        """Initializes with configuration."""
        runner = EvaluationRunner(
            sample_size=100,
            random_seed=42,
            eval_tag="test",
            output_format="json",
        )

        assert runner.config.sample_size == 100
        assert runner.config.random_seed == 42
        assert runner.config.eval_tag == "test"
        assert runner.config.output_format == "json"

    def test_lazy_loads_questions(self, sample_questions_csv):
        """Loads questions lazily."""
        runner = EvaluationRunner(questions_path=sample_questions_csv)

        # Questions not loaded yet
        assert runner._questions is None

        # Access triggers load
        questions = runner.questions
        assert len(questions) == 5
        assert runner._questions is not None

    def test_run_executes_evaluation(self, mock_crewai, sample_questions_csv):
        """Run executes batch evaluation."""
        runner = EvaluationRunner(
            questions_path=sample_questions_csv,
            sample_size=2,
            random_seed=42,
        )

        results, summary = runner.run()

        assert len(results) == 2
        assert summary.total_queries == 2

    def test_export_writes_file(self, mock_crewai, sample_questions_csv, tmp_path):
        """Export writes results to file."""
        runner = EvaluationRunner(
            questions_path=sample_questions_csv,
            sample_size=2,
        )

        results, summary = runner.run()
        output_path = tmp_path / "results.csv"
        runner.export(results, summary, output_path)

        assert output_path.exists()

    def test_format_report_returns_string(self, mock_crewai, sample_questions_csv):
        """Format report returns readable string."""
        runner = EvaluationRunner(
            questions_path=sample_questions_csv,
            sample_size=2,
        )

        results, summary = runner.run()
        report = runner.format_report(summary)

        assert "EVALUATION SUMMARY" in report
        assert "Total Queries:" in report
