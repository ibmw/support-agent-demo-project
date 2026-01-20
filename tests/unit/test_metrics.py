"""Tests for evaluation metrics calculations."""

from datetime import UTC, datetime, timedelta

import pytest

from support_agent.agent.schemas import AgentResponse
from support_agent.evaluation import (
    ActionDistribution,
    EvalSummary,
    EvaluationResult,
    LatencyStats,
    calculate_action_distribution,
    calculate_latency_stats,
    calculate_summary,
    format_summary_report,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_results() -> list[EvaluationResult]:
    """Sample evaluation results for testing."""
    return [
        EvaluationResult(
            question="How do I reset my password?",
            response=AgentResponse(
                action="CLOSE",
                message="Go to Settings > Security",
                sources=["Password Guide"],
                confidence=0.9,
            ),
            latency_ms=150.0,
            retrieved_chunks=3,
        ),
        EvaluationResult(
            question="I need a refund",
            response=AgentResponse(
                action="HANDOVER",
                message="Connecting you to billing",
                sources=[],
                confidence=0.8,
            ),
            latency_ms=200.0,
            retrieved_chunks=1,
        ),
        EvaluationResult(
            question="It's not working",
            response=AgentResponse(
                action="WAIT",
                message="Could you provide more details?",
                sources=[],
                confidence=0.5,
            ),
            latency_ms=100.0,
            retrieved_chunks=2,
        ),
        EvaluationResult(
            question="Another close query",
            response=AgentResponse(
                action="CLOSE",
                message="Here's the answer",
                sources=["Guide"],
                confidence=0.95,
            ),
            latency_ms=250.0,
            retrieved_chunks=4,
        ),
        EvaluationResult(
            question="Failed query",
            response=None,
            latency_ms=50.0,
            error="LLM API error",
        ),
    ]


# ============================================================================
# ActionDistribution Tests
# ============================================================================


class TestActionDistribution:
    """Tests for ActionDistribution schema."""

    def test_default_values(self):
        """Default values are all zero."""
        dist = ActionDistribution()
        assert dist.close == 0
        assert dist.handover == 0
        assert dist.wait == 0
        assert dist.error == 0
        assert dist.total == 0

    def test_total_computed(self):
        """Total is sum of all actions."""
        dist = ActionDistribution(close=5, handover=3, wait=2, error=1)
        assert dist.total == 11

    def test_close_rate(self):
        """Close rate calculated correctly."""
        dist = ActionDistribution(close=5, handover=3, wait=2, error=0)
        assert dist.close_rate() == pytest.approx(0.5)

    def test_handover_rate(self):
        """Handover rate calculated correctly."""
        dist = ActionDistribution(close=5, handover=3, wait=2, error=0)
        assert dist.handover_rate() == pytest.approx(0.3)

    def test_wait_rate(self):
        """Wait rate calculated correctly."""
        dist = ActionDistribution(close=5, handover=3, wait=2, error=0)
        assert dist.wait_rate() == pytest.approx(0.2)

    def test_error_rate(self):
        """Error rate calculated correctly."""
        dist = ActionDistribution(close=8, handover=0, wait=0, error=2)
        assert dist.error_rate() == pytest.approx(0.2)

    def test_success_rate(self):
        """Success rate is 1 - error_rate."""
        dist = ActionDistribution(close=8, handover=0, wait=0, error=2)
        assert dist.success_rate() == pytest.approx(0.8)

    def test_rates_with_zero_total(self):
        """Rates return 0 when total is 0."""
        dist = ActionDistribution()
        assert dist.close_rate() == 0.0
        assert dist.error_rate() == 0.0
        assert dist.success_rate() == 0.0


# ============================================================================
# LatencyStats Tests
# ============================================================================


class TestLatencyStats:
    """Tests for LatencyStats schema."""

    def test_default_values(self):
        """Default values are all zero."""
        stats = LatencyStats()
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.mean_ms == 0.0
        assert stats.median_ms == 0.0
        assert stats.p95_ms == 0.0
        assert stats.p99_ms == 0.0


# ============================================================================
# calculate_action_distribution Tests
# ============================================================================


class TestCalculateActionDistribution:
    """Tests for calculate_action_distribution function."""

    def test_counts_actions_correctly(self, sample_results):
        """Counts each action type correctly."""
        dist = calculate_action_distribution(sample_results)

        assert dist.close == 2
        assert dist.handover == 1
        assert dist.wait == 1
        assert dist.error == 1
        assert dist.total == 5

    def test_empty_results(self):
        """Empty results return zero counts."""
        dist = calculate_action_distribution([])
        assert dist.total == 0

    def test_all_same_action(self):
        """Handles all same action type."""
        results = [
            EvaluationResult(
                question=f"Q{i}",
                response=AgentResponse(action="CLOSE", message="Done", sources=[]),
                latency_ms=100.0,
            )
            for i in range(5)
        ]
        dist = calculate_action_distribution(results)

        assert dist.close == 5
        assert dist.handover == 0
        assert dist.wait == 0
        assert dist.error == 0

    def test_all_errors(self):
        """Handles all errors."""
        results = [
            EvaluationResult(
                question=f"Q{i}",
                response=None,
                latency_ms=50.0,
                error="Failed",
            )
            for i in range(3)
        ]
        dist = calculate_action_distribution(results)

        assert dist.error == 3
        assert dist.close == 0


# ============================================================================
# calculate_latency_stats Tests
# ============================================================================


class TestCalculateLatencyStats:
    """Tests for calculate_latency_stats function."""

    def test_calculates_basic_stats(self, sample_results):
        """Calculates min, max, mean, median correctly."""
        stats = calculate_latency_stats(sample_results)

        # Only successful results: 150, 200, 100, 250
        assert stats.min_ms == 100.0
        assert stats.max_ms == 250.0
        assert stats.mean_ms == pytest.approx(175.0)
        assert stats.median_ms == pytest.approx(175.0)

    def test_empty_results(self):
        """Empty results return zero stats."""
        stats = calculate_latency_stats([])
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.mean_ms == 0.0

    def test_excludes_failed_results(self):
        """Only successful results included in latency stats."""
        results = [
            EvaluationResult(
                question="Q1",
                response=AgentResponse(action="CLOSE", message="Done", sources=[]),
                latency_ms=100.0,
            ),
            EvaluationResult(
                question="Q2",
                response=None,
                latency_ms=1000.0,  # This should be excluded
                error="Failed",
            ),
        ]
        stats = calculate_latency_stats(results)

        assert stats.max_ms == 100.0  # Not 1000.0

    def test_single_result(self):
        """Single result returns same value for all stats."""
        results = [
            EvaluationResult(
                question="Q1",
                response=AgentResponse(action="CLOSE", message="Done", sources=[]),
                latency_ms=100.0,
            ),
        ]
        stats = calculate_latency_stats(results)

        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0
        assert stats.mean_ms == 100.0
        assert stats.median_ms == 100.0
        assert stats.p95_ms == 100.0
        assert stats.p99_ms == 100.0

    def test_percentiles(self):
        """Percentiles calculated correctly."""
        # Create 100 results with latencies 1-100ms
        results = [
            EvaluationResult(
                question=f"Q{i}",
                response=AgentResponse(action="CLOSE", message="Done", sources=[]),
                latency_ms=float(i + 1),
            )
            for i in range(100)
        ]
        stats = calculate_latency_stats(results)

        assert stats.p95_ms == pytest.approx(95.05, rel=0.01)
        assert stats.p99_ms == pytest.approx(99.01, rel=0.01)


# ============================================================================
# calculate_summary Tests
# ============================================================================


class TestCalculateSummary:
    """Tests for calculate_summary function."""

    def test_calculates_summary(self, sample_results):
        """Creates correct summary from results."""
        start = datetime.now(UTC)
        end = start + timedelta(seconds=10)

        summary = calculate_summary(sample_results, start, end, eval_tag="test-run")

        assert summary.total_queries == 5
        assert summary.successful == 4
        assert summary.failed == 1
        assert summary.duration_seconds == pytest.approx(10.0)
        assert summary.eval_tag == "test-run"
        assert summary.action_distribution.total == 5
        assert summary.latency.min_ms == 100.0

    def test_success_rate_computed(self, sample_results):
        """Success rate computed correctly."""
        start = datetime.now(UTC)
        end = start + timedelta(seconds=1)

        summary = calculate_summary(sample_results, start, end)
        assert summary.success_rate == pytest.approx(0.8)

    def test_queries_per_second_computed(self, sample_results):
        """Queries per second computed correctly."""
        start = datetime.now(UTC)
        end = start + timedelta(seconds=5)

        summary = calculate_summary(sample_results, start, end)
        assert summary.queries_per_second == pytest.approx(1.0)

    def test_empty_results(self):
        """Empty results produce valid summary."""
        start = datetime.now(UTC)
        end = start + timedelta(seconds=1)

        summary = calculate_summary([], start, end)
        assert summary.total_queries == 0
        assert summary.success_rate == 0.0


# ============================================================================
# format_summary_report Tests
# ============================================================================


class TestFormatSummaryReport:
    """Tests for format_summary_report function."""

    def test_contains_key_information(self, sample_results):
        """Report contains key information."""
        start = datetime.now(UTC)
        end = start + timedelta(seconds=10)
        summary = calculate_summary(sample_results, start, end, eval_tag="test-run")

        report = format_summary_report(summary)

        assert "EVALUATION SUMMARY" in report
        assert "Total Queries:" in report
        assert "5" in report
        assert "ACTION DISTRIBUTION" in report
        assert "CLOSE:" in report
        assert "LATENCY" in report
        assert "test-run" in report

    def test_formats_percentages(self, sample_results):
        """Percentages formatted correctly."""
        start = datetime.now(UTC)
        end = start + timedelta(seconds=10)
        summary = calculate_summary(sample_results, start, end)

        report = format_summary_report(summary)

        # Should have percentage signs
        assert "%" in report


# ============================================================================
# EvaluationResult Tests
# ============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult schema."""

    def test_success_when_response_present(self):
        """Success is True when response is present and no error."""
        result = EvaluationResult(
            question="Test",
            response=AgentResponse(action="CLOSE", message="Done", sources=[]),
            latency_ms=100.0,
        )
        assert result.success is True

    def test_success_false_when_error(self):
        """Success is False when error is present."""
        result = EvaluationResult(
            question="Test",
            response=None,
            latency_ms=100.0,
            error="Failed",
        )
        assert result.success is False

    def test_action_from_response(self):
        """Action extracted from response."""
        result = EvaluationResult(
            question="Test",
            response=AgentResponse(action="HANDOVER", message="Escalating", sources=[]),
            latency_ms=100.0,
        )
        assert result.action == "HANDOVER"

    def test_action_none_when_no_response(self):
        """Action is None when no response."""
        result = EvaluationResult(
            question="Test",
            response=None,
            latency_ms=100.0,
            error="Failed",
        )
        assert result.action is None


# ============================================================================
# EvalSummary Tests
# ============================================================================


class TestEvalSummary:
    """Tests for EvalSummary schema."""

    def test_computed_fields(self):
        """Computed fields work correctly."""
        summary = EvalSummary(
            total_queries=100,
            successful=80,
            failed=20,
            action_distribution=ActionDistribution(
                close=60, handover=15, wait=5, error=20
            ),
            latency=LatencyStats(),
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(seconds=50),
            duration_seconds=50.0,
        )

        assert summary.success_rate == pytest.approx(0.8)
        assert summary.queries_per_second == pytest.approx(2.0)
