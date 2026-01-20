"""
Metric calculations for evaluation results.

Provides functions to compute action distribution, latency statistics,
and aggregate evaluation summaries.
"""

import statistics
from datetime import datetime

from .schemas import (
    ActionDistribution,
    EvalSummary,
    EvaluationResult,
    LatencyStats,
)


def calculate_action_distribution(
    results: list[EvaluationResult],
) -> ActionDistribution:
    """
    Calculate the distribution of actions across evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        ActionDistribution with counts for each action type
    """
    distribution = ActionDistribution()

    for result in results:
        if not result.success:
            distribution.error += 1
        elif result.action == "CLOSE":
            distribution.close += 1
        elif result.action == "HANDOVER":
            distribution.handover += 1
        elif result.action == "WAIT":
            distribution.wait += 1

    return distribution


def calculate_latency_stats(results: list[EvaluationResult]) -> LatencyStats:
    """
    Calculate latency statistics from evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        LatencyStats with min, max, mean, median, p95, p99
    """
    # Only include successful results for latency stats
    latencies = [r.latency_ms for r in results if r.success]

    if not latencies:
        return LatencyStats()

    latencies_sorted = sorted(latencies)

    return LatencyStats(
        min_ms=min(latencies),
        max_ms=max(latencies),
        mean_ms=statistics.mean(latencies),
        median_ms=statistics.median(latencies),
        p95_ms=_percentile(latencies_sorted, 95),
        p99_ms=_percentile(latencies_sorted, 99),
    )


def _percentile(sorted_data: list[float], percentile: int) -> float:
    """
    Calculate the given percentile of sorted data.

    Args:
        sorted_data: Pre-sorted list of values
        percentile: Percentile to calculate (0-100)

    Returns:
        Value at the given percentile
    """
    if not sorted_data:
        return 0.0

    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]

    # Linear interpolation between closest ranks
    k = (percentile / 100) * (n - 1)
    f = int(k)
    c = f + 1 if f + 1 < n else f

    if f == c:
        return sorted_data[f]

    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def calculate_summary(
    results: list[EvaluationResult],
    start_time: datetime,
    end_time: datetime,
    eval_tag: str | None = None,
) -> EvalSummary:
    """
    Calculate a summary of evaluation results.

    Args:
        results: List of evaluation results
        start_time: When the evaluation started
        end_time: When the evaluation completed
        eval_tag: Optional tag for tracking

    Returns:
        EvalSummary with aggregated statistics
    """
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    duration = (end_time - start_time).total_seconds()

    return EvalSummary(
        total_queries=len(results),
        successful=successful,
        failed=failed,
        action_distribution=calculate_action_distribution(results),
        latency=calculate_latency_stats(results),
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
        eval_tag=eval_tag,
    )


def format_summary_report(summary: EvalSummary) -> str:
    """
    Format an evaluation summary as a human-readable report.

    Args:
        summary: The evaluation summary

    Returns:
        Formatted string report
    """
    dist = summary.action_distribution
    lat = summary.latency

    lines = [
        "=" * 50,
        "EVALUATION SUMMARY",
        "=" * 50,
        f"Total Queries:     {summary.total_queries}",
        f"Successful:        {summary.successful} ({summary.success_rate:.1%})",
        f"Failed:            {summary.failed}",
        f"Duration:          {summary.duration_seconds:.1f}s",
        f"Throughput:        {summary.queries_per_second:.2f} q/s",
        "",
        "ACTION DISTRIBUTION",
        "-" * 30,
        f"CLOSE:             {dist.close} ({dist.close_rate():.1%})",
        f"HANDOVER:          {dist.handover} ({dist.handover_rate():.1%})",
        f"WAIT:              {dist.wait} ({dist.wait_rate():.1%})",
        f"ERROR:             {dist.error} ({dist.error_rate():.1%})",
        "",
        "LATENCY (successful only)",
        "-" * 30,
        f"Min:               {lat.min_ms:.0f}ms",
        f"Max:               {lat.max_ms:.0f}ms",
        f"Mean:              {lat.mean_ms:.0f}ms",
        f"Median:            {lat.median_ms:.0f}ms",
        f"P95:               {lat.p95_ms:.0f}ms",
        f"P99:               {lat.p99_ms:.0f}ms",
        "=" * 50,
    ]

    if summary.eval_tag:
        lines.insert(3, f"Tag:               {summary.eval_tag}")

    return "\n".join(lines)
