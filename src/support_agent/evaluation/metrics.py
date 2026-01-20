"""
Metric calculations for evaluation results.

Provides functions to compute action distribution, latency statistics,
and aggregate evaluation summaries for both single-turn and conversation evaluations.
"""

import statistics
from datetime import datetime

from .schemas import (
    ActionDistribution,
    ConversationEvalSummary,
    ConversationMetrics,
    ConversationResult,
    ConversationScenario,
    EvalSummary,
    EvaluationResult,
    GoldenComparison,
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


# =============================================================================
# Conversation Metrics
# =============================================================================


def calculate_conversation_metrics(
    results: list[ConversationResult],
    scenarios: list[ConversationScenario] | None = None,
) -> ConversationMetrics:
    """
    Calculate aggregate metrics for conversation evaluation results.

    Args:
        results: List of conversation evaluation results
        scenarios: Optional list of scenarios for golden comparison

    Returns:
        ConversationMetrics with aggregate statistics
    """
    if not results:
        return ConversationMetrics(
            total_scenarios=0,
            successful_scenarios=0,
            failed_scenarios=0,
            resolution_rate=0.0,
            handover_rate=0.0,
            wait_rate=0.0,
            avg_turns_per_scenario=0.0,
            avg_latency_per_turn_ms=0.0,
        )

    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    # Count final actions
    resolved = sum(1 for r in results if r.is_resolved)
    handed_over = sum(1 for r in results if r.is_handed_over)
    waiting = sum(1 for r in results if r.final_action == "WAIT")

    # Calculate rates
    resolution_rate = resolved / total
    handover_rate = handed_over / total
    wait_rate = waiting / total

    # Calculate average turns
    total_turns = sum(r.turn_count for r in results)
    avg_turns = total_turns / total if total > 0 else 0.0

    # Calculate average turns to resolution (only for resolved conversations)
    resolved_results = [r for r in results if r.is_resolved]
    avg_to_resolution = (
        sum(r.turn_count for r in resolved_results) / len(resolved_results)
        if resolved_results
        else None
    )

    # Calculate average latency per turn
    successful_results = [r for r in results if r.success]
    total_latency = sum(r.total_latency_ms for r in successful_results)
    successful_turns = sum(r.turn_count for r in successful_results)
    avg_latency_per_turn = total_latency / successful_turns if successful_turns > 0 else 0.0

    # Calculate golden match rate if scenarios provided
    golden_match_rate = None
    if scenarios:
        scenario_map = {s.id: s for s in scenarios}
        matches = 0
        comparable = 0
        for r in results:
            scenario = scenario_map.get(r.scenario_id)
            if scenario and scenario.expected_final_action:
                comparable += 1
                if r.final_action == scenario.expected_final_action:
                    matches += 1
        if comparable > 0:
            golden_match_rate = matches / comparable

    return ConversationMetrics(
        total_scenarios=total,
        successful_scenarios=successful,
        failed_scenarios=failed,
        resolution_rate=resolution_rate,
        handover_rate=handover_rate,
        wait_rate=wait_rate,
        avg_turns_per_scenario=avg_turns,
        avg_turns_to_resolution=avg_to_resolution,
        avg_latency_per_turn_ms=avg_latency_per_turn,
        golden_match_rate=golden_match_rate,
    )


def compare_to_golden(
    result: ConversationResult,
    scenario: ConversationScenario,
) -> GoldenComparison:
    """
    Compare a conversation result against expected outcomes.

    Args:
        result: The conversation evaluation result
        scenario: The scenario with expected outcomes

    Returns:
        GoldenComparison with match results
    """
    # Compare final action
    final_action_match = result.final_action == scenario.expected_final_action

    # Compare per-turn expectations if provided
    turn_matches = []
    for i, expected_turn in enumerate(scenario.expected_turns):
        if i >= len(result.turns):
            turn_matches.append({
                "turn_index": i,
                "match": False,
                "reason": "Turn not evaluated",
            })
            continue

        actual_turn = result.turns[i]
        turn_match = {"turn_index": i, "match": True}

        # Check action match if expected
        if expected_turn.action is not None:
            if actual_turn.action != expected_turn.action:
                turn_match["match"] = False
                turn_match["action_mismatch"] = {
                    "expected": expected_turn.action,
                    "actual": actual_turn.action,
                }

        # Check message contains if specified
        if expected_turn.message_contains and actual_turn.response:
            missing = [
                s for s in expected_turn.message_contains
                if s.lower() not in actual_turn.response.message.lower()
            ]
            if missing:
                turn_match["match"] = False
                turn_match["missing_content"] = missing

        turn_matches.append(turn_match)

    return GoldenComparison(
        scenario_id=scenario.id,
        final_action_match=final_action_match,
        expected_final_action=scenario.expected_final_action,
        actual_final_action=result.final_action,
        turn_matches=turn_matches,
    )


def calculate_conversation_summary(
    results: list[ConversationResult],
    scenarios: list[ConversationScenario] | None,
    start_time: datetime,
    end_time: datetime,
    eval_tag: str | None = None,
) -> ConversationEvalSummary:
    """
    Calculate a summary of conversation evaluation results.

    Args:
        results: List of conversation evaluation results
        scenarios: Optional list of scenarios for golden comparison
        start_time: When the evaluation started
        end_time: When the evaluation completed
        eval_tag: Optional tag for tracking

    Returns:
        ConversationEvalSummary with aggregated statistics
    """
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    duration = (end_time - start_time).total_seconds()

    metrics = calculate_conversation_metrics(results, scenarios)

    return ConversationEvalSummary(
        total_scenarios=len(results),
        successful=successful,
        failed=failed,
        metrics=metrics,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
        eval_tag=eval_tag,
    )


def format_conversation_summary_report(summary: ConversationEvalSummary) -> str:
    """
    Format a conversation evaluation summary as a human-readable report.

    Args:
        summary: The conversation evaluation summary

    Returns:
        Formatted string report
    """
    m = summary.metrics

    lines = [
        "=" * 50,
        "CONVERSATION EVALUATION SUMMARY",
        "=" * 50,
        f"Total Scenarios:   {summary.total_scenarios}",
        f"Successful:        {summary.successful} ({summary.success_rate:.1%})",
        f"Failed:            {summary.failed}",
        f"Duration:          {summary.duration_seconds:.1f}s",
        "",
        "OUTCOME DISTRIBUTION",
        "-" * 30,
        f"Resolved (CLOSE):  {m.resolution_rate:.1%}",
        f"Handover:          {m.handover_rate:.1%}",
        f"Waiting:           {m.wait_rate:.1%}",
        "",
        "CONVERSATION METRICS",
        "-" * 30,
        f"Avg Turns/Scenario: {m.avg_turns_per_scenario:.1f}",
    ]

    if m.avg_turns_to_resolution is not None:
        lines.append(f"Avg Turns to Resolve: {m.avg_turns_to_resolution:.1f}")

    lines.extend([
        f"Avg Latency/Turn:  {m.avg_latency_per_turn_ms:.0f}ms",
    ])

    if m.golden_match_rate is not None:
        lines.extend([
            "",
            "GOLDEN COMPARISON",
            "-" * 30,
            f"Match Rate:        {m.golden_match_rate:.1%}",
        ])

    lines.append("=" * 50)

    if summary.eval_tag:
        lines.insert(3, f"Tag:               {summary.eval_tag}")

    return "\n".join(lines)
