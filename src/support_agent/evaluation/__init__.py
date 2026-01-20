"""
Evaluation module for the support agent.

Provides batch evaluation, metrics calculation, and result export functionality.
"""

from .metrics import (
    calculate_action_distribution,
    calculate_latency_stats,
    calculate_summary,
    format_summary_report,
)
from .runner import (
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
from .schemas import (
    ActionDistribution,
    BatchConfig,
    EvalSummary,
    EvaluationResult,
    LatencyStats,
)

__all__ = [
    # Schemas
    "EvaluationResult",
    "EvalSummary",
    "ActionDistribution",
    "LatencyStats",
    "BatchConfig",
    # Metrics
    "calculate_action_distribution",
    "calculate_latency_stats",
    "calculate_summary",
    "format_summary_report",
    # Runner
    "EvaluationRunner",
    "load_questions",
    "sample_questions",
    "evaluate_single",
    "run_batch_evaluation",
    "iter_batch_evaluation",
    "export_results",
    "export_results_csv",
    "export_results_json",
]
