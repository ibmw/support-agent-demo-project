"""
Evaluation module for the support agent.

Provides batch evaluation, metrics calculation, and result export functionality
for both single-turn and multi-turn conversation evaluations.
"""

from .metrics import (
    calculate_action_distribution,
    calculate_conversation_metrics,
    calculate_conversation_summary,
    calculate_latency_stats,
    calculate_summary,
    compare_to_golden,
    format_conversation_summary_report,
    format_summary_report,
)
from .runner import (
    ConversationEvaluationRunner,
    EvaluationRunner,
    evaluate_conversation,
    evaluate_single,
    export_conversation_results,
    export_conversation_results_csv,
    export_conversation_results_json,
    export_results,
    export_results_csv,
    export_results_json,
    filter_scenarios,
    iter_batch_evaluation,
    iter_conversation_evaluation,
    load_questions,
    load_scenarios,
    run_batch_evaluation,
    run_conversation_evaluation,
    sample_questions,
)
from .schemas import (
    ActionDistribution,
    BatchConfig,
    ConversationBatchConfig,
    ConversationEvalSummary,
    ConversationMetrics,
    ConversationResult,
    ConversationScenario,
    EvalSummary,
    EvaluationResult,
    ExpectedTurn,
    GoldenComparison,
    LatencyStats,
    TurnResult,
)

__all__ = [
    # Single-turn Schemas
    "EvaluationResult",
    "EvalSummary",
    "ActionDistribution",
    "LatencyStats",
    "BatchConfig",
    # Conversation Schemas
    "ConversationScenario",
    "ExpectedTurn",
    "TurnResult",
    "ConversationResult",
    "GoldenComparison",
    "ConversationMetrics",
    "ConversationEvalSummary",
    "ConversationBatchConfig",
    # Single-turn Metrics
    "calculate_action_distribution",
    "calculate_latency_stats",
    "calculate_summary",
    "format_summary_report",
    # Conversation Metrics
    "calculate_conversation_metrics",
    "calculate_conversation_summary",
    "compare_to_golden",
    "format_conversation_summary_report",
    # Single-turn Runner
    "EvaluationRunner",
    "load_questions",
    "sample_questions",
    "evaluate_single",
    "run_batch_evaluation",
    "iter_batch_evaluation",
    "export_results",
    "export_results_csv",
    "export_results_json",
    # Conversation Runner
    "ConversationEvaluationRunner",
    "load_scenarios",
    "filter_scenarios",
    "evaluate_conversation",
    "run_conversation_evaluation",
    "iter_conversation_evaluation",
    "export_conversation_results",
    "export_conversation_results_csv",
    "export_conversation_results_json",
]
