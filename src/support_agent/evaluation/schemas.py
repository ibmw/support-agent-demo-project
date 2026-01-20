"""
Pydantic schemas for evaluation results and summaries.

Defines data structures for single-turn, batch, and conversation evaluation results.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from ..agent.schemas import ActionType, AgentResponse


# =============================================================================
# Conversation Evaluation Schemas
# =============================================================================


class ExpectedTurn(BaseModel):
    """
    Expected outcome for a single turn in a conversation scenario.

    Attributes:
        action: Expected action for this turn (optional, for golden comparison)
        message_contains: Substrings that should appear in the response (optional)
    """

    action: ActionType | None = None
    message_contains: list[str] = Field(default_factory=list)


class ConversationScenario(BaseModel):
    """
    A multi-turn conversation scenario for evaluation.

    Attributes:
        id: Unique scenario identifier
        name: Human-readable scenario name
        description: Description of what this scenario tests
        category: Optional category for grouping (e.g., "password_reset", "refund")
        user_messages: List of user messages in order
        expected_final_action: The expected action after all turns
        expected_turns: Optional per-turn expectations for golden comparison
        max_turns: Maximum turns allowed before considering scenario failed
    """

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = ""
    category: str | None = None
    user_messages: list[str] = Field(..., min_length=1)
    expected_final_action: ActionType | None = None
    expected_turns: list[ExpectedTurn] = Field(default_factory=list)
    max_turns: int = Field(default=10, ge=1)

    @property
    def turn_count(self) -> int:
        """Number of user turns in this scenario."""
        return len(self.user_messages)


class TurnResult(BaseModel):
    """
    Result of evaluating a single turn in a conversation.

    Attributes:
        turn_index: 0-based index of this turn
        user_message: The user's message
        response: The agent's response (None if error)
        latency_ms: Response time in milliseconds
        error: Error message if turn failed
    """

    turn_index: int = Field(ge=0)
    user_message: str
    response: AgentResponse | None = None
    latency_ms: float = Field(ge=0)
    error: str | None = None

    @computed_field
    @property
    def success(self) -> bool:
        """Whether this turn completed successfully."""
        return self.response is not None and self.error is None

    @computed_field
    @property
    def action(self) -> ActionType | None:
        """The action from the response, if available."""
        return self.response.action if self.response else None


class ConversationResult(BaseModel):
    """
    Result of evaluating a complete conversation scenario.

    Attributes:
        scenario_id: ID of the scenario evaluated
        scenario_name: Name of the scenario
        turns: Results for each turn
        total_latency_ms: Total latency across all turns
        timestamp: When the evaluation was run
        error: Error message if conversation evaluation failed completely
    """

    scenario_id: str
    scenario_name: str
    turns: list[TurnResult] = Field(default_factory=list)
    total_latency_ms: float = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None

    @computed_field
    @property
    def success(self) -> bool:
        """Whether all turns completed successfully."""
        return all(t.success for t in self.turns) and self.error is None

    @computed_field
    @property
    def turn_count(self) -> int:
        """Number of turns evaluated."""
        return len(self.turns)

    @computed_field
    @property
    def final_action(self) -> ActionType | None:
        """The action from the final turn."""
        if not self.turns:
            return None
        return self.turns[-1].action

    @computed_field
    @property
    def is_resolved(self) -> bool:
        """Whether the conversation ended with CLOSE."""
        return self.final_action == "CLOSE"

    @computed_field
    @property
    def is_handed_over(self) -> bool:
        """Whether the conversation ended with HANDOVER."""
        return self.final_action == "HANDOVER"

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per turn."""
        if not self.turns:
            return 0.0
        return self.total_latency_ms / len(self.turns)


class GoldenComparison(BaseModel):
    """
    Comparison of a conversation result against expected outcomes.

    Attributes:
        scenario_id: ID of the scenario
        final_action_match: Whether final action matches expected
        expected_final_action: The expected final action
        actual_final_action: The actual final action
        turn_matches: Per-turn comparison results
        overall_match: Whether the conversation matches expectations overall
    """

    scenario_id: str
    final_action_match: bool
    expected_final_action: ActionType | None
    actual_final_action: ActionType | None
    turn_matches: list[dict] = Field(default_factory=list)

    @computed_field
    @property
    def overall_match(self) -> bool:
        """Whether the conversation matches all expectations."""
        if not self.final_action_match:
            return False
        # If turn matches are specified, all must pass
        if self.turn_matches:
            return all(t.get("match", False) for t in self.turn_matches)
        return True


class ConversationMetrics(BaseModel):
    """
    Aggregate metrics for conversation evaluation.

    Attributes:
        total_scenarios: Number of scenarios evaluated
        successful_scenarios: Scenarios where all turns succeeded
        failed_scenarios: Scenarios with at least one failed turn
        resolution_rate: Percentage ending with CLOSE
        handover_rate: Percentage ending with HANDOVER
        wait_rate: Percentage ending with WAIT
        avg_turns_per_scenario: Average number of turns
        avg_turns_to_resolution: Average turns for resolved conversations
        avg_latency_per_turn_ms: Average latency per turn
        golden_match_rate: Percentage matching expected outcomes (if applicable)
    """

    total_scenarios: int = Field(ge=0)
    successful_scenarios: int = Field(ge=0)
    failed_scenarios: int = Field(ge=0)
    resolution_rate: float = Field(ge=0, le=1)
    handover_rate: float = Field(ge=0, le=1)
    wait_rate: float = Field(ge=0, le=1)
    avg_turns_per_scenario: float = Field(ge=0)
    avg_turns_to_resolution: float | None = None
    avg_latency_per_turn_ms: float = Field(ge=0)
    golden_match_rate: float | None = None


class ConversationEvalSummary(BaseModel):
    """
    Summary of a conversation evaluation run.

    Attributes:
        total_scenarios: Number of scenarios evaluated
        successful: Number of successful scenario evaluations
        failed: Number of failed evaluations
        metrics: Aggregate conversation metrics
        start_time: When the evaluation started
        end_time: When the evaluation completed
        duration_seconds: Total evaluation duration
        eval_tag: Optional tag for this evaluation run
    """

    total_scenarios: int = Field(ge=0)
    successful: int = Field(ge=0)
    failed: int = Field(ge=0)
    metrics: ConversationMetrics
    start_time: datetime
    end_time: datetime
    duration_seconds: float = Field(ge=0)
    eval_tag: str | None = None

    @computed_field
    @property
    def success_rate(self) -> float:
        """Percentage of successful evaluations."""
        return self.successful / self.total_scenarios if self.total_scenarios > 0 else 0.0


class ConversationBatchConfig(BaseModel):
    """Configuration for conversation batch evaluation runs."""

    scenario_ids: list[str] | None = Field(
        default=None,
        description="Specific scenario IDs to evaluate (None = all)",
    )
    categories: list[str] | None = Field(
        default=None,
        description="Filter by scenario categories (None = all)",
    )
    max_concurrent: int = Field(
        default=1,
        ge=1,
        description="Maximum concurrent evaluations",
    )
    output_format: Literal["csv", "json"] = Field(
        default="json",
        description="Output file format",
    )
    eval_tag: str | None = Field(
        default=None,
        description="Tag for this evaluation run (for tracking in LangFuse)",
    )


class EvaluationResult(BaseModel):
    """
    Result of evaluating a single query.

    Attributes:
        question: The original customer query
        response: The agent's response
        latency_ms: Response time in milliseconds
        retrieved_chunks: Number of knowledge base chunks retrieved
        timestamp: When the evaluation was run
        error: Error message if evaluation failed
    """

    question: str
    response: AgentResponse | None = None
    latency_ms: float = Field(ge=0)
    retrieved_chunks: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None

    @computed_field
    @property
    def success(self) -> bool:
        """Whether the evaluation completed successfully."""
        return self.response is not None and self.error is None

    @computed_field
    @property
    def action(self) -> ActionType | None:
        """The action from the response, if available."""
        return self.response.action if self.response else None


class ActionDistribution(BaseModel):
    """Distribution of actions across evaluation results."""

    close: int = 0
    handover: int = 0
    wait: int = 0
    error: int = 0

    @computed_field
    @property
    def total(self) -> int:
        """Total number of evaluations."""
        return self.close + self.handover + self.wait + self.error

    def close_rate(self) -> float:
        """Percentage of CLOSE actions."""
        return self.close / self.total if self.total > 0 else 0.0

    def handover_rate(self) -> float:
        """Percentage of HANDOVER actions."""
        return self.handover / self.total if self.total > 0 else 0.0

    def wait_rate(self) -> float:
        """Percentage of WAIT actions."""
        return self.wait / self.total if self.total > 0 else 0.0

    def error_rate(self) -> float:
        """Percentage of errors."""
        return self.error / self.total if self.total > 0 else 0.0

    def success_rate(self) -> float:
        """Percentage of successful evaluations (non-errors)."""
        if self.total == 0:
            return 0.0
        return 1.0 - self.error_rate()


class LatencyStats(BaseModel):
    """Latency statistics for evaluation results."""

    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


class EvalSummary(BaseModel):
    """
    Summary of a batch evaluation run.

    Attributes:
        total_queries: Total number of queries evaluated
        successful: Number of successful evaluations
        failed: Number of failed evaluations
        action_distribution: Distribution of agent actions
        latency: Latency statistics
        start_time: When the evaluation started
        end_time: When the evaluation completed
        duration_seconds: Total evaluation duration
        eval_tag: Optional tag for this evaluation run (for LangFuse)
    """

    total_queries: int = Field(ge=0)
    successful: int = Field(ge=0)
    failed: int = Field(ge=0)
    action_distribution: ActionDistribution
    latency: LatencyStats
    start_time: datetime
    end_time: datetime
    duration_seconds: float = Field(ge=0)
    eval_tag: str | None = None

    @computed_field
    @property
    def success_rate(self) -> float:
        """Percentage of successful evaluations."""
        return self.successful / self.total_queries if self.total_queries > 0 else 0.0

    @computed_field
    @property
    def queries_per_second(self) -> float:
        """Throughput in queries per second."""
        return (
            self.total_queries / self.duration_seconds
            if self.duration_seconds > 0
            else 0.0
        )


class BatchConfig(BaseModel):
    """Configuration for batch evaluation runs."""

    sample_size: int | None = Field(
        default=None,
        ge=1,
        description="Number of queries to sample (None = all)",
    )
    random_seed: int | None = Field(
        default=None,
        description="Random seed for reproducible sampling",
    )
    max_concurrent: int = Field(
        default=1,
        ge=1,
        description="Maximum concurrent evaluations",
    )
    output_format: Literal["csv", "json"] = Field(
        default="csv",
        description="Output file format",
    )
    eval_tag: str | None = Field(
        default=None,
        description="Tag for this evaluation run (for tracking in LangFuse)",
    )
