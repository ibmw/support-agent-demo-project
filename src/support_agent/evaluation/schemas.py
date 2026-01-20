"""
Pydantic schemas for evaluation results and summaries.

Defines data structures for single-turn and batch evaluation results.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from ..agent.schemas import ActionType, AgentResponse


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
