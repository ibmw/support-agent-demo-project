"""
Pydantic schemas for agent responses and actions.

Defines the structured output format for the support agent.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Action types for the support agent
ActionType = Literal["CLOSE", "HANDOVER", "WAIT"]


class AgentResponse(BaseModel):
    """
    Structured response from the support agent.

    Attributes:
        action: The recommended action for this query
            - CLOSE: Response is sufficient to handle the query
            - HANDOVER: Query is outside scope, escalate to human
            - WAIT: Query needs clarification before proceeding
        message: The response message to the customer
        sources: List of knowledge base sources used (article titles or sections)
        confidence: Optional confidence score (0.0 to 1.0)
    """

    action: ActionType = Field(
        ...,
        description="Action to take: CLOSE (resolved), HANDOVER (escalate), WAIT (clarify)",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Response message to the customer",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Knowledge base sources referenced in the response",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the response (0.0 to 1.0)",
    )

    def is_resolved(self) -> bool:
        """Check if the query was resolved (CLOSE action)."""
        return self.action == "CLOSE"

    def needs_human(self) -> bool:
        """Check if the query needs human intervention."""
        return self.action == "HANDOVER"

    def needs_clarification(self) -> bool:
        """Check if the query needs clarification from the user."""
        return self.action == "WAIT"
