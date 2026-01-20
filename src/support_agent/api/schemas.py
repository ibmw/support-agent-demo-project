"""
API request and response schemas.

Defines the HTTP interface models, separate from internal domain schemas.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ..agent.schemas import ActionType

# ============================================================================
# Health Check
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Chat (Single-turn)
# ============================================================================


class ChatRequest(BaseModel):
    """Single-turn chat request (stateless)."""

    message: str = Field(..., min_length=1, max_length=4000, description="User message")


class ChatResponse(BaseModel):
    """Chat response from the agent."""

    action: ActionType = Field(
        ..., description="Action: CLOSE (resolved), HANDOVER (escalate), WAIT (clarify)"
    )
    message: str = Field(..., description="Agent response message")
    sources: list[str] = Field(
        default_factory=list, description="Knowledge base sources referenced"
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score"
    )


# ============================================================================
# Sessions (Multi-turn)
# ============================================================================


class CreateSessionRequest(BaseModel):
    """Request to create a new conversation session."""

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata (e.g., customer_id, channel)",
    )


class SessionResponse(BaseModel):
    """Session information response."""

    id: UUID = Field(..., description="Session ID")
    turn_count: int = Field(..., description="Number of turns in the conversation")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_action: ActionType | None = Field(
        default=None, description="Action from last assistant turn"
    )
    is_resolved: bool = Field(
        default=False, description="Whether conversation was resolved (CLOSE)"
    )


class TurnResponse(BaseModel):
    """A single conversation turn."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Turn timestamp")
    action: ActionType | None = Field(
        default=None, description="Action (assistant turns only)"
    )
    sources: list[str] = Field(
        default_factory=list, description="Sources (assistant turns only)"
    )


class SessionHistoryResponse(BaseModel):
    """Full session with conversation history."""

    id: UUID = Field(..., description="Session ID")
    turns: list[TurnResponse] = Field(
        default_factory=list, description="Conversation turns"
    )
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Session metadata"
    )


class SendMessageRequest(BaseModel):
    """Request to send a message in a session (multi-turn)."""

    message: str = Field(..., min_length=1, max_length=4000, description="User message")


class SendMessageResponse(BaseModel):
    """Response after sending a message in a session."""

    session_id: UUID = Field(..., description="Session ID")
    action: ActionType = Field(..., description="Agent action")
    message: str = Field(..., description="Agent response message")
    sources: list[str] = Field(
        default_factory=list, description="Knowledge base sources"
    )
    confidence: float | None = Field(default=None, description="Confidence score")
    turn_count: int = Field(..., description="Total turns after this exchange")


# ============================================================================
# Error Responses
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: str | None = Field(default=None, description="Request ID for tracing")
