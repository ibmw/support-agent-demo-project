"""
Pydantic schemas for conversation memory management.

Defines data structures for tracking conversation turns and sessions.
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ..agent.schemas import ActionType, AgentResponse


class ConversationTurn(BaseModel):
    """
    A single turn in a conversation.

    Attributes:
        role: Who sent this message ('user' or 'assistant')
        content: The message content
        timestamp: When the message was sent
        action: The action taken by the assistant (only for assistant turns)
        sources: Knowledge base sources referenced (only for assistant turns)
    """

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    action: ActionType | None = None
    sources: list[str] = Field(default_factory=list)

    @classmethod
    def from_user(cls, content: str) -> "ConversationTurn":
        """Create a user turn."""
        return cls(role="user", content=content)

    @classmethod
    def from_agent_response(cls, response: AgentResponse) -> "ConversationTurn":
        """Create an assistant turn from an AgentResponse."""
        return cls(
            role="assistant",
            content=response.message,
            action=response.action,
            sources=response.sources,
        )

    def to_context_string(self) -> str:
        """Format turn for context window (to include in prompts)."""
        role_label = "Customer" if self.role == "user" else "Support Agent"
        return f"{role_label}: {self.content}"


class ConversationSession(BaseModel):
    """
    A conversation session containing multiple turns.

    Attributes:
        id: Unique session identifier
        turns: List of conversation turns in chronological order
        created_at: When the session was created
        updated_at: When the session was last updated
        metadata: Optional metadata (e.g., customer_id, channel)
    """

    id: UUID = Field(default_factory=uuid4)
    turns: list[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str] = Field(default_factory=dict)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn and update the timestamp."""
        self.turns.append(turn)
        self.updated_at = datetime.now(UTC)

    def add_user_message(self, content: str) -> ConversationTurn:
        """Convenience method to add a user message."""
        turn = ConversationTurn.from_user(content)
        self.add_turn(turn)
        return turn

    def add_agent_response(self, response: AgentResponse) -> ConversationTurn:
        """Convenience method to add an agent response."""
        turn = ConversationTurn.from_agent_response(response)
        self.add_turn(turn)
        return turn

    def get_last_n_turns(self, n: int) -> list[ConversationTurn]:
        """
        Get the last N turns for context windowing.

        Args:
            n: Number of turns to retrieve

        Returns:
            List of the most recent turns (may be fewer if session is shorter)
        """
        if n <= 0:
            return []
        return self.turns[-n:]

    def get_context_window(self, max_turns: int) -> str:
        """
        Format recent turns as a context string for the agent prompt.

        Args:
            max_turns: Maximum number of turns to include

        Returns:
            Formatted conversation history string
        """
        recent_turns = self.get_last_n_turns(max_turns)
        if not recent_turns:
            return ""
        return "\n".join(turn.to_context_string() for turn in recent_turns)

    @property
    def turn_count(self) -> int:
        """Number of turns in the conversation."""
        return len(self.turns)

    @property
    def last_action(self) -> ActionType | None:
        """Get the action from the most recent assistant turn."""
        for turn in reversed(self.turns):
            if turn.role == "assistant" and turn.action:
                return turn.action
        return None

    def is_resolved(self) -> bool:
        """Check if the conversation was resolved (last action was CLOSE)."""
        return self.last_action == "CLOSE"

    def is_handed_over(self) -> bool:
        """Check if the conversation was handed over to a human."""
        return self.last_action == "HANDOVER"
