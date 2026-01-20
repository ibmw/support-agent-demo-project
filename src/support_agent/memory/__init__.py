"""
Conversation memory module for multi-turn conversation support.

Provides session management and context windowing for the support agent.
"""

from .manager import (
    SessionManager,
    get_session_manager,
    reset_session_manager,
)
from .schemas import ConversationSession, ConversationTurn

__all__ = [
    # Schemas
    "ConversationTurn",
    "ConversationSession",
    # Manager
    "SessionManager",
    "get_session_manager",
    "reset_session_manager",
]
