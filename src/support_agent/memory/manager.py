"""
Session management for conversation memory.

Provides in-memory session storage with context windowing support.
"""

from uuid import UUID

from ..config import settings
from ..exceptions import SessionNotFoundError
from ..logging import get_logger
from .schemas import ConversationSession, ConversationTurn

logger = get_logger(__name__, component="memory")


class SessionManager:
    """
    Manages conversation sessions in memory.

    Provides session lifecycle management (create, retrieve, update, expire)
    and context windowing for multi-turn conversations.

    Note:
        Sessions are stored in memory and will be lost on restart.
        For persistence, consider extending with a storage backend.
    """

    def __init__(self, max_turns: int | None = None):
        """
        Initialize the session manager.

        Args:
            max_turns: Maximum turns to keep in context (defaults to settings)
        """
        self._sessions: dict[UUID, ConversationSession] = {}
        self.max_turns = max_turns or settings.max_conversation_turns

        logger.info(
            "SessionManager initialized",
            max_turns=self.max_turns,
        )

    def create_session(
        self, metadata: dict[str, str] | None = None
    ) -> ConversationSession:
        """
        Create a new conversation session.

        Args:
            metadata: Optional metadata to attach (e.g., customer_id, channel)

        Returns:
            New ConversationSession instance
        """
        session = ConversationSession(metadata=metadata or {})
        self._sessions[session.id] = session

        logger.info(
            "Session created",
            session_id=str(session.id),
            metadata=metadata,
        )
        return session

    def get_session(self, session_id: UUID) -> ConversationSession:
        """
        Retrieve a session by ID.

        Args:
            session_id: The session UUID

        Returns:
            The conversation session

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning(
                "Session not found",
                session_id=str(session_id),
            )
            raise SessionNotFoundError(f"Session {session_id} not found")
        return session

    def get_or_create_session(
        self, session_id: UUID | None = None, metadata: dict[str, str] | None = None
    ) -> ConversationSession:
        """
        Get an existing session or create a new one.

        Args:
            session_id: Optional session ID to retrieve
            metadata: Metadata for new session (ignored if session exists)

        Returns:
            Existing or new ConversationSession
        """
        if session_id is not None:
            try:
                return self.get_session(session_id)
            except SessionNotFoundError:
                pass

        return self.create_session(metadata=metadata)

    def add_turn(self, session_id: UUID, turn: ConversationTurn) -> ConversationSession:
        """
        Add a turn to a session.

        Args:
            session_id: The session to update
            turn: The turn to add

        Returns:
            Updated session

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        session = self.get_session(session_id)
        session.add_turn(turn)

        logger.debug(
            "Turn added to session",
            session_id=str(session_id),
            role=turn.role,
            turn_count=session.turn_count,
        )
        return session

    def get_context_window(self, session_id: UUID, max_turns: int | None = None) -> str:
        """
        Get the context window for a session.

        Args:
            session_id: The session ID
            max_turns: Override max turns (defaults to instance setting)

        Returns:
            Formatted conversation history string

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        session = self.get_session(session_id)
        turns_to_include = max_turns if max_turns is not None else self.max_turns

        context = session.get_context_window(turns_to_include)

        # Log if context was truncated
        if session.turn_count > turns_to_include:
            logger.debug(
                "Context window truncated",
                session_id=str(session_id),
                total_turns=session.turn_count,
                included_turns=turns_to_include,
            )

        return context

    def delete_session(self, session_id: UUID) -> bool:
        """
        Delete a session.

        Args:
            session_id: The session to delete

        Returns:
            True if session was deleted, False if it didn't exist
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(
                "Session deleted",
                session_id=str(session_id),
            )
            return True
        return False

    def clear_all_sessions(self) -> int:
        """
        Clear all sessions (useful for testing).

        Returns:
            Number of sessions cleared
        """
        count = len(self._sessions)
        self._sessions.clear()
        logger.info("All sessions cleared", session_count=count)
        return count

    @property
    def session_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    def list_sessions(self) -> list[UUID]:
        """List all session IDs."""
        return list(self._sessions.keys())


# Module-level singleton for convenience
_manager: SessionManager | None = None


def get_session_manager(**kwargs) -> SessionManager:
    """
    Get or create the singleton SessionManager instance.

    Args:
        **kwargs: Arguments passed to SessionManager constructor (only used on first call)

    Returns:
        SessionManager instance
    """
    global _manager
    if _manager is None:
        _manager = SessionManager(**kwargs)
    return _manager


def reset_session_manager() -> None:
    """Reset the singleton manager (useful for testing)."""
    global _manager
    _manager = None
