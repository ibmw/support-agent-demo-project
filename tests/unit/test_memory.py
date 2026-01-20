"""Tests for the memory module (session management and windowing)."""

from datetime import UTC
from uuid import uuid4

import pytest

from support_agent.agent.schemas import AgentResponse
from support_agent.exceptions import SessionNotFoundError
from support_agent.memory import (
    ConversationSession,
    ConversationTurn,
    SessionManager,
    get_session_manager,
    reset_session_manager,
)


class TestConversationTurn:
    """Tests for ConversationTurn schema."""

    def test_create_user_turn(self):
        """Test creating a user turn."""
        turn = ConversationTurn.from_user("How do I reset my password?")

        assert turn.role == "user"
        assert turn.content == "How do I reset my password?"
        assert turn.action is None
        assert turn.sources == []
        assert turn.timestamp is not None

    def test_create_assistant_turn_from_response(self):
        """Test creating an assistant turn from AgentResponse."""
        response = AgentResponse(
            action="CLOSE",
            message="You can reset your password in Settings > Security.",
            sources=["Password Reset Guide"],
            confidence=0.95,
        )
        turn = ConversationTurn.from_agent_response(response)

        assert turn.role == "assistant"
        assert turn.content == response.message
        assert turn.action == "CLOSE"
        assert turn.sources == ["Password Reset Guide"]

    def test_turn_to_context_string_user(self):
        """Test context string formatting for user turn."""
        turn = ConversationTurn.from_user("What's your refund policy?")
        assert turn.to_context_string() == "Customer: What's your refund policy?"

    def test_turn_to_context_string_assistant(self):
        """Test context string formatting for assistant turn."""
        response = AgentResponse(
            action="CLOSE",
            message="We offer 30-day refunds.",
            sources=[],
        )
        turn = ConversationTurn.from_agent_response(response)
        assert turn.to_context_string() == "Support Agent: We offer 30-day refunds."

    def test_turn_timestamp_is_utc(self):
        """Test that timestamps are in UTC."""
        turn = ConversationTurn.from_user("test")
        assert turn.timestamp.tzinfo == UTC

    def test_turn_requires_content(self):
        """Test that content cannot be empty."""
        with pytest.raises(ValueError):
            ConversationTurn(role="user", content="")


class TestConversationSession:
    """Tests for ConversationSession schema."""

    def test_create_empty_session(self):
        """Test creating an empty session."""
        session = ConversationSession()

        assert session.id is not None
        assert session.turns == []
        assert session.turn_count == 0
        assert session.metadata == {}
        assert session.created_at is not None
        assert session.updated_at is not None

    def test_create_session_with_metadata(self):
        """Test creating a session with metadata."""
        session = ConversationSession(
            metadata={"customer_id": "cust-123", "channel": "web"}
        )

        assert session.metadata["customer_id"] == "cust-123"
        assert session.metadata["channel"] == "web"

    def test_add_user_message(self):
        """Test adding a user message."""
        session = ConversationSession()
        turn = session.add_user_message("Hello!")

        assert session.turn_count == 1
        assert turn.role == "user"
        assert turn.content == "Hello!"

    def test_add_agent_response(self):
        """Test adding an agent response."""
        session = ConversationSession()
        response = AgentResponse(
            action="WAIT",
            message="Could you provide more details?",
            sources=[],
        )
        turn = session.add_agent_response(response)

        assert session.turn_count == 1
        assert turn.role == "assistant"
        assert turn.action == "WAIT"

    def test_get_last_n_turns(self):
        """Test retrieving last N turns."""
        session = ConversationSession()
        session.add_user_message("First")
        session.add_user_message("Second")
        session.add_user_message("Third")

        last_two = session.get_last_n_turns(2)
        assert len(last_two) == 2
        assert last_two[0].content == "Second"
        assert last_two[1].content == "Third"

    def test_get_last_n_turns_exceeds_total(self):
        """Test that get_last_n_turns returns all if n > total."""
        session = ConversationSession()
        session.add_user_message("Only one")

        turns = session.get_last_n_turns(10)
        assert len(turns) == 1

    def test_get_last_n_turns_zero(self):
        """Test that n=0 returns empty list."""
        session = ConversationSession()
        session.add_user_message("test")

        assert session.get_last_n_turns(0) == []

    def test_get_last_n_turns_negative(self):
        """Test that negative n returns empty list."""
        session = ConversationSession()
        session.add_user_message("test")

        assert session.get_last_n_turns(-1) == []

    def test_get_context_window(self):
        """Test formatting context window."""
        session = ConversationSession()
        session.add_user_message("Hi there!")
        session.add_agent_response(
            AgentResponse(action="WAIT", message="Hello! How can I help?", sources=[])
        )
        session.add_user_message("I need help with billing")

        context = session.get_context_window(max_turns=3)
        expected = (
            "Customer: Hi there!\n"
            "Support Agent: Hello! How can I help?\n"
            "Customer: I need help with billing"
        )
        assert context == expected

    def test_get_context_window_empty_session(self):
        """Test context window for empty session."""
        session = ConversationSession()
        assert session.get_context_window(max_turns=5) == ""

    def test_last_action(self):
        """Test getting last action from session."""
        session = ConversationSession()
        session.add_user_message("Question 1")
        session.add_agent_response(
            AgentResponse(action="WAIT", message="Need more info", sources=[])
        )
        session.add_user_message("Here's more info")
        session.add_agent_response(
            AgentResponse(action="CLOSE", message="Thanks!", sources=[])
        )

        assert session.last_action == "CLOSE"

    def test_last_action_no_assistant_turns(self):
        """Test last_action when there are no assistant turns."""
        session = ConversationSession()
        session.add_user_message("Hello")
        assert session.last_action is None

    def test_is_resolved(self):
        """Test is_resolved check."""
        session = ConversationSession()
        session.add_agent_response(
            AgentResponse(action="CLOSE", message="Done!", sources=[])
        )
        assert session.is_resolved() is True

    def test_is_handed_over(self):
        """Test is_handed_over check."""
        session = ConversationSession()
        session.add_agent_response(
            AgentResponse(action="HANDOVER", message="Escalating...", sources=[])
        )
        assert session.is_handed_over() is True

    def test_updated_at_changes_on_add_turn(self):
        """Test that updated_at changes when turns are added."""
        session = ConversationSession()
        initial_updated = session.updated_at

        # Small delay to ensure timestamp difference
        import time

        time.sleep(0.01)

        session.add_user_message("New message")
        assert session.updated_at > initial_updated


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singleton before each test."""
        reset_session_manager()
        yield
        reset_session_manager()

    def test_create_session(self):
        """Test creating a new session."""
        manager = SessionManager(max_turns=5)
        session = manager.create_session()

        assert session.id is not None
        assert manager.session_count == 1

    def test_create_session_with_metadata(self):
        """Test creating session with metadata."""
        manager = SessionManager()
        session = manager.create_session(metadata={"user": "test"})

        assert session.metadata["user"] == "test"

    def test_get_session(self):
        """Test retrieving an existing session."""
        manager = SessionManager()
        session = manager.create_session()

        retrieved = manager.get_session(session.id)
        assert retrieved.id == session.id

    def test_get_session_not_found(self):
        """Test that getting non-existent session raises error."""
        manager = SessionManager()
        fake_id = uuid4()

        with pytest.raises(SessionNotFoundError):
            manager.get_session(fake_id)

    def test_get_or_create_session_existing(self):
        """Test get_or_create returns existing session."""
        manager = SessionManager()
        original = manager.create_session()

        session = manager.get_or_create_session(session_id=original.id)
        assert session.id == original.id

    def test_get_or_create_session_new(self):
        """Test get_or_create creates new session if not found."""
        manager = SessionManager()
        manager.get_or_create_session(session_id=uuid4())

        assert manager.session_count == 1

    def test_get_or_create_session_no_id(self):
        """Test get_or_create creates new session when no ID provided."""
        manager = SessionManager()
        session = manager.get_or_create_session()

        assert session is not None
        assert manager.session_count == 1

    def test_add_turn(self):
        """Test adding a turn to session via manager."""
        manager = SessionManager()
        session = manager.create_session()
        turn = ConversationTurn.from_user("Hello")

        updated = manager.add_turn(session.id, turn)
        assert updated.turn_count == 1

    def test_add_turn_session_not_found(self):
        """Test adding turn to non-existent session raises error."""
        manager = SessionManager()
        turn = ConversationTurn.from_user("Hello")

        with pytest.raises(SessionNotFoundError):
            manager.add_turn(uuid4(), turn)

    def test_get_context_window(self):
        """Test getting context window via manager."""
        manager = SessionManager(max_turns=2)
        session = manager.create_session()
        session.add_user_message("First")
        session.add_user_message("Second")
        session.add_user_message("Third")

        context = manager.get_context_window(session.id)

        # Should only include last 2 turns
        assert "First" not in context
        assert "Second" in context
        assert "Third" in context

    def test_get_context_window_custom_max(self):
        """Test context window with custom max_turns override."""
        manager = SessionManager(max_turns=10)
        session = manager.create_session()
        session.add_user_message("First")
        session.add_user_message("Second")
        session.add_user_message("Third")

        context = manager.get_context_window(session.id, max_turns=1)
        assert "First" not in context
        assert "Second" not in context
        assert "Third" in context

    def test_delete_session(self):
        """Test deleting a session."""
        manager = SessionManager()
        session = manager.create_session()

        result = manager.delete_session(session.id)
        assert result is True
        assert manager.session_count == 0

    def test_delete_session_not_found(self):
        """Test deleting non-existent session returns False."""
        manager = SessionManager()
        result = manager.delete_session(uuid4())
        assert result is False

    def test_clear_all_sessions(self):
        """Test clearing all sessions."""
        manager = SessionManager()
        manager.create_session()
        manager.create_session()
        manager.create_session()

        count = manager.clear_all_sessions()
        assert count == 3
        assert manager.session_count == 0

    def test_list_sessions(self):
        """Test listing all session IDs."""
        manager = SessionManager()
        s1 = manager.create_session()
        s2 = manager.create_session()

        sessions = manager.list_sessions()
        assert len(sessions) == 2
        assert s1.id in sessions
        assert s2.id in sessions

    def test_default_max_turns_from_settings(self):
        """Test that max_turns defaults to settings value."""
        from support_agent.config import settings

        manager = SessionManager()
        assert manager.max_turns == settings.max_conversation_turns


class TestSessionManagerSingleton:
    """Tests for the singleton pattern."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singleton before each test."""
        reset_session_manager()
        yield
        reset_session_manager()

    def test_get_session_manager_creates_singleton(self):
        """Test that get_session_manager creates a singleton."""
        manager1 = get_session_manager()
        manager2 = get_session_manager()

        assert manager1 is manager2

    def test_reset_session_manager(self):
        """Test that reset creates a new instance."""
        manager1 = get_session_manager()
        reset_session_manager()
        manager2 = get_session_manager()

        assert manager1 is not manager2

    def test_singleton_kwargs_only_on_first_call(self):
        """Test that kwargs only apply on first call."""
        manager1 = get_session_manager(max_turns=3)
        manager2 = get_session_manager(max_turns=100)

        # Second call's kwargs are ignored
        assert manager1.max_turns == 3
        assert manager2.max_turns == 3
