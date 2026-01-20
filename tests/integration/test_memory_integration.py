"""Integration tests for memory integration with the support agent crew."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from support_agent.agent.crew import SupportCrew, reset_crew
from support_agent.agent.schemas import AgentResponse
from support_agent.exceptions import SessionNotFoundError
from support_agent.memory import (
    SessionManager,
    get_session_manager,
    reset_session_manager,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons before each test."""
    reset_crew()
    reset_session_manager()
    yield
    reset_crew()
    reset_session_manager()


@pytest.fixture
def mock_crew_result():
    """Factory fixture for creating mock crew results with pydantic output."""

    def _create_result(
        action: str = "CLOSE",
        message: str = "Here's the answer to your question.",
        sources: list[str] | None = None,
        confidence: float = 0.9,
    ) -> MagicMock:
        """Create a mock crew result with pydantic AgentResponse."""
        response = AgentResponse(
            action=action,
            message=message,
            sources=sources or ["Getting Started Guide"],
            confidence=confidence,
        )
        mock_result = MagicMock()
        mock_result.pydantic = response
        mock_result.tasks_output = []
        mock_result.token_usage = None
        return mock_result

    return _create_result


@pytest.fixture
def mock_crewai():
    """Mock CrewAI components."""
    with (
        patch("support_agent.agent.crew.Agent") as mock_agent,
        patch("support_agent.agent.crew.Crew") as mock_crew_cls,
        patch("support_agent.agent.crew.Task") as mock_task,
    ):
        yield {
            "agent": mock_agent,
            "crew_cls": mock_crew_cls,
            "task": mock_task,
        }


class TestProcessQueryWithSession:
    """Tests for process_query with session_id parameter."""

    def test_process_query_without_session(self, mock_crewai, mock_crew_result):
        """Query without session_id works as before."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        crew = SupportCrew()
        response = crew.process_query("How do I reset my password?")

        assert response.action == "CLOSE"
        # No session created
        assert get_session_manager().session_count == 0

    def test_process_query_with_new_session(self, mock_crewai, mock_crew_result):
        """Query with session_id uses conversation history."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        manager = get_session_manager()
        session = manager.create_session()

        crew = SupportCrew()
        response = crew.process_query(
            "How do I reset my password?", session_id=session.id
        )

        assert response.action == "CLOSE"
        # Query and response stored in session
        assert session.turn_count == 2  # user + assistant

    def test_process_query_stores_user_message(self, mock_crewai, mock_crew_result):
        """User query is stored in session before processing."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        manager = get_session_manager()
        session = manager.create_session()

        crew = SupportCrew()
        crew.process_query("What is the pricing?", session_id=session.id)

        # First turn is user message
        assert session.turns[0].role == "user"
        assert session.turns[0].content == "What is the pricing?"

    def test_process_query_stores_agent_response(self, mock_crewai, mock_crew_result):
        """Agent response is stored in session after processing."""
        result = mock_crew_result(
            action="CLOSE",
            message="Our pricing starts at $10/month.",
            sources=["Pricing Guide"],
        )
        mock_crewai["crew_cls"].return_value.kickoff.return_value = result

        manager = get_session_manager()
        session = manager.create_session()

        crew = SupportCrew()
        crew.process_query("What is the pricing?", session_id=session.id)

        # Second turn is assistant message
        assert session.turns[1].role == "assistant"
        assert session.turns[1].content == "Our pricing starts at $10/month."
        assert session.turns[1].action == "CLOSE"
        assert "Pricing Guide" in session.turns[1].sources

    def test_process_query_with_store_disabled(self, mock_crewai, mock_crew_result):
        """Query with store_in_session=False doesn't store turns."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        manager = get_session_manager()
        session = manager.create_session()

        crew = SupportCrew()
        crew.process_query(
            "How do I reset my password?",
            session_id=session.id,
            store_in_session=False,
        )

        # No turns stored
        assert session.turn_count == 0

    def test_process_query_nonexistent_session_raises(self, mock_crewai):
        """Query with non-existent session_id raises error."""
        crew = SupportCrew()
        fake_id = uuid4()

        with pytest.raises(SessionNotFoundError):
            crew.process_query("Test query", session_id=fake_id)


class TestProcessQueryWithHistory:
    """Tests for conversation history in task description."""

    def test_task_includes_history(self, mock_crewai, mock_crew_result):
        """Task description includes conversation history."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        manager = get_session_manager()
        session = manager.create_session()
        # Add previous conversation
        session.add_user_message("What integrations do you support?")
        session.add_agent_response(
            AgentResponse(
                action="CLOSE",
                message="We support Slack, GitHub, and Jira.",
                sources=["Integration Guide"],
            )
        )

        crew = SupportCrew()
        crew.process_query("How do I set up Slack?", session_id=session.id)

        # Verify Task was called with history in description
        task_call_kwargs = mock_crewai["task"].call_args.kwargs
        description = task_call_kwargs.get("description", "")
        assert "Conversation History" in description
        assert "What integrations do you support" in description
        assert "Slack, GitHub, and Jira" in description

    def test_task_without_history_for_new_session(self, mock_crewai, mock_crew_result):
        """Task description uses simple template for new session."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        manager = get_session_manager()
        session = manager.create_session()  # Empty session

        crew = SupportCrew()
        crew.process_query("How do I reset my password?", session_id=session.id)

        # Verify Task uses simple template (no history section)
        task_call_kwargs = mock_crewai["task"].call_args.kwargs
        description = task_call_kwargs.get("description", "")
        # Should NOT have conversation history section for empty session
        assert "Conversation History" not in description


class TestProcessConversation:
    """Tests for process_conversation convenience method."""

    def test_creates_new_session_when_none_provided(
        self, mock_crewai, mock_crew_result
    ):
        """Creates new session when session_id is None."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        crew = SupportCrew()
        response, session_id = crew.process_conversation("Hello!")

        assert response.action == "CLOSE"
        assert session_id is not None

        # Session exists
        manager = get_session_manager()
        session = manager.get_session(session_id)
        assert session.turn_count == 2

    def test_uses_existing_session_when_provided(self, mock_crewai, mock_crew_result):
        """Uses existing session when session_id is provided."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        manager = get_session_manager()
        existing = manager.create_session()

        crew = SupportCrew()
        _, returned_id = crew.process_conversation("Hello!", session_id=existing.id)

        assert returned_id == existing.id
        assert manager.session_count == 1  # No new session created

    def test_creates_session_if_not_found(self, mock_crewai, mock_crew_result):
        """Creates new session if provided session_id doesn't exist."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        fake_id = uuid4()

        crew = SupportCrew()
        _, session_id = crew.process_conversation("Hello!", session_id=fake_id)

        # New session created (not the fake one)
        assert session_id != fake_id

    def test_creates_session_with_metadata(self, mock_crewai, mock_crew_result):
        """Creates session with provided metadata."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        crew = SupportCrew()
        _, session_id = crew.process_conversation(
            "Hello!",
            metadata={"customer_id": "cust-123", "channel": "web"},
        )

        manager = get_session_manager()
        session = manager.get_session(session_id)
        assert session.metadata["customer_id"] == "cust-123"
        assert session.metadata["channel"] == "web"

    def test_multi_turn_conversation(self, mock_crewai, mock_crew_result):
        """Full multi-turn conversation flow works."""
        # First response
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result(
            action="WAIT",
            message="Which product are you asking about?",
        )

        crew = SupportCrew()

        # Turn 1
        response1, session_id = crew.process_conversation("I have a question")
        assert response1.action == "WAIT"

        # Turn 2 - update mock for next response
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result(
            action="CLOSE",
            message="You can reset your password in Settings.",
            sources=["Password Reset Guide"],
        )

        response2, _ = crew.process_conversation(
            "About resetting my password in the mobile app",
            session_id=session_id,
        )
        assert response2.action == "CLOSE"

        # Verify full conversation stored
        manager = get_session_manager()
        session = manager.get_session(session_id)
        assert session.turn_count == 4  # 2 user + 2 assistant

        # Verify order
        assert session.turns[0].role == "user"
        assert session.turns[1].role == "assistant"
        assert session.turns[2].role == "user"
        assert session.turns[3].role == "assistant"


@pytest.mark.integration
class TestContextWindowIntegration:
    """Integration tests for context windowing with the agent."""

    def test_context_window_limits_history(self, mock_crewai, mock_crew_result):
        """Context window limits conversation history sent to agent."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        # Create session manager with small window
        reset_session_manager()
        manager = SessionManager(max_turns=2)
        # Manually set the singleton
        import support_agent.memory.manager as manager_module

        manager_module._manager = manager

        session = manager.create_session()
        # Add many turns
        for i in range(5):
            session.add_user_message(f"Message {i}")
            session.add_agent_response(
                AgentResponse(action="WAIT", message=f"Response {i}", sources=[])
            )

        crew = SupportCrew()
        crew.process_query("Final question", session_id=session.id)

        # Verify task description only contains recent history
        task_call_kwargs = mock_crewai["task"].call_args.kwargs
        description = task_call_kwargs.get("description", "")

        # Should have last 2 turns (1 user + 1 assistant from turns 4)
        assert "Message 4" in description
        assert "Response 4" in description
        # Should NOT have earlier messages
        assert "Message 0" not in description
        assert "Message 1" not in description

    def test_session_status_after_close(self, mock_crewai, mock_crew_result):
        """Session is marked as resolved after CLOSE action."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result(
            action="CLOSE",
            message="Issue resolved!",
        )

        crew = SupportCrew()
        _, session_id = crew.process_conversation("Help me please")

        manager = get_session_manager()
        session = manager.get_session(session_id)
        assert session.is_resolved() is True
        assert session.is_handed_over() is False

    def test_session_status_after_handover(self, mock_crewai, mock_crew_result):
        """Session is marked as handed over after HANDOVER action."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result(
            action="HANDOVER",
            message="Connecting you to a human.",
        )

        crew = SupportCrew()
        _, session_id = crew.process_conversation("I need a refund")

        manager = get_session_manager()
        session = manager.get_session(session_id)
        assert session.is_resolved() is False
        assert session.is_handed_over() is True
