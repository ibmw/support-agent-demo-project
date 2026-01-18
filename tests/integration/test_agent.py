"""Integration tests for the support agent crew."""

from unittest.mock import MagicMock, patch

import pytest

from support_agent.agent.crew import SupportCrew, get_support_crew, reset_crew
from support_agent.agent.schemas import AgentResponse
from support_agent.exceptions import AgentError, InvalidResponseError


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the crew singleton before each test."""
    reset_crew()
    yield
    reset_crew()


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


class TestSupportCrewInit:
    """Tests for SupportCrew initialization."""

    def test_default_initialization(self, mock_crewai):
        """Crew initializes with default settings."""
        with patch("support_agent.agent.crew.settings") as mock_settings:
            mock_settings.llm_model = "gpt-4o-mini"

            crew = SupportCrew()

            assert crew.model == "gpt-4o-mini"
            assert crew.temperature == 0.7
            assert crew.verbose is False

    def test_custom_model(self, mock_crewai):
        """Crew accepts custom model."""
        crew = SupportCrew(model="gpt-4o")
        assert crew.model == "gpt-4o"

    def test_custom_temperature(self, mock_crewai):
        """Crew accepts custom temperature."""
        crew = SupportCrew(temperature=0.5)
        assert crew.temperature == 0.5

    def test_verbose_mode(self, mock_crewai):
        """Crew accepts verbose flag."""
        crew = SupportCrew(verbose=True)
        assert crew.verbose is True


class TestSupportCrewProcessQuery:
    """Tests for query processing."""

    def test_successful_close_response(self, mock_crewai, mock_crew_result):
        """Processes query and returns CLOSE response."""
        mock_result = mock_crew_result(
            action="CLOSE",
            message="To set up the integration, go to Settings > Integrations.",
            sources=["Integration Guide"],
            confidence=0.95,
        )
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_result

        crew = SupportCrew()
        response = crew.process_query("How do I set up the integration?")

        assert isinstance(response, AgentResponse)
        assert response.action == "CLOSE"
        assert "integration" in response.message.lower()
        assert "Integration Guide" in response.sources
        assert response.confidence == 0.95

    def test_handover_response(self, mock_crewai, mock_crew_result):
        """Processes query and returns HANDOVER response."""
        mock_result = mock_crew_result(
            action="HANDOVER",
            message="I'll connect you with our billing team.",
            sources=[],
            confidence=0.8,
        )
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_result

        crew = SupportCrew()
        response = crew.process_query("I need a refund for my subscription.")

        assert response.action == "HANDOVER"
        assert response.needs_human() is True

    def test_wait_response(self, mock_crewai, mock_crew_result):
        """Processes query and returns WAIT response."""
        mock_result = mock_crew_result(
            action="WAIT",
            message="Could you tell me which product you're asking about?",
            sources=[],
            confidence=0.5,
        )
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_result

        crew = SupportCrew()
        response = crew.process_query("It's not working")

        assert response.action == "WAIT"
        assert response.needs_clarification() is True

    def test_creates_task_with_query(self, mock_crewai, mock_crew_result):
        """Task is created with the query."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result()

        crew = SupportCrew()
        crew.process_query("How do I reset my password?")

        # Verify Task was called with output_pydantic
        mock_crewai["task"].assert_called_once()
        task_call_kwargs = mock_crewai["task"].call_args.kwargs
        assert "reset my password" in task_call_kwargs.get("description", "")
        assert task_call_kwargs.get("output_pydantic") == AgentResponse

    def test_uses_tasks_output_fallback(self, mock_crewai):
        """Falls back to tasks_output when pydantic is None."""
        response = AgentResponse(
            action="CLOSE",
            message="Answer from task output",
            sources=["Source"],
            confidence=0.85,
        )
        mock_task_output = MagicMock()
        mock_task_output.pydantic = response

        mock_result = MagicMock()
        mock_result.pydantic = None
        mock_result.tasks_output = [mock_task_output]
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_result

        crew = SupportCrew()
        result = crew.process_query("Test query")

        assert result.action == "CLOSE"
        assert result.message == "Answer from task output"


class TestErrorHandling:
    """Tests for error handling."""

    def test_raises_agent_error_on_execution_failure(self, mock_crewai):
        """Raises AgentError when crew execution fails."""
        mock_crewai["crew_cls"].return_value.kickoff.side_effect = RuntimeError(
            "LLM API error"
        )

        crew = SupportCrew()

        with pytest.raises(AgentError) as exc_info:
            crew.process_query("Test query")

        assert "Failed to process query" in str(exc_info.value)

    def test_raises_invalid_response_when_no_pydantic(self, mock_crewai):
        """Raises InvalidResponseError when no pydantic output available."""
        mock_result = MagicMock()
        mock_result.pydantic = None
        mock_result.tasks_output = []
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_result

        crew = SupportCrew()

        with pytest.raises(InvalidResponseError) as exc_info:
            crew.process_query("Test query")

        assert "valid pydantic response" in str(exc_info.value)

    def test_raises_invalid_response_on_wrong_type(self, mock_crewai):
        """Raises InvalidResponseError when pydantic output is wrong type."""
        mock_result = MagicMock()
        mock_result.pydantic = {"not": "an AgentResponse"}  # Wrong type
        mock_result.tasks_output = []
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_result

        crew = SupportCrew()

        with pytest.raises(InvalidResponseError) as exc_info:
            crew.process_query("Test query")

        assert "Expected AgentResponse" in str(exc_info.value)


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_support_crew_returns_singleton(self, mock_crewai):
        """get_support_crew returns same instance."""
        crew1 = get_support_crew()
        crew2 = get_support_crew()

        assert crew1 is crew2

    def test_reset_crew_clears_singleton(self, mock_crewai):
        """reset_crew clears the singleton."""
        crew1 = get_support_crew()
        reset_crew()
        crew2 = get_support_crew()

        assert crew1 is not crew2


@pytest.mark.integration
class TestSupportCrewIntegration:
    """Integration tests that test actual component interaction.

    These tests verify the crew components work together correctly
    but still mock external dependencies (LLM, vector DB).
    """

    def test_full_query_flow_with_mocked_llm(self, mock_crewai, mock_crew_result):
        """Full query processing flow works end-to-end."""
        mock_result = mock_crew_result(
            action="CLOSE",
            message="Based on the Integration Guide, you need to: 1. Go to Settings 2. Click Integrations 3. Select your app",
            sources=["Integration Guide > Steps"],
            confidence=0.92,
        )
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_result

        crew = SupportCrew(model="gpt-4o-mini", verbose=False)
        response = crew.process_query("How do I connect my integration?")

        # Verify response structure
        assert response.action == "CLOSE"
        assert response.is_resolved()
        assert len(response.sources) > 0
        assert response.confidence > 0.8

        # Verify crew was created and used
        mock_crewai["crew_cls"].assert_called()
        mock_crewai["crew_cls"].return_value.kickoff.assert_called_once()
