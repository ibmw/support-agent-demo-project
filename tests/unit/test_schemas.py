"""Unit tests for agent schemas."""

import pytest
from pydantic import ValidationError

from support_agent.agent.schemas import AgentResponse


class TestAgentResponse:
    """Tests for the AgentResponse Pydantic model."""

    def test_valid_close_response(self):
        """Creates valid CLOSE response."""
        response = AgentResponse(
            action="CLOSE",
            message="Here's how to set up your integration.",
            sources=["Integration Guide", "Getting Started"],
            confidence=0.95,
        )
        assert response.action == "CLOSE"
        assert "integration" in response.message.lower()
        assert len(response.sources) == 2
        assert response.confidence == 0.95

    def test_valid_handover_response(self):
        """Creates valid HANDOVER response."""
        response = AgentResponse(
            action="HANDOVER",
            message="I'll connect you with a billing specialist.",
            sources=[],
            confidence=0.8,
        )
        assert response.action == "HANDOVER"
        assert response.sources == []

    def test_valid_wait_response(self):
        """Creates valid WAIT response."""
        response = AgentResponse(
            action="WAIT",
            message="Could you clarify which feature you're asking about?",
        )
        assert response.action == "WAIT"
        assert response.sources == []  # Default empty list
        assert response.confidence is None  # Default None

    def test_minimal_response(self):
        """Creates response with only required fields."""
        response = AgentResponse(
            action="CLOSE",
            message="Your question has been answered.",
        )
        assert response.action == "CLOSE"
        assert response.message == "Your question has been answered."
        assert response.sources == []
        assert response.confidence is None

    def test_invalid_action_rejected(self):
        """Rejects invalid action type."""
        with pytest.raises(ValidationError) as exc_info:
            AgentResponse(
                action="INVALID",
                message="Some message",
            )
        errors = exc_info.value.errors()
        assert any("action" in str(e["loc"]) for e in errors)

    def test_empty_message_rejected(self):
        """Rejects empty message."""
        with pytest.raises(ValidationError) as exc_info:
            AgentResponse(
                action="CLOSE",
                message="",
            )
        errors = exc_info.value.errors()
        assert any("message" in str(e["loc"]) for e in errors)

    def test_confidence_must_be_between_0_and_1(self):
        """Confidence must be in valid range."""
        # Valid at boundaries
        response_low = AgentResponse(action="CLOSE", message="Test", confidence=0.0)
        assert response_low.confidence == 0.0

        response_high = AgentResponse(action="CLOSE", message="Test", confidence=1.0)
        assert response_high.confidence == 1.0

        # Invalid below 0
        with pytest.raises(ValidationError):
            AgentResponse(action="CLOSE", message="Test", confidence=-0.1)

        # Invalid above 1
        with pytest.raises(ValidationError):
            AgentResponse(action="CLOSE", message="Test", confidence=1.1)

    def test_sources_must_be_list(self):
        """Sources must be a list of strings."""
        # Single string should fail type validation
        with pytest.raises(ValidationError):
            AgentResponse(
                action="CLOSE",
                message="Test",
                sources="Single Source",  # Should be list
            )

    def test_is_resolved_helper(self):
        """is_resolved returns True only for CLOSE."""
        close_response = AgentResponse(action="CLOSE", message="Done")
        handover_response = AgentResponse(action="HANDOVER", message="Escalating")
        wait_response = AgentResponse(action="WAIT", message="Clarify please")

        assert close_response.is_resolved() is True
        assert handover_response.is_resolved() is False
        assert wait_response.is_resolved() is False

    def test_needs_human_helper(self):
        """needs_human returns True only for HANDOVER."""
        close_response = AgentResponse(action="CLOSE", message="Done")
        handover_response = AgentResponse(action="HANDOVER", message="Escalating")
        wait_response = AgentResponse(action="WAIT", message="Clarify please")

        assert close_response.needs_human() is False
        assert handover_response.needs_human() is True
        assert wait_response.needs_human() is False

    def test_needs_clarification_helper(self):
        """needs_clarification returns True only for WAIT."""
        close_response = AgentResponse(action="CLOSE", message="Done")
        handover_response = AgentResponse(action="HANDOVER", message="Escalating")
        wait_response = AgentResponse(action="WAIT", message="Clarify please")

        assert close_response.needs_clarification() is False
        assert handover_response.needs_clarification() is False
        assert wait_response.needs_clarification() is True

    def test_model_serialization(self):
        """Response can be serialized to dict/JSON."""
        response = AgentResponse(
            action="CLOSE",
            message="Here's your answer.",
            sources=["Article 1"],
            confidence=0.9,
        )
        data = response.model_dump()

        assert data["action"] == "CLOSE"
        assert data["message"] == "Here's your answer."
        assert data["sources"] == ["Article 1"]
        assert data["confidence"] == 0.9

    def test_model_json_serialization(self):
        """Response can be serialized to JSON string."""
        response = AgentResponse(
            action="HANDOVER",
            message="Escalating to human.",
        )
        json_str = response.model_dump_json()

        assert '"action":"HANDOVER"' in json_str or '"action": "HANDOVER"' in json_str
        assert "Escalating to human." in json_str

    def test_model_parsing_from_dict(self):
        """Response can be parsed from dict."""
        data = {
            "action": "WAIT",
            "message": "Please clarify.",
            "sources": ["FAQ"],
            "confidence": 0.6,
        }
        response = AgentResponse(**data)

        assert response.action == "WAIT"
        assert response.sources == ["FAQ"]

    def test_action_case_sensitive(self):
        """Action type is case-sensitive (must be uppercase)."""
        with pytest.raises(ValidationError):
            AgentResponse(action="close", message="Test")

        with pytest.raises(ValidationError):
            AgentResponse(action="Close", message="Test")
