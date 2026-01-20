"""Integration tests for the FastAPI backend."""

from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from support_agent.agent.schemas import AgentResponse
from support_agent.api.main import create_app
from support_agent.memory import reset_session_manager


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test."""
    reset_session_manager()
    yield
    reset_session_manager()


@pytest.fixture
def mock_agent_response():
    """Factory for creating mock AgentResponse."""

    def _create(
        action: str = "CLOSE",
        message: str = "Here's the answer to your question.",
        sources: list[str] | None = None,
        confidence: float = 0.9,
    ) -> AgentResponse:
        return AgentResponse(
            action=action,
            message=message,
            sources=sources or ["Getting Started Guide"],
            confidence=confidence,
        )

    return _create


# ============================================================================
# Health Endpoint Tests
# ============================================================================


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client):
        """Health endpoint returns status ok."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data

    def test_health_includes_request_id_header(self, client):
        """Health response includes X-Request-ID header."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers


# ============================================================================
# Chat Endpoint Tests (Single-turn)
# ============================================================================


class TestChatEndpoint:
    """Tests for POST /chat."""

    def test_chat_success(self, client, mock_agent_response):
        """Chat endpoint processes query and returns response."""
        mock_response = mock_agent_response(
            action="CLOSE",
            message="To reset your password, go to Settings > Security.",
            sources=["Password Reset Guide"],
            confidence=0.95,
        )

        with patch("support_agent.api.routes.get_support_crew") as mock_crew:
            mock_crew.return_value.process_query.return_value = mock_response

            response = client.post(
                "/chat",
                json={"message": "How do I reset my password?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "CLOSE"
        assert data["message"] == "To reset your password, go to Settings > Security."
        assert data["sources"] == ["Password Reset Guide"]
        assert data["confidence"] == 0.95

    def test_chat_handover_response(self, client, mock_agent_response):
        """Chat returns HANDOVER for out-of-scope queries."""
        mock_response = mock_agent_response(
            action="HANDOVER",
            message="I'll connect you with a human agent for billing inquiries.",
            sources=[],
            confidence=0.8,
        )

        with patch("support_agent.api.routes.get_support_crew") as mock_crew:
            mock_crew.return_value.process_query.return_value = mock_response

            response = client.post(
                "/chat",
                json={"message": "I want a refund for my order"},
            )

        assert response.status_code == 200
        assert response.json()["action"] == "HANDOVER"

    def test_chat_wait_response(self, client, mock_agent_response):
        """Chat returns WAIT when clarification needed."""
        mock_response = mock_agent_response(
            action="WAIT",
            message="Could you please provide more details about your issue?",
            sources=[],
            confidence=0.6,
        )

        with patch("support_agent.api.routes.get_support_crew") as mock_crew:
            mock_crew.return_value.process_query.return_value = mock_response

            response = client.post(
                "/chat",
                json={"message": "It's not working"},
            )

        assert response.status_code == 200
        assert response.json()["action"] == "WAIT"

    def test_chat_empty_message_rejected(self, client):
        """Chat rejects empty messages."""
        response = client.post("/chat", json={"message": ""})

        assert response.status_code == 422  # Validation error

    def test_chat_missing_message_rejected(self, client):
        """Chat rejects requests without message field."""
        response = client.post("/chat", json={})

        assert response.status_code == 422

    def test_chat_agent_error_returns_500(self, client):
        """Chat returns 500 when agent fails."""
        from support_agent.exceptions import AgentError

        with patch("support_agent.api.routes.get_support_crew") as mock_crew:
            mock_crew.return_value.process_query.side_effect = AgentError(
                "LLM connection failed"
            )

            response = client.post(
                "/chat",
                json={"message": "How do I reset my password?"},
            )

        assert response.status_code == 500


# ============================================================================
# Session Management Tests
# ============================================================================


class TestCreateSession:
    """Tests for POST /sessions."""

    def test_create_session_success(self, client):
        """Creates a new session."""
        response = client.post("/sessions", json={})

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["turn_count"] == 0
        assert data["is_resolved"] is False
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_session_with_metadata(self, client):
        """Creates session with custom metadata."""
        response = client.post(
            "/sessions",
            json={"metadata": {"customer_id": "cust_123", "channel": "web"}},
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data

    def test_create_session_no_body(self, client):
        """Creates session without request body."""
        response = client.post("/sessions")

        assert response.status_code == 201


class TestGetSession:
    """Tests for GET /sessions/{id}."""

    def test_get_session_success(self, client):
        """Retrieves an existing session."""
        # Create session first
        create_response = client.post("/sessions", json={})
        session_id = create_response.json()["id"]

        # Get session
        response = client.get(f"/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert data["turns"] == []
        assert "created_at" in data
        assert "metadata" in data

    def test_get_session_not_found(self, client):
        """Returns 404 for non-existent session."""
        fake_id = str(uuid4())
        response = client.get(f"/sessions/{fake_id}")

        assert response.status_code == 404

    def test_get_session_invalid_uuid(self, client):
        """Returns 422 for invalid UUID format."""
        response = client.get("/sessions/not-a-uuid")

        assert response.status_code == 422


class TestSendMessage:
    """Tests for POST /sessions/{id}/messages."""

    def test_send_message_success(self, client, mock_agent_response):
        """Sends message and gets response in session."""
        from uuid import UUID

        from support_agent.memory import get_session_manager

        # Create session
        create_response = client.post("/sessions", json={})
        session_id = create_response.json()["id"]

        mock_response = mock_agent_response(
            action="WAIT",
            message="What specific feature are you having trouble with?",
            sources=[],
        )

        def mock_process_query(query, session_id, store_in_session=True):
            """Mock that also stores turns like the real implementation."""
            if store_in_session and session_id:
                manager = get_session_manager()
                session = manager.get_session(UUID(str(session_id)))
                session.add_user_message(query)
                session.add_agent_response(mock_response)
            return mock_response

        with patch("support_agent.api.routes.get_support_crew") as mock_crew:
            mock_crew.return_value.process_query.side_effect = mock_process_query

            response = client.post(
                f"/sessions/{session_id}/messages",
                json={"message": "I need help with the integration"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["action"] == "WAIT"
        assert data["turn_count"] == 2  # User message + agent response

    def test_send_message_session_not_found(self, client):
        """Returns 404 for non-existent session."""
        fake_id = str(uuid4())

        response = client.post(
            f"/sessions/{fake_id}/messages",
            json={"message": "Hello"},
        )

        assert response.status_code == 404

    def test_send_message_empty_rejected(self, client):
        """Rejects empty messages."""
        create_response = client.post("/sessions", json={})
        session_id = create_response.json()["id"]

        response = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": ""},
        )

        assert response.status_code == 422

    def test_multi_turn_conversation(self, client, mock_agent_response):
        """Handles multi-turn conversation within session."""
        from uuid import UUID

        from support_agent.memory import get_session_manager

        # Create session
        create_response = client.post("/sessions", json={})
        session_id = create_response.json()["id"]

        # First turn
        response1_mock = mock_agent_response(
            action="WAIT",
            message="What platform are you using?",
        )

        def mock_process_query_1(query, session_id, store_in_session=True):
            if store_in_session and session_id:
                manager = get_session_manager()
                session = manager.get_session(UUID(str(session_id)))
                session.add_user_message(query)
                session.add_agent_response(response1_mock)
            return response1_mock

        with patch("support_agent.api.routes.get_support_crew") as mock_crew:
            mock_crew.return_value.process_query.side_effect = mock_process_query_1

            response1 = client.post(
                f"/sessions/{session_id}/messages",
                json={"message": "Integration not working"},
            )

        assert response1.json()["turn_count"] == 2

        # Second turn
        response2_mock = mock_agent_response(
            action="CLOSE",
            message="For Slack, go to Settings > Apps > Slack and click Reconnect.",
            sources=["Slack Integration Guide"],
        )

        def mock_process_query_2(query, session_id, store_in_session=True):
            if store_in_session and session_id:
                manager = get_session_manager()
                session = manager.get_session(UUID(str(session_id)))
                session.add_user_message(query)
                session.add_agent_response(response2_mock)
            return response2_mock

        with patch("support_agent.api.routes.get_support_crew") as mock_crew:
            mock_crew.return_value.process_query.side_effect = mock_process_query_2

            response2 = client.post(
                f"/sessions/{session_id}/messages",
                json={"message": "I'm using Slack"},
            )

        assert response2.json()["turn_count"] == 4
        assert response2.json()["action"] == "CLOSE"


class TestDeleteSession:
    """Tests for DELETE /sessions/{id}."""

    def test_delete_session_success(self, client):
        """Deletes an existing session."""
        # Create session
        create_response = client.post("/sessions", json={})
        session_id = create_response.json()["id"]

        # Delete session
        response = client.delete(f"/sessions/{session_id}")

        assert response.status_code == 204

        # Verify deleted
        get_response = client.get(f"/sessions/{session_id}")
        assert get_response.status_code == 404

    def test_delete_session_not_found(self, client):
        """Returns 404 for non-existent session."""
        fake_id = str(uuid4())
        response = client.delete(f"/sessions/{fake_id}")

        assert response.status_code == 404

    def test_delete_session_twice(self, client):
        """Returns 404 when deleting already deleted session."""
        # Create and delete
        create_response = client.post("/sessions", json={})
        session_id = create_response.json()["id"]
        client.delete(f"/sessions/{session_id}")

        # Try to delete again
        response = client.delete(f"/sessions/{session_id}")
        assert response.status_code == 404


# ============================================================================
# CORS and Middleware Tests
# ============================================================================


class TestMiddleware:
    """Tests for middleware behavior."""

    def test_request_id_header_present(self, client):
        """All responses include X-Request-ID header."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8  # 8-char UUID prefix

    def test_cors_headers_for_allowed_origin(self, client):
        """CORS headers present for allowed origins."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:7860",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS preflight should succeed
        assert response.status_code == 200


# ============================================================================
# Error Response Format Tests
# ============================================================================


class TestErrorResponses:
    """Tests for error response format consistency."""

    def test_404_error_format(self, client):
        """404 errors have consistent format."""
        fake_id = str(uuid4())
        response = client.get(f"/sessions/{fake_id}")

        assert response.status_code == 404
        # FastAPI HTTPException returns detail field
        data = response.json()
        assert "detail" in data

    def test_validation_error_format(self, client):
        """Validation errors have details."""
        response = client.post("/chat", json={"message": ""})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
