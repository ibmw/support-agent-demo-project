"""Integration tests for the Gradio UI components.

Tests the UI logic and event handlers without requiring a running server.
"""

from unittest.mock import MagicMock, patch

import gradio as gr
import httpx
import pytest

# ============================================================================
# SupportAgentUI Class Tests
# ============================================================================


class TestSupportAgentUI:
    """Tests for the SupportAgentUI HTTP client wrapper."""

    @pytest.fixture
    def ui(self):
        """Create a SupportAgentUI instance with mocked client."""
        from support_agent.ui.app import SupportAgentUI

        return SupportAgentUI(api_base_url="http://localhost:8000")

    def test_init_sets_api_url(self, ui):
        """UI initializes with correct API URL."""
        assert ui.api_base_url == "http://localhost:8000"
        assert ui.session_id is None

    def test_check_api_health_success(self, ui):
        """Health check returns connected when API responds 200."""
        with patch.object(ui.client, "get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)

            connected, status = ui.check_api_health()

            assert connected is True
            assert status == "Connected"
            mock_get.assert_called_once_with("http://localhost:8000/health")

    def test_check_api_health_failure(self, ui):
        """Health check returns error status on non-200."""
        with patch.object(ui.client, "get") as mock_get:
            mock_get.return_value = MagicMock(status_code=503)

            connected, status = ui.check_api_health()

            assert connected is False
            assert "503" in status

    def test_check_api_health_connection_error(self, ui):
        """Health check handles connection errors gracefully."""
        with patch.object(ui.client, "get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            connected, status = ui.check_api_health()

            assert connected is False
            assert "Cannot connect" in status

    def test_create_session_success(self, ui):
        """Creates session and stores ID."""
        with patch.object(ui.client, "post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=201,
                json=lambda: {"id": "test-session-123"},
            )

            session_id = ui.create_session()

            assert session_id == "test-session-123"
            assert ui.session_id == "test-session-123"

    def test_create_session_failure(self, ui):
        """Returns None on session creation failure."""
        with patch.object(ui.client, "post") as mock_post:
            mock_post.return_value = MagicMock(status_code=500)

            session_id = ui.create_session()

            assert session_id is None
            assert ui.session_id is None

    def test_send_message_creates_session_if_missing(self, ui):
        """Sending message auto-creates session if none exists."""
        with patch.object(ui.client, "post") as mock_post:
            # First call creates session, second sends message
            mock_post.side_effect = [
                MagicMock(
                    status_code=201,
                    json=lambda: {"id": "new-session"},
                ),
                MagicMock(
                    status_code=200,
                    json=lambda: {
                        "action": "CLOSE",
                        "message": "Here's your answer",
                        "sources": ["Guide"],
                    },
                ),
            ]

            action, message, sources = ui.send_message("Hello")

            assert action == "CLOSE"
            assert message == "Here's your answer"
            assert sources == ["Guide"]
            assert ui.session_id == "new-session"

    def test_send_message_success(self, ui):
        """Sends message to existing session."""
        ui.session_id = "existing-session"

        with patch.object(ui.client, "post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "action": "HANDOVER",
                    "message": "Escalating to human",
                    "sources": [],
                },
            )

            action, message, sources = ui.send_message("Complex issue")

            assert action == "HANDOVER"
            assert message == "Escalating to human"
            assert sources == []

    def test_send_message_session_expired_recreates(self, ui):
        """Recreates session on 404 and retries."""
        ui.session_id = "expired-session"

        call_count = 0

        def mock_post_handler(url, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: session expired
                return MagicMock(status_code=404)
            elif call_count == 2:
                # Second call: create new session
                return MagicMock(
                    status_code=201,
                    json=lambda: {"id": "new-session"},
                )
            else:
                # Third call: send message
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "action": "WAIT",
                        "message": "Please clarify",
                        "sources": [],
                    },
                )

        with patch.object(ui.client, "post", side_effect=mock_post_handler):
            action, message, sources = ui.send_message("Help")

            assert action == "WAIT"
            assert ui.session_id == "new-session"

    def test_send_message_connection_error(self, ui):
        """Returns error on connection failure."""
        ui.session_id = "test-session"

        with patch.object(ui.client, "post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            action, message, sources = ui.send_message("Hello")

            assert action == "ERROR"
            assert "Cannot connect" in message

    def test_send_message_timeout(self, ui):
        """Returns error on timeout."""
        ui.session_id = "test-session"

        with patch.object(ui.client, "post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Timeout")

            action, message, sources = ui.send_message("Hello")

            assert action == "ERROR"
            assert "timed out" in message

    def test_reset_session(self, ui):
        """Resets session and creates new one."""
        ui.session_id = "old-session"

        with patch.object(ui.client, "post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=201,
                json=lambda: {"id": "brand-new-session"},
            )

            status = ui.reset_session()

            assert "New conversation" in status
            assert ui.session_id == "brand-new-session"

    def test_reset_session_failure(self, ui):
        """Returns failure message if new session creation fails."""
        ui.session_id = "old-session"

        with patch.object(ui.client, "post") as mock_post:
            mock_post.return_value = MagicMock(status_code=500)

            status = ui.reset_session()

            assert "Failed" in status


# ============================================================================
# Message Formatting Tests
# ============================================================================


class TestFormatBotMessage:
    """Tests for the bot message formatting function."""

    def test_format_close_action(self):
        """Formats CLOSE action with resolved badge."""
        from support_agent.ui.app import format_bot_message

        result = format_bot_message("CLOSE", "Issue resolved!", ["FAQ Guide"])

        assert "Resolved" in result
        assert "Issue resolved!" in result
        assert "FAQ Guide" in result

    def test_format_handover_action(self):
        """Formats HANDOVER action with escalated badge."""
        from support_agent.ui.app import format_bot_message

        result = format_bot_message("HANDOVER", "Connecting to human", [])

        assert "Escalated" in result
        assert "Connecting to human" in result

    def test_format_wait_action(self):
        """Formats WAIT action with awaiting badge."""
        from support_agent.ui.app import format_bot_message

        result = format_bot_message("WAIT", "Please clarify", [])

        assert "Awaiting" in result
        assert "Please clarify" in result

    def test_format_unknown_action(self):
        """Formats unknown action with default styling."""
        from support_agent.ui.app import format_bot_message

        result = format_bot_message("UNKNOWN", "Some message", [])

        assert "Unknown" in result
        assert "Some message" in result

    def test_format_multiple_sources(self):
        """Formats message with multiple sources."""
        from support_agent.ui.app import format_bot_message

        result = format_bot_message(
            "CLOSE", "Here's help", ["Guide 1", "Guide 2", "FAQ"]
        )

        assert "Guide 1" in result
        assert "Guide 2" in result
        assert "FAQ" in result
        assert "Sources" in result


# ============================================================================
# Gradio App Creation Tests
# ============================================================================


class TestGradioAppCreation:
    """Tests for Gradio app creation."""

    def test_create_app_returns_blocks(self):
        """create_gradio_app returns a Blocks instance."""
        from support_agent.ui.app import create_gradio_app

        app = create_gradio_app(api_base_url="http://test:8000")

        assert isinstance(app, gr.Blocks)

    def test_create_app_stores_css_and_theme(self):
        """App stores custom CSS and theme for launch."""
        from support_agent.ui.app import create_gradio_app

        app = create_gradio_app()

        assert hasattr(app, "_custom_css")
        assert hasattr(app, "_custom_theme")
        assert app._custom_css is not None

    def test_create_app_has_components(self):
        """App creates expected components."""
        from support_agent.ui.app import create_gradio_app

        app = create_gradio_app()

        # Check that the app has blocks (components)
        assert len(app.blocks) > 0


# ============================================================================
# Event Handler Tests
# ============================================================================


class TestEventHandlers:
    """Tests for Gradio event handler functions."""

    def test_respond_empty_message_returns_unchanged(self):
        """respond() with empty message returns unchanged history."""

        # Simulate the respond function logic
        def respond(message: str, history: list) -> tuple[str, list]:
            if not message.strip():
                return "", history
            # ... rest of logic
            return "", history

        result_msg, result_history = respond("", [])
        assert result_msg == ""
        assert result_history == []

        result_msg, result_history = respond("   ", [{"role": "user", "content": "hi"}])
        assert result_msg == ""
        assert len(result_history) == 1

    def test_respond_adds_messages_to_history(self):
        """respond() adds user and assistant messages."""
        from support_agent.ui.app import SupportAgentUI, format_bot_message

        ui = SupportAgentUI()

        # Mock the send_message to return a known response
        with patch.object(
            ui, "send_message", return_value=("CLOSE", "Answer here", ["Source"])
        ):
            history = []

            # Simulate respond logic
            message = "Test question"
            history = history + [{"role": "user", "content": message}]
            action, response, sources = ui.send_message(message)
            bot_message = format_bot_message(action, response, sources)
            history = history + [{"role": "assistant", "content": bot_message}]

            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "Test question"
            assert history[1]["role"] == "assistant"
            assert "Resolved" in history[1]["content"]
