"""
Gradio chat interface for the support agent.

Communicates with the FastAPI backend for all operations.
Compatible with Gradio 6.0+.
"""

import gradio as gr
import httpx

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__, component="ui")

# API base URL
API_BASE_URL = f"http://{settings.api_host}:{settings.api_port}"

# Action badge styling
ACTION_STYLES = {
    "CLOSE": ("✓ Resolved", "#22c55e", "#dcfce7"),  # Green
    "HANDOVER": ("↗ Escalated", "#f59e0b", "#fef3c7"),  # Amber
    "WAIT": ("⏳ Awaiting", "#3b82f6", "#dbeafe"),  # Blue
}

# Custom CSS for a distinctive look
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary-color: #6366f1;
    --primary-hover: #4f46e5;
    --bg-gradient-start: #0f172a;
    --bg-gradient-end: #1e1b4b;
    --surface-color: rgba(30, 41, 59, 0.8);
    --surface-border: rgba(99, 102, 241, 0.2);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --accent-glow: rgba(99, 102, 241, 0.4);
}

.gradio-container {
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%) !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.1) 0%, transparent 100%);
    border-bottom: 1px solid var(--surface-border);
    margin-bottom: 1.5rem;
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.main-header p {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

.chat-container {
    background: var(--surface-color) !important;
    border: 1px solid var(--surface-border) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2), 0 0 40px var(--accent-glow);
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--surface-color);
    border: 1px solid var(--surface-border);
    border-radius: 8px;
    font-size: 0.85rem;
    color: var(--text-primary);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

.status-connected {
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
}

.status-error {
    background: #ef4444;
    box-shadow: 0 0 8px #ef4444;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.new-session-btn {
    background: linear-gradient(135deg, var(--primary-color) 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(99, 102, 241, 0.3) !important;
}

.new-session-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
}

footer {
    display: none !important;
}
"""

# Custom theme for Gradio 6.0
CUSTOM_THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Space Grotesk"),
)


def format_bot_message(action: str, message: str, sources: list[str]) -> str:
    """Format the bot response with action badge and sources."""
    label, color, bg_color = ACTION_STYLES.get(
        action, ("Unknown", "#6b7280", "#f3f4f6")
    )

    # Build the response with action badge
    parts = [f"**{label}**\n\n{message}"]

    # Add sources if present
    if sources:
        sources_text = " • ".join(f"`{s}`" for s in sources)
        parts.append(f"\n\n---\n📚 **Sources:** {sources_text}")

    return "".join(parts)


class SupportAgentUI:
    """Gradio UI that communicates with the FastAPI backend."""

    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url
        self.session_id: str | None = None
        self.client = httpx.Client(timeout=120.0)  # Long timeout for LLM calls

    def check_api_health(self) -> tuple[bool, str]:
        """Check if the API is healthy."""
        try:
            response = self.client.get(f"{self.api_base_url}/health")
            if response.status_code == 200:
                return True, "Connected"
            return False, f"API returned {response.status_code}"
        except httpx.ConnectError:
            return False, "Cannot connect to API"
        except Exception as e:
            return False, str(e)

    def create_session(self) -> str | None:
        """Create a new session via the API."""
        try:
            response = self.client.post(f"{self.api_base_url}/sessions", json={})
            if response.status_code == 201:
                data = response.json()
                self.session_id = data["id"]
                logger.info("Session created", session_id=self.session_id)
                return self.session_id
            logger.error("Failed to create session", status=response.status_code)
            return None
        except Exception as e:
            logger.error("Session creation error", error=str(e))
            return None

    def send_message(self, message: str) -> tuple[str, str, list[str]]:
        """
        Send a message to the current session.

        Returns:
            Tuple of (action, response_message, sources)
        """
        if not self.session_id:
            self.create_session()

        if not self.session_id:
            return "ERROR", "Failed to create session. Is the API running?", []

        try:
            response = self.client.post(
                f"{self.api_base_url}/sessions/{self.session_id}/messages",
                json={"message": message},
            )

            if response.status_code == 200:
                data = response.json()
                return data["action"], data["message"], data.get("sources", [])
            elif response.status_code == 404:
                # Session expired, create new one and retry
                self.create_session()
                return self.send_message(message)
            else:
                return "ERROR", f"API error: {response.status_code}", []

        except httpx.ConnectError:
            return (
                "ERROR",
                "Cannot connect to API. Make sure the server is running.",
                [],
            )
        except httpx.TimeoutException:
            return "ERROR", "Request timed out. The agent is taking too long.", []
        except Exception as e:
            logger.error("Message send error", error=str(e))
            return "ERROR", f"Error: {str(e)}", []

    def reset_session(self) -> str:
        """Reset the current session."""
        self.session_id = None
        if self.create_session():
            return "✨ New conversation started!"
        return "⚠️ Failed to start new conversation"


def create_gradio_app(api_base_url: str | None = None) -> gr.Blocks:
    """
    Create the Gradio chat interface.

    Args:
        api_base_url: Override API URL (defaults to settings)

    Returns:
        Configured Gradio Blocks app
    """
    ui = SupportAgentUI(api_base_url or API_BASE_URL)

    # Gradio 6.0: theme and css go in launch(), but we store them for later
    with gr.Blocks(title="Support Agent") as app:
        # Store theme/css for launch
        app._custom_css = CUSTOM_CSS
        app._custom_theme = CUSTOM_THEME

        # Header
        gr.HTML(
            """
            <div class="main-header">
                <h1>🤖 Support Agent</h1>
                <p>AI-powered customer support assistant</p>
            </div>
            """
        )

        # Status indicator
        with gr.Row():
            with gr.Column(scale=4):
                pass
            with gr.Column(scale=2):
                status_display = gr.HTML(
                    '<div class="status-indicator">'
                    '<span class="status-dot status-connected"></span>'
                    "<span>Checking connection...</span>"
                    "</div>"
                )

        # Chat interface - Gradio 6.x uses messages format (dict with role/content)
        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
            elem_classes=["chat-container"],
            buttons=["copy"],  # Gradio 6.x: replaces show_copy_button
        )

        # Input area
        with gr.Row():
            with gr.Column(scale=5):
                msg_input = gr.Textbox(
                    placeholder="Type your question here...",
                    label="Message",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                )
            with gr.Column(scale=1, min_width=120):
                submit_btn = gr.Button("Send", variant="primary", size="lg")

        # Control buttons
        with gr.Row():
            new_session_btn = gr.Button(
                "🔄 New Conversation",
                elem_classes=["new-session-btn"],
                size="sm",
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False,
            )

        # Event handlers
        def respond(message: str, history: list) -> tuple[str, list]:
            """Handle user message submission."""
            if not message.strip():
                return "", history

            # Add user message to history (Gradio 6.0 messages format)
            history = history + [{"role": "user", "content": message}]

            # Get response from API
            action, response, sources = ui.send_message(message)

            # Format the bot response
            if action == "ERROR":
                bot_message = f"⚠️ {response}"
            else:
                bot_message = format_bot_message(action, response, sources)

            # Add assistant message to history
            history = history + [{"role": "assistant", "content": bot_message}]

            return "", history

        def new_session() -> tuple[list, str]:
            """Start a new conversation."""
            status = ui.reset_session()
            return [], status

        def check_connection() -> str:
            """Check API connection status."""
            connected, status = ui.check_api_health()
            if connected:
                return (
                    '<div class="status-indicator">'
                    '<span class="status-dot status-connected"></span>'
                    f"<span>{status}</span>"
                    "</div>"
                )
            return (
                '<div class="status-indicator">'
                '<span class="status-dot status-error"></span>'
                f"<span>{status}</span>"
                "</div>"
            )

        # Wire up events
        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        new_session_btn.click(
            fn=new_session,
            inputs=[],
            outputs=[chatbot, status_text],
        )

        # Check connection on load
        app.load(fn=check_connection, outputs=[status_display])

    logger.info("Gradio app created", api_url=ui.api_base_url)
    return app


def launch_app(app: gr.Blocks, **kwargs) -> None:
    """Launch the Gradio app with proper theme and CSS for Gradio 6.0."""
    css = getattr(app, "_custom_css", None)
    theme = getattr(app, "_custom_theme", None)

    app.launch(css=css, theme=theme, **kwargs)


# Module-level app for direct usage
def get_app() -> gr.Blocks:
    """Get the Gradio app instance."""
    return create_gradio_app()
