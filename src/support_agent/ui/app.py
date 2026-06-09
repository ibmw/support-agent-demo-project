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
    --bg-gradient-start: #0b0f19;
    --bg-gradient-end: #1a153b;
    --surface-color: rgba(22, 30, 49, 0.7);
    --surface-border: rgba(99, 102, 241, 0.15);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-glow: rgba(99, 102, 241, 0.25);
}

.gradio-container {
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 100%) !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 1.5rem 1rem;
    background: transparent;
    border-bottom: none;
    margin-bottom: 1.5rem;
}

.main-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 40%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
    filter: drop-shadow(0 0 20px rgba(165, 180, 252, 0.2));
}

.main-header p {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 400;
}

/* Header Controls row */
.header-controls {
    align-items: center !important;
    margin-bottom: 1rem !important;
    gap: 15px !important;
}

/* Chat container styles */
.chat-container {
    background: var(--surface-color) !important;
    border: 1px solid var(--surface-border) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 30px var(--accent-glow) !important;
    padding: 10px !important;
}

/* Override Gradio Message bubbles */
.chat-container .message {
    padding: 14px 18px !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    margin-bottom: 12px !important;
    max-width: 80% !important;
    width: fit-content !important;
    border-radius: 16px !important;
}

/* User Message styling */
.chat-container .message.user {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border-radius: 18px 18px 0px 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    margin-left: auto !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25) !important;
}

/* Assistant Message styling */
.chat-container .message.assistant,
.chat-container .message.bot,
.chat-container .message.pending {
    background: rgba(30, 41, 59, 0.65) !important;
    color: #f1f5f9 !important;
    border-radius: 18px 18px 18px 0px !important;
    border: 1px solid rgba(99, 102, 241, 0.18) !important;
    margin-right: auto !important;
}

/* Input Area overrides */
.textbox textarea {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    padding: 14px 18px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-family: inherit !important;
}

.textbox textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.35) !important;
    background: rgba(15, 23, 42, 0.75) !important;
    outline: none !important;
}

/* Connection Status styling */
.status-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 8px 16px;
    background: rgba(22, 30, 49, 0.5);
    border: 1px solid var(--surface-border);
    border-radius: 12px;
    font-size: 0.85rem;
    color: var(--text-primary);
    height: 38px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

.status-connected {
    background: #22c55e;
    box-shadow: 0 0 10px #22c55e;
}

.status-error {
    background: #ef4444;
    box-shadow: 0 0 10px #ef4444;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* New session button in control header */
.new-session-btn {
    background: rgba(99, 102, 241, 0.15) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    border-radius: 12px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    height: 38px !important;
    width: fit-content !important;
    min-width: 180px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
}

.new-session-btn:hover {
    background: rgba(99, 102, 241, 0.3) !important;
    border-color: #818cf8 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.3) !important;
}

/* Custom premium action badges */
.action-badge-container {
    margin-bottom: 10px;
}

.action-badge {
    display: inline-flex;
    align-items: center;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 6px;
}

.action-badge.action-close {
    background: rgba(34, 197, 94, 0.15) !important;
    color: #4ade80 !important;
    border: 1px solid rgba(34, 197, 94, 0.3) !important;
}

.action-badge.action-handover {
    background: rgba(245, 158, 11, 0.15) !important;
    color: #fbbf24 !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
}

.action-badge.action-wait {
    background: rgba(59, 130, 246, 0.15) !important;
    color: #60a5fa !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
}

/* Custom sources display */
.sources-section {
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px dashed rgba(255, 255, 255, 0.1);
}

.sources-title {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-bottom: 6px;
    font-weight: 600;
}

.sources-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.source-tag {
    font-size: 0.75rem;
    background: rgba(99, 102, 241, 0.12);
    color: #c7d2fe;
    border: 1px solid rgba(99, 102, 241, 0.25);
    padding: 3px 10px;
    border-radius: 12px;
    font-family: inherit;
    transition: all 0.2s ease;
}

.source-tag:hover {
    background: rgba(99, 102, 241, 0.22);
    border-color: rgba(99, 102, 241, 0.5);
    color: #ffffff;
    transform: translateY(-1px);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.25);
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
    """Format the bot response with custom HTML classes for premium styling."""
    label, color, bg_color = ACTION_STYLES.get(
        action, ("Unknown", "#6b7280", "#f3f4f6")
    )

    # Action badge wrapped in styled HTML span
    badge_html = f'<div class="action-badge-container"><span class="action-badge action-{action.lower()}">{label}</span></div>'
    parts = [f'{badge_html}\n\n{message}']

    # Sources formatted as individual interactive tags
    if sources:
        sources_html = "".join(f'<span class="source-tag">{s}</span>' for s in sources)
        parts.append(f'\n\n<div class="sources-section"><div class="sources-title">Sources:</div><div class="sources-list">{sources_html}</div></div>')

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
    Create the Gradio chat interface with a HuggingChat-style layout.

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

        # Left Sidebar (incorporating stats and controls)
        with gr.Sidebar(label="Support Menu", open=True) as sidebar:
            gr.HTML(
                """
                <div class="sidebar-header">
                    <h2>🤖 Support Agent</h2>
                </div>
                """
            )

            new_session_btn = gr.Button(
                "🔄 New Conversation",
                elem_classes=["new-session-btn"],
                size="sm",
            )

            gr.HTML("<div class='sidebar-divider'></div>")

            gr.HTML(
                """
                <div class="sidebar-info">
                    <h3>📚 Knowledge Base</h3>
                    <p>Indexed chunks: <b>1031</b></p>
                    <p>Powered by ChromaDB</p>
                </div>
                """
            )

            gr.HTML("<div class='sidebar-divider'></div>")

            gr.HTML(
                """
                <div class="sidebar-footer">
                    <a href="/docs" target="_blank" class="docs-link">📖 API Documentation</a>
                </div>
                """
            )

            status_display = gr.HTML(
                '<div class="status-indicator">'
                '<span class="status-dot status-connected"></span>'
                "<span>Checking connection...</span>"
                "</div>"
            )

        # Right Main Content area
        with gr.Column(elem_classes=["main-content"]):
            # Welcome banner
            gr.HTML(
                """
                <div class="welcome-container">
                    <h1>🤖 Support Agent</h1>
                    <p>Ask anything about setup, integrations, billing, and troubleshooting.</p>
                </div>
                """
            )

            # Chatbot component
            chatbot = gr.Chatbot(
                height=500,
                elem_classes=["chat-container"],
                buttons=["copy"],  # Gradio 6.x: replaces show_copy_button
            )

            # Quick starter questions
            with gr.Row(elem_classes=["starter-questions"]):
                q1 = gr.Button("How do I reset my password?", size="sm", elem_classes=["starter-btn"])
                q2 = gr.Button("How do I integrate with Shopify?", size="sm", elem_classes=["starter-btn"])
                q3 = gr.Button("What is the refund policy?", size="sm", elem_classes=["starter-btn"])

            # Input area
            with gr.Row(elem_classes=["input-row"]):
                with gr.Column(scale=5):
                    msg_input = gr.Textbox(
                        placeholder="Ask anything...",
                        label="Message",
                        lines=1,
                        max_lines=5,
                        show_label=False,
                        elem_classes=["textbox"]
                    )
                with gr.Column(scale=1, min_width=100):
                    submit_btn = gr.Button("Send", variant="primary", size="lg", elem_classes=["submit-btn"])

            # Disclaimer
            gr.HTML(
                """
                <div class="disclaimer-text">
                    * Support Agent • Responses are generated using RAG over Help Center articles.
                </div>
                """
            )

        # Hidden status text for state management
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

        def load_starter(text: str) -> str:
            """Return the starter question text."""
            return text

        # Wire up starter buttons
        for btn in [q1, q2, q3]:
            btn.click(
                fn=load_starter,
                inputs=[btn],
                outputs=[msg_input],
            ).then(
                fn=respond,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
            )

        # Wire up inputs
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
