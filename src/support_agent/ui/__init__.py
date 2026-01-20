"""
Gradio UI for the support agent.

Provides a chat interface that communicates with the FastAPI backend.
"""

from .app import create_gradio_app, launch_app

__all__ = ["create_gradio_app", "launch_app"]
