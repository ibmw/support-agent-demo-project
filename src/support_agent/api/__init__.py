"""
FastAPI backend for the support agent.

Provides REST endpoints for chat and session management.
"""

from .main import create_app

__all__ = ["create_app"]
