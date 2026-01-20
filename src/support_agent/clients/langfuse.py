"""
LangFuse observability client for tracing LLM calls and tracking metrics.

This module provides:
- LangFuse client initialization (reads from environment variables)
- Helper functions for trace management
- Score tracking for evaluations
- Automatic disabling for tests via LANGFUSE_ENABLED=false
"""

import os
from functools import cache, wraps
from typing import Any

from openai import OpenAI

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__, component="langfuse")


def _is_langfuse_enabled() -> bool:
    """Check if LangFuse is enabled."""
    return settings.langfuse_enabled


def _configure_langfuse_env() -> None:
    """
    Configure LangFuse environment variables from settings.

    LangFuse SDK reads from environment variables, so we ensure they're set
    from our Pydantic settings.
    """
    if not _is_langfuse_enabled():
        return

    # Only set if not already in environment (allows override)
    if "LANGFUSE_PUBLIC_KEY" not in os.environ:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    if "LANGFUSE_SECRET_KEY" not in os.environ:
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    if "LANGFUSE_HOST" not in os.environ:
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host


def _get_langfuse_imports():
    """Lazy import LangFuse to avoid initialization when disabled."""
    from langfuse import Langfuse, get_client
    from langfuse import observe as langfuse_observe
    from langfuse.openai import OpenAI as LangfuseOpenAI

    return Langfuse, get_client, langfuse_observe, LangfuseOpenAI


@cache
def get_langfuse_client():
    """
    Get the LangFuse client singleton.

    Initializes LangFuse with credentials from settings and verifies connection.
    Returns None if LangFuse is disabled.

    Returns:
        Configured Langfuse client instance, or None if disabled

    Raises:
        RuntimeError: If authentication fails (only when enabled)
    """
    if not _is_langfuse_enabled():
        logger.debug("LangFuse is disabled")
        return None

    _configure_langfuse_env()

    _, get_client_fn, _, _ = _get_langfuse_imports()
    client = get_client_fn()

    # Verify connection
    if client.auth_check():
        logger.info(
            "LangFuse client initialized",
            host=settings.langfuse_host,
        )
    else:
        logger.error(
            "LangFuse authentication failed",
            host=settings.langfuse_host,
        )
        raise RuntimeError(
            "LangFuse authentication failed. Check your API keys and host."
        )

    return client


def get_langfuse_openai_client() -> OpenAI:
    """
    Get an OpenAI client, optionally wrapped with LangFuse tracing.

    When LangFuse is enabled, returns a LangFuse-wrapped client that
    automatically traces all OpenAI API calls.

    When LangFuse is disabled (e.g., in tests), returns a regular OpenAI client.

    Returns:
        OpenAI client (with or without LangFuse tracing)
    """
    if not _is_langfuse_enabled():
        logger.debug("Using standard OpenAI client (LangFuse disabled)")
        return OpenAI(api_key=settings.openai_api_key)

    _configure_langfuse_env()

    _, _, _, LangfuseOpenAI = _get_langfuse_imports()
    return LangfuseOpenAI(api_key=settings.openai_api_key)


def flush_langfuse() -> None:
    """
    Flush any pending LangFuse events.

    Call this before application shutdown to ensure all traces are sent.
    No-op if LangFuse is disabled.
    """
    if not _is_langfuse_enabled():
        return

    try:
        client = get_langfuse_client()
        if client:
            client.flush()
            logger.debug("LangFuse events flushed")
    except Exception as e:
        logger.warning("Failed to flush LangFuse events", error=str(e))


def shutdown_langfuse() -> None:
    """
    Shutdown the LangFuse client gracefully.

    Flushes pending events and releases resources.
    No-op if LangFuse is disabled.
    """
    if not _is_langfuse_enabled():
        return

    try:
        client = get_langfuse_client()
        if client:
            client.shutdown()
            logger.info("LangFuse client shutdown complete")
    except Exception as e:
        logger.warning("Failed to shutdown LangFuse client", error=str(e))


def observe(name: str | None = None, **kwargs: Any):
    """
    Decorator to trace function execution with LangFuse.

    When LangFuse is disabled, returns the function unchanged (no-op decorator).

    Args:
        name: Optional name for the trace/span
        **kwargs: Additional arguments passed to langfuse.observe

    Returns:
        Decorated function with tracing (or original function if disabled)
    """
    if not _is_langfuse_enabled():
        # Return a no-op decorator when disabled
        def noop_decorator(func):
            return func

        return noop_decorator

    # Import and use the real LangFuse observe decorator
    _, _, langfuse_observe, _ = _get_langfuse_imports()
    return langfuse_observe(name=name, **kwargs)


# Export the observe decorator for use in other modules
__all__ = [
    "get_langfuse_client",
    "get_langfuse_openai_client",
    "flush_langfuse",
    "shutdown_langfuse",
    "observe",
]
