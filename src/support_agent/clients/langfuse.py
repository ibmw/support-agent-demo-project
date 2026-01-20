"""
LangFuse observability client for tracing LLM calls and tracking metrics.

This module provides:
- LangFuse client initialization (reads from environment variables)
- Helper functions for trace management
- Score tracking for evaluations
"""

import os
from functools import cache

from langfuse import Langfuse, get_client, observe
from langfuse.openai import OpenAI as LangfuseOpenAI

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__, component="langfuse")


def _configure_langfuse_env() -> None:
    """
    Configure LangFuse environment variables from settings.

    LangFuse SDK reads from environment variables, so we ensure they're set
    from our Pydantic settings.
    """
    # Only set if not already in environment (allows override)
    if "LANGFUSE_PUBLIC_KEY" not in os.environ:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    if "LANGFUSE_SECRET_KEY" not in os.environ:
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    if "LANGFUSE_HOST" not in os.environ:
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host


@cache
def get_langfuse_client() -> Langfuse:
    """
    Get the LangFuse client singleton.

    Initializes LangFuse with credentials from settings and verifies connection.

    Returns:
        Configured Langfuse client instance

    Raises:
        RuntimeError: If authentication fails
    """
    _configure_langfuse_env()

    client = get_client()

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


def get_langfuse_openai_client() -> LangfuseOpenAI:
    """
    Get an OpenAI client wrapped with LangFuse tracing.

    This client automatically traces all OpenAI API calls to LangFuse.

    Returns:
        LangFuse-wrapped OpenAI client
    """
    _configure_langfuse_env()

    return LangfuseOpenAI(api_key=settings.openai_api_key)


def flush_langfuse() -> None:
    """
    Flush any pending LangFuse events.

    Call this before application shutdown to ensure all traces are sent.
    """
    try:
        client = get_langfuse_client()
        client.flush()
        logger.debug("LangFuse events flushed")
    except Exception as e:
        logger.warning("Failed to flush LangFuse events", error=str(e))


def shutdown_langfuse() -> None:
    """
    Shutdown the LangFuse client gracefully.

    Flushes pending events and releases resources.
    """
    try:
        client = get_langfuse_client()
        client.shutdown()
        logger.info("LangFuse client shutdown complete")
    except Exception as e:
        logger.warning("Failed to shutdown LangFuse client", error=str(e))


# Export the observe decorator for use in other modules
__all__ = [
    "get_langfuse_client",
    "get_langfuse_openai_client",
    "flush_langfuse",
    "shutdown_langfuse",
    "observe",
]
