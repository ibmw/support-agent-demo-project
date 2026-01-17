"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from .config import settings


def setup_logging() -> None:
    """
    Configure structlog with appropriate processors for dev/prod.

    Call this once at application startup.
    """
    # Shared processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.log_format == "json":
        # Production: JSON output for log aggregation
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Colored console output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback,
            ),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            _get_log_level(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging to use structlog
    _configure_stdlib_logging()


def _get_log_level(level_name: str) -> int:
    """Convert log level name to logging constant."""
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(level_name.upper(), logging.INFO)


def _configure_stdlib_logging() -> None:
    """Configure stdlib logging to route through structlog."""
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(_get_log_level(settings.log_level))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add structlog handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True)
            if settings.log_format == "console"
            else structlog.processors.JSONRenderer(),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
            ],
        )
    )
    root_logger.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(
    name: str | None = None, **initial_context: Any
) -> structlog.BoundLogger:
    """
    Get a logger instance with optional initial context.

    Args:
        name: Logger name (typically __name__)
        **initial_context: Initial context to bind to the logger

    Returns:
        Configured structlog logger

    Example:
        logger = get_logger(__name__, component="indexer")
        logger.info("Starting indexing", article_count=200)
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_context(**context: Any) -> None:
    """
    Bind context variables that will be included in all subsequent log messages.

    Useful for request-scoped context like session_id, request_id.

    Args:
        **context: Key-value pairs to bind

    Example:
        bind_context(session_id="abc123", request_id="req-456")
        logger.info("Processing request")  # Will include session_id and request_id
    """
    structlog.contextvars.bind_contextvars(**context)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def unbind_context(*keys: str) -> None:
    """
    Remove specific keys from the bound context.

    Args:
        *keys: Keys to remove from context
    """
    structlog.contextvars.unbind_contextvars(*keys)
