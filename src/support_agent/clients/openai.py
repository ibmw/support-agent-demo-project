"""
Centralized OpenAI client with retry logic and structured logging.

This module provides a singleton OpenAI client that handles:
- Automatic retries with exponential backoff
- Rate limit handling with wait time extraction
- Structured logging for all API calls
- Consistent error handling across the application
"""

import random
import re
import time
from typing import Any

from openai import APIConnectionError, OpenAI, RateLimitError

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__, component="openai_client")

# Singleton instance
_client: "OpenAIClient | None" = None


class OpenAIClient:
    """
    OpenAI client wrapper with retry logic and logging.

    Provides methods for embeddings and chat completions with automatic
    retry on transient errors (rate limits, connection issues).
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize the OpenAI client.

        Args:
            max_retries: Maximum retry attempts for failed requests
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds between retries
        """
        self._client = OpenAI(api_key=settings.openai_api_key)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        logger.info(
            "OpenAI client initialized",
            max_retries=max_retries,
            embedding_model=settings.embedding_model,
            llm_model=settings.llm_model,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RateLimitError: If rate limits exceeded after all retries
            APIConnectionError: If connection errors persist after all retries
        """
        return self._retry_with_backoff(
            operation="embed_texts",
            func=self._embed_texts_impl,
            texts=texts,
        )

    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embed_texts([text])[0]

    def _embed_texts_impl(self, texts: list[str]) -> list[list[float]]:
        """Internal implementation of embed_texts."""
        response = self._client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to settings.llm_model)
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to the API

        Returns:
            Generated response text

        Raises:
            RateLimitError: If rate limits exceeded after all retries
            APIConnectionError: If connection errors persist after all retries
        """
        return self._retry_with_backoff(
            operation="chat_completion",
            func=self._chat_completion_impl,
            messages=messages,
            model=model or settings.llm_model,
            temperature=temperature,
            **kwargs,
        )

    def _chat_completion_impl(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Internal implementation of chat_completion."""
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def _retry_with_backoff(
        self,
        operation: str,
        func: callable,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function with retry logic and exponential backoff.

        Args:
            operation: Name of the operation (for logging)
            func: Function to execute
            **kwargs: Arguments to pass to the function

        Returns:
            Result of the function

        Raises:
            RateLimitError: If rate limits exceeded after all retries
            APIConnectionError: If connection errors persist after all retries
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                log = logger.bind(operation=operation, attempt=attempt + 1)

                if attempt > 0:
                    log.debug("Retrying operation")

                result = func(**kwargs)

                # Log success
                log.debug(
                    "Operation completed",
                    **self._get_operation_metadata(operation, kwargs),
                )
                return result

            except (RateLimitError, APIConnectionError) as e:
                last_exception = e
                log = logger.bind(
                    operation=operation,
                    attempt=attempt + 1,
                    error_type=type(e).__name__,
                )

                if attempt == self.max_retries - 1:
                    log.error(
                        "Operation failed after max retries",
                        error=str(e),
                    )
                    raise

                # Calculate wait time
                wait_time = self._extract_wait_time(e) or self._calculate_backoff(
                    attempt
                )

                log.warning(
                    "Operation failed, retrying",
                    error=str(e),
                    wait_seconds=round(wait_time, 2),
                )

                time.sleep(wait_time)

        # Should not reach here, but just in case
        raise last_exception or APIConnectionError(
            request=None, message="Max retries exceeded"
        )

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        jitter = random.uniform(0, delay * 0.25)
        return delay + jitter

    def _extract_wait_time(self, error: Exception) -> float | None:
        """
        Extract wait time from rate limit error message if available.

        Args:
            error: The exception

        Returns:
            Wait time in seconds, or None if not found
        """
        error_msg = str(error)
        match = re.search(
            r"try again in (\d+(?:\.\d+)?)\s*(?:s|second|seconds)",
            error_msg,
            re.IGNORECASE,
        )
        if match:
            return float(match.group(1))
        return None

    def _get_operation_metadata(
        self, operation: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Get metadata for logging based on operation type."""
        if operation == "embed_texts":
            texts = kwargs.get("texts", [])
            return {
                "text_count": len(texts),
                "total_chars": sum(len(t) for t in texts),
            }
        elif operation == "chat_completion":
            messages = kwargs.get("messages", [])
            return {
                "message_count": len(messages),
                "model": kwargs.get("model", settings.llm_model),
            }
        return {}


def get_openai_client() -> OpenAIClient:
    """
    Get the singleton OpenAI client instance.

    Returns:
        Configured OpenAIClient instance
    """
    global _client
    if _client is None:
        _client = OpenAIClient()
    return _client


def reset_client() -> None:
    """Reset the singleton client (useful for testing)."""
    global _client
    _client = None
