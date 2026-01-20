"""Unit tests for the centralized OpenAI client."""

from unittest.mock import MagicMock, patch

import pytest

from support_agent.clients.openai import OpenAIClient, get_openai_client, reset_client
from tests.helpers import generate_fake_embedding


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton client before each test."""
    reset_client()
    yield
    reset_client()


@pytest.fixture
def mock_openai():
    """Mock the LangFuse-wrapped OpenAI client."""
    with patch(
        "support_agent.clients.openai.get_langfuse_openai_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


class TestOpenAIClientInit:
    """Tests for client initialization."""

    def test_default_parameters(self, mock_openai):
        """Client initializes with default retry parameters."""
        client = OpenAIClient()

        assert client.max_retries == 5
        assert client.base_delay == 1.0
        assert client.max_delay == 60.0

    def test_custom_parameters(self, mock_openai):
        """Client accepts custom retry parameters."""
        client = OpenAIClient(max_retries=3, base_delay=0.5, max_delay=30.0)

        assert client.max_retries == 3
        assert client.base_delay == 0.5
        assert client.max_delay == 30.0


class TestEmbedTexts:
    """Tests for embed_texts method."""

    def test_embeds_single_text(self, mock_openai):
        """Successfully embeds a single text."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=generate_fake_embedding("test"))]
        mock_openai.embeddings.create.return_value = mock_response

        client = OpenAIClient()
        result = client.embed_texts(["test text"])

        assert len(result) == 1
        assert len(result[0]) == 1536
        mock_openai.embeddings.create.assert_called_once()

    def test_embeds_multiple_texts(self, mock_openai):
        """Successfully embeds multiple texts."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=generate_fake_embedding("text1")),
            MagicMock(embedding=generate_fake_embedding("text2")),
            MagicMock(embedding=generate_fake_embedding("text3")),
        ]
        mock_openai.embeddings.create.return_value = mock_response

        client = OpenAIClient()
        result = client.embed_texts(["text1", "text2", "text3"])

        assert len(result) == 3

    def test_retries_on_rate_limit(self, mock_openai):
        """Retries on RateLimitError."""
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_openai.embeddings.create.side_effect = [
            RateLimitError("Rate limit", response=MagicMock(), body={}),
            mock_response,
        ]

        client = OpenAIClient()
        with patch("time.sleep"):
            result = client.embed_texts(["test"])

        assert len(result) == 1
        assert mock_openai.embeddings.create.call_count == 2

    def test_retries_on_connection_error(self, mock_openai):
        """Retries on APIConnectionError."""
        from openai import APIConnectionError

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_openai.embeddings.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            mock_response,
        ]

        client = OpenAIClient()
        with patch("time.sleep"):
            result = client.embed_texts(["test"])

        assert len(result) == 1

    def test_raises_after_max_retries(self, mock_openai):
        """Raises exception after max retries."""
        from openai import RateLimitError

        mock_openai.embeddings.create.side_effect = RateLimitError(
            "Rate limit", response=MagicMock(), body={}
        )

        client = OpenAIClient(max_retries=3)
        with patch("time.sleep"):
            with pytest.raises(RateLimitError):
                client.embed_texts(["test"])

        assert mock_openai.embeddings.create.call_count == 3


class TestEmbedQuery:
    """Tests for embed_query method."""

    def test_returns_single_embedding(self, mock_openai):
        """Returns a single embedding vector."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=generate_fake_embedding("query"))]
        mock_openai.embeddings.create.return_value = mock_response

        client = OpenAIClient()
        result = client.embed_query("test query")

        assert len(result) == 1536
        assert isinstance(result, list)


class TestChatCompletion:
    """Tests for chat_completion method."""

    def test_generates_completion(self, mock_openai):
        """Successfully generates a chat completion."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_openai.chat.completions.create.return_value = mock_response

        client = OpenAIClient()
        result = client.chat_completion([{"role": "user", "content": "Hi"}])

        assert result == "Hello!"
        mock_openai.chat.completions.create.assert_called_once()

    def test_uses_default_model(self, mock_openai):
        """Uses default model from settings."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_openai.chat.completions.create.return_value = mock_response

        client = OpenAIClient()
        client.chat_completion([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        # Should use the model from settings (gpt-4o-mini by default)
        assert "model" in call_kwargs

    def test_accepts_custom_model(self, mock_openai):
        """Accepts custom model parameter."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_openai.chat.completions.create.return_value = mock_response

        client = OpenAIClient()
        client.chat_completion(
            [{"role": "user", "content": "Hi"}],
            model="gpt-4o",
        )

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    def test_retries_on_error(self, mock_openai):
        """Retries chat completion on transient errors."""
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Success"))]
        mock_openai.chat.completions.create.side_effect = [
            RateLimitError("Rate limit", response=MagicMock(), body={}),
            mock_response,
        ]

        client = OpenAIClient()
        with patch("time.sleep"):
            result = client.chat_completion([{"role": "user", "content": "Hi"}])

        assert result == "Success"
        assert mock_openai.chat.completions.create.call_count == 2


class TestBackoffCalculation:
    """Tests for exponential backoff."""

    def test_initial_delay(self, mock_openai):
        """First attempt has base delay."""
        client = OpenAIClient(base_delay=1.0)
        delay = client._calculate_backoff(0)

        # Base delay + jitter (0-25%)
        assert 1.0 <= delay <= 1.25

    def test_exponential_increase(self, mock_openai):
        """Delay doubles with each attempt."""
        client = OpenAIClient(base_delay=1.0, max_delay=100.0)

        delay_0 = client._calculate_backoff(0)
        delay_1 = client._calculate_backoff(1)
        delay_2 = client._calculate_backoff(2)

        # Each should roughly double (accounting for jitter)
        assert delay_1 > delay_0
        assert delay_2 > delay_1

    def test_respects_max_delay(self, mock_openai):
        """Delay is capped at max_delay."""
        client = OpenAIClient(base_delay=1.0, max_delay=5.0)
        delay = client._calculate_backoff(10)

        # Should not exceed max_delay + jitter
        assert delay <= 5.0 * 1.25


class TestWaitTimeExtraction:
    """Tests for extracting wait time from error messages."""

    def test_extracts_seconds(self, mock_openai):
        """Extracts wait time in seconds."""
        client = OpenAIClient()
        error = Exception("Rate limit exceeded, try again in 30 seconds")
        wait_time = client._extract_wait_time(error)

        assert wait_time == 30.0

    def test_extracts_decimal_seconds(self, mock_openai):
        """Extracts decimal wait time."""
        client = OpenAIClient()
        error = Exception("try again in 1.5 seconds")
        wait_time = client._extract_wait_time(error)

        assert wait_time == 1.5

    def test_returns_none_if_not_found(self, mock_openai):
        """Returns None if no wait time in message."""
        client = OpenAIClient()
        error = Exception("Generic error message")
        wait_time = client._extract_wait_time(error)

        assert wait_time is None


class TestSingleton:
    """Tests for singleton behavior."""

    def test_get_openai_client_returns_same_instance(self, mock_openai):
        """get_openai_client returns the same instance."""
        client1 = get_openai_client()
        client2 = get_openai_client()

        assert client1 is client2

    def test_reset_client_clears_instance(self, mock_openai):
        """reset_client clears the singleton."""
        client1 = get_openai_client()
        reset_client()
        client2 = get_openai_client()

        assert client1 is not client2
