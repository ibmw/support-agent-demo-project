"""Custom exceptions for the Support Agent application."""


class SupportAgentError(Exception):
    """Base exception for all Support Agent errors."""

    pass


# Indexing exceptions
class IndexingError(SupportAgentError):
    """Error during article indexing."""

    pass


class ParsingError(IndexingError):
    """Error parsing article content."""

    pass


# RAG exceptions
class RetrievalError(SupportAgentError):
    """Error during document retrieval."""

    pass


class EmbeddingError(SupportAgentError):
    """Error generating embeddings."""

    pass


# Agent exceptions
class AgentError(SupportAgentError):
    """Error in agent execution."""

    pass


class InvalidResponseError(AgentError):
    """Agent returned an invalid or unparseable response."""

    pass


# Configuration exceptions
class ConfigurationError(SupportAgentError):
    """Error in application configuration."""

    pass


# Memory exceptions
class SessionNotFoundError(SupportAgentError):
    """Conversation session not found."""

    pass
