"""Unit tests for CrewAI RAG tools."""

from unittest.mock import MagicMock, patch

import pytest

from support_agent.rag import tools
from support_agent.rag.retriever import KnowledgeBaseRetriever


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = MagicMock(spec=KnowledgeBaseRetriever)
    return retriever


@pytest.fixture(autouse=True)
def reset_retriever():
    """Reset the module-level retriever before each test."""
    tools._retriever = None
    yield
    tools._retriever = None


class TestRetrieverManagement:
    """Tests for retriever instance management."""

    def test_get_retriever_creates_instance(self):
        """get_retriever creates a new instance if none exists."""
        with patch("support_agent.rag.tools.KnowledgeBaseRetriever") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            result = tools.get_retriever()

            mock_cls.assert_called_once()
            assert result == mock_instance

    def test_get_retriever_returns_existing(self):
        """get_retriever returns existing instance."""
        mock_instance = MagicMock()
        tools._retriever = mock_instance

        result = tools.get_retriever()

        assert result == mock_instance

    def test_set_retriever_replaces_instance(self):
        """set_retriever replaces the existing instance."""
        old_retriever = MagicMock()
        new_retriever = MagicMock()
        tools._retriever = old_retriever

        tools.set_retriever(new_retriever)

        assert tools._retriever == new_retriever
        assert tools.get_retriever() == new_retriever


class TestSearchKnowledgeBase:
    """Tests for search_knowledge_base tool."""

    def test_calls_retrieve_with_context(self, mock_retriever):
        """Tool calls retriever.retrieve_with_context."""
        tools.set_retriever(mock_retriever)
        mock_retriever.retrieve_with_context.return_value = "Context from KB"

        # Call the underlying function (not the decorated tool)
        result = tools.search_knowledge_base.func("how to reset password")

        mock_retriever.retrieve_with_context.assert_called_once_with(
            "how to reset password"
        )
        assert result == "Context from KB"

    def test_returns_formatted_context(self, mock_retriever):
        """Tool returns formatted context string."""
        tools.set_retriever(mock_retriever)
        context = "[Source 1: Password Reset]\nStep 1: Click forgot password"
        mock_retriever.retrieve_with_context.return_value = context

        result = tools.search_knowledge_base.func("password help")

        assert "[Source 1:" in result
        assert "Step 1:" in result

    def test_handles_no_results(self, mock_retriever):
        """Tool handles no results gracefully."""
        tools.set_retriever(mock_retriever)
        mock_retriever.retrieve_with_context.return_value = (
            "No relevant information found in the knowledge base."
        )

        result = tools.search_knowledge_base.func("obscure topic")

        assert "No relevant information found" in result


class TestGetArticleSources:
    """Tests for get_article_sources tool."""

    def test_returns_comma_separated_sources(self, mock_retriever):
        """Tool returns sources as comma-separated string."""
        tools.set_retriever(mock_retriever)
        mock_retriever.get_sources.return_value = [
            "Getting Started",
            "FAQ > Account Setup",
            "Troubleshooting",
        ]

        result = tools.get_article_sources.func("account help")

        assert "Getting Started" in result
        assert "FAQ > Account Setup" in result
        assert "Troubleshooting" in result
        assert ", " in result

    def test_returns_message_for_no_sources(self, mock_retriever):
        """Tool returns message when no sources found."""
        tools.set_retriever(mock_retriever)
        mock_retriever.get_sources.return_value = []

        result = tools.get_article_sources.func("unknown topic")

        assert "No relevant articles found" in result


class TestToolDecorators:
    """Tests for tool decorator metadata."""

    def test_search_tool_has_name(self):
        """search_knowledge_base has correct tool name."""
        assert tools.search_knowledge_base.name == "search_knowledge_base"

    def test_search_tool_has_description(self):
        """search_knowledge_base has a description."""
        assert tools.search_knowledge_base.description
        assert "knowledge base" in tools.search_knowledge_base.description.lower()

    def test_sources_tool_has_name(self):
        """get_article_sources has correct tool name."""
        assert tools.get_article_sources.name == "get_article_sources"

    def test_sources_tool_has_description(self):
        """get_article_sources has a description."""
        assert tools.get_article_sources.description
        assert "sources" in tools.get_article_sources.description.lower()
