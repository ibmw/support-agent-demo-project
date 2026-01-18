"""Unit tests for the knowledge base retriever."""

from unittest.mock import MagicMock, patch

import pytest

from support_agent.rag.retriever import KnowledgeBaseRetriever, RetrievalResult
from tests.helpers import generate_fake_embedding


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_basic_creation(self):
        """RetrievalResult is created with required fields."""
        result = RetrievalResult(
            text="Some content",
            article_title="Getting Started",
        )
        assert result.text == "Some content"
        assert result.article_title == "Getting Started"
        assert result.section_title is None
        assert result.score == 0.0
        assert result.metadata == {}

    def test_with_all_fields(self):
        """RetrievalResult accepts all fields."""
        result = RetrievalResult(
            text="Content here",
            article_title="Guide",
            section_title="Setup",
            score=0.95,
            metadata={"is_faq": True},
        )
        assert result.section_title == "Setup"
        assert result.score == 0.95
        assert result.metadata["is_faq"] is True

    def test_source_with_section(self):
        """Source property includes section when present."""
        result = RetrievalResult(
            text="Content",
            article_title="Integration Guide",
            section_title="OAuth Setup",
        )
        assert result.source == "Integration Guide > OAuth Setup"

    def test_source_without_section(self):
        """Source property returns just article title when no section."""
        result = RetrievalResult(
            text="Content",
            article_title="FAQ",
        )
        assert result.source == "FAQ"


@pytest.fixture
def mock_openai_client():
    """Mock the centralized OpenAI client."""
    with patch("support_agent.rag.retriever.get_openai_client") as mock_get:
        mock_client = MagicMock()

        def embed_query(text):
            return generate_fake_embedding(text)

        mock_client.embed_query.side_effect = embed_query
        mock_get.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client."""
    with patch("support_agent.rag.retriever.chromadb.PersistentClient") as mock_cls:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 10
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_cls.return_value = mock_client
        yield mock_client, mock_collection


@pytest.fixture
def retriever(mock_openai_client, mock_chroma_client):
    """Create retriever with mocked dependencies."""
    with patch("support_agent.rag.retriever.settings") as mock_settings:
        mock_settings.chroma_persist_dir = "/tmp/test_db"
        mock_settings.chroma_collection_name = "test_collection"
        mock_settings.retrieval_top_k = 5

        retriever = KnowledgeBaseRetriever()
        return retriever


class TestKnowledgeBaseRetrieverInit:
    """Tests for retriever initialization."""

    def test_default_top_k_from_settings(self, mock_openai_client, mock_chroma_client):
        """Uses top_k from settings by default."""
        with patch("support_agent.rag.retriever.settings") as mock_settings:
            mock_settings.chroma_persist_dir = "/tmp/test"
            mock_settings.chroma_collection_name = "test"
            mock_settings.retrieval_top_k = 10

            retriever = KnowledgeBaseRetriever()

            assert retriever.top_k == 10

    def test_custom_top_k(self, mock_openai_client, mock_chroma_client):
        """Accepts custom top_k parameter."""
        with patch("support_agent.rag.retriever.settings") as mock_settings:
            mock_settings.chroma_persist_dir = "/tmp/test"
            mock_settings.chroma_collection_name = "test"
            mock_settings.retrieval_top_k = 5

            retriever = KnowledgeBaseRetriever(top_k=3)

            assert retriever.top_k == 3

    def test_score_threshold_default_none(self, mock_openai_client, mock_chroma_client):
        """Score threshold is None by default."""
        with patch("support_agent.rag.retriever.settings") as mock_settings:
            mock_settings.chroma_persist_dir = "/tmp/test"
            mock_settings.chroma_collection_name = "test"
            mock_settings.retrieval_top_k = 5

            retriever = KnowledgeBaseRetriever()

            assert retriever.score_threshold is None

    def test_custom_score_threshold(self, mock_openai_client, mock_chroma_client):
        """Accepts custom score threshold."""
        with patch("support_agent.rag.retriever.settings") as mock_settings:
            mock_settings.chroma_persist_dir = "/tmp/test"
            mock_settings.chroma_collection_name = "test"
            mock_settings.retrieval_top_k = 5

            retriever = KnowledgeBaseRetriever(score_threshold=0.7)

            assert retriever.score_threshold == 0.7


class TestRetrieve:
    """Tests for the retrieve method."""

    def test_returns_empty_for_empty_collection(
        self, retriever, mock_chroma_client, mock_openai_client
    ):
        """Returns empty list when collection is empty."""
        _, mock_collection = mock_chroma_client
        mock_collection.count.return_value = 0

        results = retriever.retrieve("test query")

        assert results == []
        mock_openai_client.embed_query.assert_not_called()

    def test_queries_with_embedding(
        self, retriever, mock_chroma_client, mock_openai_client
    ):
        """Creates embedding and queries collection."""
        _, mock_collection = mock_chroma_client
        mock_collection.query.return_value = {
            "documents": [["Content 1", "Content 2"]],
            "metadatas": [
                [
                    {"article_title": "Article 1", "section": "Setup"},
                    {"article_title": "Article 2"},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        results = retriever.retrieve("how to setup")

        mock_openai_client.embed_query.assert_called_once_with("how to setup")
        mock_collection.query.assert_called_once()
        assert len(results) == 2

    def test_parses_results_correctly(self, retriever, mock_chroma_client):
        """Results are parsed into RetrievalResult objects."""
        _, mock_collection = mock_chroma_client
        mock_collection.query.return_value = {
            "documents": [["Step 1: Do this"]],
            "metadatas": [[{"article_title": "Guide", "section": "Steps"}]],
            "distances": [[0.05]],
        }

        results = retriever.retrieve("how to")

        assert len(results) == 1
        result = results[0]
        assert result.text == "Step 1: Do this"
        assert result.article_title == "Guide"
        assert result.section_title == "Steps"
        assert result.score == pytest.approx(0.95, abs=0.01)

    def test_calculates_similarity_score(self, retriever, mock_chroma_client):
        """Distance is converted to similarity score."""
        _, mock_collection = mock_chroma_client
        mock_collection.query.return_value = {
            "documents": [["Content"]],
            "metadatas": [[{"article_title": "Article"}]],
            "distances": [[0.3]],  # Distance of 0.3
        }

        results = retriever.retrieve("query")

        # Similarity = 1 - distance = 1 - 0.3 = 0.7
        assert results[0].score == pytest.approx(0.7, abs=0.01)

    def test_applies_score_threshold(self, mock_openai_client, mock_chroma_client):
        """Filters out results below score threshold."""
        with patch("support_agent.rag.retriever.settings") as mock_settings:
            mock_settings.chroma_persist_dir = "/tmp/test"
            mock_settings.chroma_collection_name = "test"
            mock_settings.retrieval_top_k = 5

            retriever = KnowledgeBaseRetriever(score_threshold=0.8)

            _, mock_collection = mock_chroma_client
            mock_collection.query.return_value = {
                "documents": [["High score", "Low score"]],
                "metadatas": [[{"article_title": "A1"}, {"article_title": "A2"}]],
                "distances": [[0.1, 0.5]],  # Scores: 0.9, 0.5
            }

            results = retriever.retrieve("query")

            # Only the high score result (0.9 >= 0.8) should be included
            assert len(results) == 1
            assert results[0].score == pytest.approx(0.9, abs=0.01)

    def test_handles_missing_metadata(self, retriever, mock_chroma_client):
        """Handles results with missing metadata gracefully."""
        _, mock_collection = mock_chroma_client
        mock_collection.query.return_value = {
            "documents": [["Content"]],
            "metadatas": [[{}]],  # Empty metadata
            "distances": [[0.1]],
        }

        results = retriever.retrieve("query")

        assert len(results) == 1
        assert results[0].article_title == "Unknown"
        assert results[0].section_title is None


class TestRetrieveWithContext:
    """Tests for retrieve_with_context method."""

    def test_formats_context_string(self, retriever, mock_chroma_client):
        """Returns formatted context for LLM."""
        _, mock_collection = mock_chroma_client
        mock_collection.query.return_value = {
            "documents": [["First chunk", "Second chunk"]],
            "metadatas": [
                [
                    {"article_title": "Guide A", "section": "Intro"},
                    {"article_title": "Guide B"},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        context = retriever.retrieve_with_context("query")

        assert "[Source 1: Guide A > Intro]" in context
        assert "First chunk" in context
        assert "[Source 2: Guide B]" in context
        assert "Second chunk" in context
        assert "---" in context  # Separator

    def test_returns_message_for_no_results(self, retriever, mock_chroma_client):
        """Returns appropriate message when no results found."""
        _, mock_collection = mock_chroma_client
        mock_collection.count.return_value = 0

        context = retriever.retrieve_with_context("query")

        assert "No relevant information found" in context


class TestGetSources:
    """Tests for get_sources method."""

    def test_returns_source_list(self, retriever, mock_chroma_client):
        """Returns list of source strings."""
        _, mock_collection = mock_chroma_client
        mock_collection.query.return_value = {
            "documents": [["A", "B", "C"]],
            "metadatas": [
                [
                    {"article_title": "Article 1", "section": "Section 1"},
                    {"article_title": "Article 2"},
                    {"article_title": "Article 3", "section": "Section 3"},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }

        sources = retriever.get_sources("query")

        assert len(sources) == 3
        assert "Article 1 > Section 1" in sources
        assert "Article 2" in sources
        assert "Article 3 > Section 3" in sources

    def test_returns_empty_for_no_results(self, retriever, mock_chroma_client):
        """Returns empty list when no results."""
        _, mock_collection = mock_chroma_client
        mock_collection.count.return_value = 0

        sources = retriever.get_sources("query")

        assert sources == []
