"""Integration tests for the ChromaDB indexer."""

from unittest.mock import MagicMock, patch

import pytest

from support_agent.indexing.chunker import Chunk
from support_agent.indexing.indexer import KnowledgeBaseIndexer
from tests.helpers import generate_fake_embedding


@pytest.fixture
def mock_openai_client():
    """Mock the centralized OpenAI client."""
    with patch("support_agent.indexing.indexer.get_openai_client") as mock_get:
        mock_client = MagicMock()

        def embed_texts(texts):
            return [generate_fake_embedding(text) for text in texts]

        mock_client.embed_texts.side_effect = embed_texts
        mock_get.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_chroma_client(tmp_path):
    """Mock ChromaDB client with temp directory."""
    with patch("support_agent.indexing.indexer.chromadb.PersistentClient") as mock_cls:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection
        mock_cls.return_value = mock_client
        yield mock_client, mock_collection


@pytest.fixture
def indexer(mock_openai_client, mock_chroma_client):
    """Create indexer with mocked dependencies."""
    with patch("support_agent.indexing.indexer.settings") as mock_settings:
        mock_settings.chroma_persist_dir = "/tmp/test_db"
        mock_settings.chroma_collection_name = "test_collection"

        indexer = KnowledgeBaseIndexer()
        return indexer


class TestKnowledgeBaseIndexerInit:
    """Tests for indexer initialization."""

    def test_uses_centralized_client(self, mock_openai_client, mock_chroma_client):
        """Indexer uses the centralized OpenAI client."""
        with patch("support_agent.indexing.indexer.settings") as mock_settings:
            mock_settings.chroma_persist_dir = "/tmp/test"
            mock_settings.chroma_collection_name = "test"

            indexer = KnowledgeBaseIndexer()

            assert indexer.openai_client is mock_openai_client

    def test_creates_chroma_collection(self, mock_openai_client, mock_chroma_client):
        """Indexer creates/gets ChromaDB collection."""
        mock_client, mock_collection = mock_chroma_client

        with patch("support_agent.indexing.indexer.settings") as mock_settings:
            mock_settings.chroma_persist_dir = "/tmp/test"
            mock_settings.chroma_collection_name = "my_collection"

            _indexer = KnowledgeBaseIndexer()  # noqa: F841

            mock_client.get_or_create_collection.assert_called_once()
            call_args = mock_client.get_or_create_collection.call_args
            assert call_args.kwargs["name"] == "my_collection"
            assert call_args.kwargs["metadata"] == {"hnsw:space": "cosine"}


class TestIndexChunks:
    """Tests for indexing chunks into ChromaDB."""

    def test_indexes_single_chunk(self, indexer, mock_chroma_client, sample_chunks):
        """Successfully indexes a single chunk."""
        _, mock_collection = mock_chroma_client
        chunk = sample_chunks[0]

        indexer.index_chunks([chunk])

        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        assert call_args.kwargs["ids"] == [chunk.id]
        assert call_args.kwargs["documents"] == [chunk.text]

    def test_indexes_multiple_chunks(self, indexer, mock_chroma_client, sample_chunks):
        """Successfully indexes multiple chunks."""
        _, mock_collection = mock_chroma_client

        indexer.index_chunks(sample_chunks)

        mock_collection.upsert.assert_called()
        call_args = mock_collection.upsert.call_args
        assert len(call_args.kwargs["ids"]) == len(sample_chunks)

    def test_indexes_in_batches(self, indexer, mock_chroma_client):
        """Large chunk lists are indexed in batches."""
        _, mock_collection = mock_chroma_client

        # Create 150 chunks to test batching (default batch_size=100)
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                text=f"Content {i}",
                article_title="Test",
                metadata={"index": i},
            )
            for i in range(150)
        ]

        indexer.index_chunks(chunks, batch_size=100)

        # Should be called twice: 100 + 50
        assert mock_collection.upsert.call_count == 2

    def test_includes_metadata(self, indexer, mock_chroma_client):
        """Chunk metadata is passed to ChromaDB."""
        _, mock_collection = mock_chroma_client

        chunk = Chunk(
            id="test_chunk",
            text="Test content",
            article_title="Test Article",
            metadata={"source": "help_center", "is_faq": True},
        )

        indexer.index_chunks([chunk])

        call_args = mock_collection.upsert.call_args
        assert call_args.kwargs["metadatas"] == [
            {"source": "help_center", "is_faq": True}
        ]

    def test_handles_empty_metadata(self, indexer, mock_chroma_client):
        """Chunks without metadata are handled."""
        _, mock_collection = mock_chroma_client

        chunk = Chunk(
            id="test",
            text="Content",
            article_title="Title",
            metadata=None,
        )

        indexer.index_chunks([chunk])

        call_args = mock_collection.upsert.call_args
        assert call_args.kwargs["metadatas"] == [{}]

    def test_calls_embed_texts(self, indexer, mock_openai_client, mock_chroma_client):
        """Uses OpenAI client's embed_texts method."""
        chunk = Chunk(id="test", text="Content", article_title="Title")
        indexer.index_chunks([chunk])

        mock_openai_client.embed_texts.assert_called_once_with(["Content"])


class TestCollectionStats:
    """Tests for collection statistics."""

    def test_returns_stats_dict(self, indexer, mock_chroma_client):
        """get_collection_stats returns dictionary with name and count."""
        _, mock_collection = mock_chroma_client
        mock_collection.count.return_value = 42

        stats = indexer.get_collection_stats()

        assert stats["name"] == "test_collection"
        assert stats["count"] == 42


class TestClearCollection:
    """Tests for clearing the collection."""

    def test_clears_and_recreates(self, indexer, mock_chroma_client):
        """clear_collection deletes and recreates the collection."""
        mock_client, mock_collection = mock_chroma_client

        with patch("support_agent.indexing.indexer.settings") as mock_settings:
            mock_settings.chroma_collection_name = "test_collection"

            indexer.clear_collection()

        mock_client.delete_collection.assert_called_with("test_collection")
        mock_client.create_collection.assert_called()


@pytest.mark.integration
class TestFullIndexingFlow:
    """Full integration tests for the indexing pipeline."""

    def test_full_indexing_flow(self, indexer, mock_chroma_client, sample_chunks):
        """Test complete indexing flow with sample chunks."""
        _, mock_collection = mock_chroma_client
        mock_collection.count.return_value = 0

        # Index chunks
        indexer.index_chunks(sample_chunks)

        # Verify stats
        mock_collection.count.return_value = len(sample_chunks)
        stats = indexer.get_collection_stats()

        assert stats["count"] == len(sample_chunks)
        mock_collection.upsert.assert_called()

    def test_reindex_after_clear(self, indexer, mock_chroma_client, sample_chunks):
        """Test clearing and reindexing."""
        mock_client, mock_collection = mock_chroma_client

        with patch("support_agent.indexing.indexer.settings") as mock_settings:
            mock_settings.chroma_collection_name = "test_collection"

            # Index, clear, reindex
            indexer.index_chunks(sample_chunks)
            indexer.clear_collection()
            indexer.index_chunks(sample_chunks)

        # Upsert should be called twice (before and after clear)
        assert mock_collection.upsert.call_count == 2
        mock_client.delete_collection.assert_called_once()
