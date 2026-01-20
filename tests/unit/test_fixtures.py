"""Tests for the test fixtures themselves.

Ensures mock_llm and test_chroma fixtures work correctly.
"""

import pytest

from support_agent.agent.crew import SupportCrew, reset_crew


@pytest.fixture(autouse=True)
def reset_crew_singleton():
    """Reset the crew singleton before each test."""
    reset_crew()
    yield
    reset_crew()


class TestMockLLM:
    """Tests for the mock_llm fixture."""

    def test_default_response(self, mock_llm):
        """Returns default CLOSE response when no response configured."""
        crew = SupportCrew()
        response = crew.process_query("Test query")

        assert response.action == "CLOSE"
        assert response.message == "Default mock response"
        assert mock_llm.call_count == 1

    def test_configured_response(self, mock_llm):
        """Returns configured response."""
        mock_llm.set_response(
            action="HANDOVER",
            message="Transferring to human agent",
            sources=["Support Policy"],
            confidence=0.85,
        )

        crew = SupportCrew()
        response = crew.process_query("I need a refund")

        assert response.action == "HANDOVER"
        assert response.message == "Transferring to human agent"
        assert "Support Policy" in response.sources
        assert response.confidence == 0.85

    def test_multiple_responses(self, mock_llm):
        """Returns different responses for sequential calls."""
        mock_llm.set_response(action="WAIT", message="First response")
        mock_llm.set_response(action="CLOSE", message="Second response")

        crew = SupportCrew()

        response1 = crew.process_query("First query")
        response2 = crew.process_query("Second query")

        assert response1.action == "WAIT"
        assert response2.action == "CLOSE"
        assert mock_llm.call_count == 2

    def test_tracks_queries(self, mock_llm):
        """Tracks all queries sent to the LLM."""
        crew = SupportCrew()
        crew.process_query("How do I reset my password?")
        crew.process_query("What are your pricing plans?")

        assert mock_llm.call_count == 2
        assert "password" in mock_llm.queries[0]
        assert "pricing" in mock_llm.queries[1]
        assert "pricing" in mock_llm.last_query

    def test_reset_clears_tracking(self, mock_llm):
        """Reset clears call tracking."""
        crew = SupportCrew()
        crew.process_query("Query 1")
        crew.process_query("Query 2")

        assert mock_llm.call_count == 2

        mock_llm.reset()

        assert mock_llm.call_count == 0
        assert mock_llm.queries == []
        assert mock_llm.last_query is None

    def test_fluent_api(self, mock_llm):
        """Supports fluent API for chaining."""
        mock_llm.set_response(action="WAIT").set_response(action="CLOSE")

        crew = SupportCrew()
        r1 = crew.process_query("Q1")
        r2 = crew.process_query("Q2")

        assert r1.action == "WAIT"
        assert r2.action == "CLOSE"

    def test_set_responses_batch(self, mock_llm):
        """Can set multiple responses at once."""
        mock_llm.set_responses([
            {"action": "WAIT", "message": "Need more info"},
            {"action": "CLOSE", "message": "Got it"},
        ])

        crew = SupportCrew()
        r1 = crew.process_query("Q1")
        r2 = crew.process_query("Q2")

        assert r1.action == "WAIT"
        assert r2.action == "CLOSE"


class TestTestChroma:
    """Tests for the test_chroma fixture."""

    def test_provides_client_and_collection(self, test_chroma):
        """Provides both client and collection."""
        client, collection = test_chroma

        assert client is not None
        assert collection is not None
        assert collection.count() == 0

    def test_collection_operations(self, test_chroma, mock_embeddings):
        """Collection supports standard operations."""
        _, collection = test_chroma

        # Add documents
        texts = ["First document", "Second document"]
        embeddings = mock_embeddings(texts)

        collection.add(
            ids=["doc1", "doc2"],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"type": "first"}, {"type": "second"}],
        )

        assert collection.count() == 2

        # Query
        results = collection.query(
            query_embeddings=[mock_embeddings(["First"])[0]],
            n_results=2,
            include=["documents", "metadatas"],
        )

        assert len(results["documents"][0]) == 2
        assert results["metadatas"] is not None

    def test_upsert_works(self, test_chroma, mock_embeddings):
        """Upsert operation works correctly."""
        _, collection = test_chroma

        # Initial add
        collection.add(
            ids=["doc1"],
            documents=["Original content"],
            embeddings=mock_embeddings(["Original content"]),
        )

        # Upsert with updated content
        collection.upsert(
            ids=["doc1"],
            documents=["Updated content"],
            embeddings=mock_embeddings(["Updated content"]),
        )

        # Should still be 1 document
        assert collection.count() == 1

        # Get the document
        result = collection.get(ids=["doc1"], include=["documents"])
        assert result["documents"][0] == "Updated content"


class TestTestChromaWithData:
    """Tests for the test_chroma_with_data fixture."""

    def test_pre_populated(self, test_chroma_with_data):
        """Collection is pre-populated with sample chunks."""
        _, collection = test_chroma_with_data

        # Should have 3 sample chunks
        assert collection.count() == 3

    def test_can_query(self, test_chroma_with_data, mock_embeddings):
        """Can query the pre-populated data."""
        _, collection = test_chroma_with_data

        results = collection.query(
            query_embeddings=[mock_embeddings(["integration setup"])[0]],
            n_results=3,
        )

        assert len(results["documents"][0]) == 3


class TestMockEmbeddings:
    """Tests for the mock_embeddings fixture."""

    def test_returns_correct_dimensions(self, mock_embeddings):
        """Returns 1536-dimension vectors."""
        embeddings = mock_embeddings(["test text"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    def test_deterministic(self, mock_embeddings):
        """Same text returns same embedding."""
        e1 = mock_embeddings(["hello world"])
        e2 = mock_embeddings(["hello world"])

        assert e1 == e2

    def test_different_texts_different_embeddings(self, mock_embeddings):
        """Different texts return different embeddings."""
        embeddings = mock_embeddings(["text one", "text two"])

        assert embeddings[0] != embeddings[1]

    def test_normalized(self, mock_embeddings):
        """Embeddings are L2 normalized (unit vectors)."""
        import numpy as np

        embeddings = mock_embeddings(["test"])
        norm = np.linalg.norm(embeddings[0])

        assert abs(norm - 1.0) < 1e-6
