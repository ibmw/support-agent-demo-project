"""Shared pytest fixtures for the test suite."""

import os

# Set test environment variables BEFORE importing anything that uses settings
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "test-public-key")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "test-secret-key")
# Disable LangFuse tracing during tests to avoid 401 errors with fake credentials
os.environ.setdefault("LANGFUSE_ENABLED", "false")

import pytest

from support_agent.indexing import Article, Chunk
from support_agent.indexing.features import ContentFeatures

# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_html() -> str:
    """Sample HTML content for testing parser."""
    return """
    <style>p { margin: 0; }</style>
    <h1>Getting Started</h1>
    <p>Welcome to our Help Center.</p>
    <h2>Steps</h2>
    <ol>
        <li>Go to Settings</li>
        <li>Click on Integrations</li>
        <li>Select your app</li>
    </ol>
    <h2>FAQ</h2>
    <ul>
        <li>Question 1: Answer here</li>
        <li>Question 2: Another answer</li>
    </ul>
    <img src="example.png" />
    """


@pytest.fixture
def sample_markdown() -> str:
    """Expected markdown output from sample HTML."""
    return """# Getting Started
Welcome to our Help Center.
## Steps
1. Go to Settings
2. Click on Integrations
3. Select your app
## FAQ
- Question 1: Answer here
- Question 2: Another answer"""


@pytest.fixture
def sample_article(sample_html: str) -> Article:
    """Sample Article for testing."""
    return Article(
        title="Getting Started Guide",
        md_content="# Getting Started\nWelcome to our Help Center.",
        raw_html=sample_html,
        features=ContentFeatures(has_steps=True, content_type="how-to"),
    )


@pytest.fixture
def sample_articles() -> list[Article]:
    """Small set of articles for testing."""
    return [
        Article(
            title="Integration Guide",
            md_content="# Integration\n## Steps\n1. Connect\n2. Configure",
            raw_html="<h1>Integration</h1>",
            features=ContentFeatures(is_integration=True, has_steps=True),
        ),
        Article(
            title="Troubleshooting FAQ",
            md_content="# FAQ\nQ: Not working?\nA: Try restarting.",
            raw_html="<h1>FAQ</h1>",
            features=ContentFeatures(is_faq=True, is_troubleshooting=True),
        ),
        Article(
            title="Pricing Plans",
            md_content="# Pricing\nStarter: $10/mo\nPro: $25/mo",
            raw_html="<h1>Pricing</h1>",
            features=ContentFeatures(has_pricing=True),
        ),
    ]


@pytest.fixture
def sample_chunks(sample_articles: list[Article]) -> list[Chunk]:
    """Pre-chunked test data."""
    return [
        Chunk(
            id="integration_guide_0",
            text="# Integration\n## Steps\n1. Connect\n2. Configure",
            article_title="Integration Guide",
            section_title="Steps",
            metadata={"source": "help_center", "is_integration": True},
        ),
        Chunk(
            id="troubleshooting_faq_0",
            text="# FAQ\nQ: Not working?\nA: Try restarting.",
            article_title="Troubleshooting FAQ",
            section_title="FAQ",
            metadata={"source": "help_center", "is_faq": True},
        ),
        Chunk(
            id="pricing_plans_0",
            text="# Pricing\nStarter: $10/mo\nPro: $25/mo",
            article_title="Pricing Plans",
            metadata={"source": "help_center", "has_pricing": True},
        ),
    ]


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_embeddings():
    """
    Mock embedding function that returns deterministic vectors.

    Returns a function that generates fake embeddings based on text hash.
    """
    from tests.helpers import generate_fake_embedding

    def _embed(texts: list[str]) -> list[list[float]]:
        """Generate fake embeddings (1536 dimensions like text-embedding-3-small)."""
        return [generate_fake_embedding(text) for text in texts]

    return _embed


@pytest.fixture
def mock_llm_response():
    """
    Mock LLM response for testing agent logic.

    Returns a factory function to create mock responses.
    """

    def _create_response(
        action: str = "CLOSE",
        message: str = "Here's the answer to your question.",
        sources: list[str] | None = None,
    ) -> dict:
        return {
            "action": action,
            "message": message,
            "sources": sources or ["Getting Started Guide"],
            "confidence": 0.95,
        }

    return _create_response


# ============================================================================
# Mock LLM Fixture
# ============================================================================


@pytest.fixture
def mock_llm():
    """
    Mock LLM fixture for predictable CrewAI agent responses.

    Provides a context manager and response configurator that mocks
    the entire CrewAI stack. Responses are AgentResponse objects.

    Usage:
        def test_something(mock_llm):
            # Configure the response
            mock_llm.set_response(
                action="CLOSE",
                message="Here's your answer",
                sources=["Guide"],
                confidence=0.9,
            )

            # Your code that uses the agent
            crew = SupportCrew()
            response = crew.process_query("How do I...?")

            assert response.action == "CLOSE"
            assert mock_llm.call_count == 1
            assert "How do I" in mock_llm.last_query
    """
    from unittest.mock import MagicMock, patch

    from support_agent.agent.schemas import AgentResponse

    class MockLLM:
        def __init__(self):
            self._responses: list[AgentResponse] = []
            self._default_response = AgentResponse(
                action="CLOSE",
                message="Default mock response",
                sources=["Mock Source"],
                confidence=0.9,
            )
            self._call_count = 0
            self._queries: list[str] = []
            self._patches = []
            self._mock_crew_cls = None

        def set_response(
            self,
            action: str = "CLOSE",
            message: str = "Mock response",
            sources: list[str] | None = None,
            confidence: float = 0.9,
        ) -> "MockLLM":
            """Set the next response to return."""
            response = AgentResponse(
                action=action,
                message=message,
                sources=sources or [],
                confidence=confidence,
            )
            self._responses.append(response)
            return self

        def set_responses(self, responses: list[dict]) -> "MockLLM":
            """Set multiple responses for sequential calls."""
            for r in responses:
                self.set_response(**r)
            return self

        def _get_next_response(self) -> AgentResponse:
            """Get the next response (cycles if multiple configured)."""
            if self._responses:
                return self._responses.pop(0)
            return self._default_response

        def _create_mock_result(self, query: str) -> MagicMock:
            """Create a mock crew result with pydantic output."""
            self._call_count += 1
            self._queries.append(query)
            response = self._get_next_response()
            mock_result = MagicMock()
            mock_result.pydantic = response
            mock_result.tasks_output = []
            # Add token usage for testing
            mock_result.token_usage = MagicMock()
            mock_result.token_usage.total_tokens = 100
            mock_result.token_usage.prompt_tokens = 80
            mock_result.token_usage.completion_tokens = 20
            mock_result.token_usage.successful_requests = 1
            return mock_result

        @property
        def call_count(self) -> int:
            """Number of times the LLM was called."""
            return self._call_count

        @property
        def queries(self) -> list[str]:
            """List of all queries sent to the LLM."""
            return self._queries

        @property
        def last_query(self) -> str | None:
            """The most recent query sent to the LLM."""
            return self._queries[-1] if self._queries else None

        def reset(self) -> None:
            """Reset call tracking."""
            self._call_count = 0
            self._queries = []
            self._responses = []

    mock = MockLLM()

    # Patch CrewAI components
    with (
        patch("support_agent.agent.crew.Agent") as _mock_agent,
        patch("support_agent.agent.crew.Crew") as mock_crew_cls,
        patch("support_agent.agent.crew.Task") as mock_task,
        patch("support_agent.agent.crew.get_langfuse_client") as mock_langfuse,
    ):
        # Configure mock langfuse to not fail
        mock_langfuse_client = MagicMock()
        mock_langfuse.return_value = mock_langfuse_client

        # Configure crew to capture query and return mock response
        def kickoff_side_effect(*args, **kwargs):
            # Extract query from the task description
            task_call = mock_task.call_args
            if task_call:
                description = task_call.kwargs.get("description", "")
                return mock._create_mock_result(description)
            return mock._create_mock_result("")

        mock_crew_cls.return_value.kickoff.side_effect = kickoff_side_effect
        mock._mock_crew_cls = mock_crew_cls

        yield mock


# ============================================================================
# Test ChromaDB Fixture
# ============================================================================


@pytest.fixture
def test_chroma(tmp_path):
    """
    Provide an in-memory ChromaDB instance for testing.

    Creates a real ChromaDB client with a temporary directory,
    giving more realistic test behavior than mocking.

    Usage:
        def test_retrieval(test_chroma, mock_embeddings):
            client, collection = test_chroma

            # Add documents
            collection.add(
                ids=["doc1", "doc2"],
                documents=["First doc", "Second doc"],
                embeddings=mock_embeddings(["First doc", "Second doc"]),
            )

            # Query
            results = collection.query(
                query_embeddings=[mock_embeddings(["query"])[0]],
                n_results=2,
            )
            assert len(results["documents"][0]) == 2
    """
    import chromadb

    # Create persistent client in temp directory (acts like in-memory for tests)
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_test"))

    # Create a test collection with cosine similarity
    collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"},
    )

    yield client, collection

    # Cleanup: delete collection after test
    try:
        client.delete_collection("test_collection")
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def test_chroma_with_data(test_chroma, mock_embeddings, sample_chunks):
    """
    ChromaDB instance pre-populated with sample chunks.

    Useful for tests that need data already indexed.

    Usage:
        def test_retrieval(test_chroma_with_data, mock_embeddings):
            client, collection = test_chroma_with_data

            # Collection already has sample_chunks indexed
            results = collection.query(
                query_embeddings=[mock_embeddings(["integration"])[0]],
                n_results=3,
            )
            assert collection.count() == 3
    """
    client, collection = test_chroma

    # Index sample chunks
    texts = [chunk.text for chunk in sample_chunks]
    embeddings = mock_embeddings(texts)

    collection.add(
        ids=[chunk.id for chunk in sample_chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[chunk.metadata or {} for chunk in sample_chunks],
    )

    return client, collection
