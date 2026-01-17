"""Shared pytest fixtures for the test suite."""

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
# Mock Fixtures (for future use)
# ============================================================================


@pytest.fixture
def mock_embeddings():
    """
    Mock embedding function that returns deterministic vectors.

    Returns a function that generates fake embeddings based on text hash.
    """

    def _embed(texts: list[str]) -> list[list[float]]:
        """Generate fake embeddings (384 dimensions like text-embedding-3-small)."""
        embeddings = []
        for text in texts:
            # Create deterministic fake embedding from text hash
            seed = hash(text) % 10000
            embedding = [(seed + i) % 100 / 100.0 for i in range(384)]
            embeddings.append(embedding)
        return embeddings

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
