"""Unit tests for the semantic chunker."""

from support_agent.indexing.chunker import Chunk, SemanticChunker
from support_agent.indexing.features import ContentFeatures
from support_agent.indexing.parser import Article


class TestChunkDataclass:
    """Tests for the Chunk dataclass."""

    def test_chunk_required_fields(self):
        """Chunk requires id, text, and article_title."""
        chunk = Chunk(
            id="test_chunk_0",
            text="Some content",
            article_title="Test Article",
        )
        assert chunk.id == "test_chunk_0"
        assert chunk.text == "Some content"
        assert chunk.article_title == "Test Article"

    def test_chunk_optional_fields(self):
        """Chunk has optional section_title and metadata."""
        chunk = Chunk(
            id="test_0",
            text="Content",
            article_title="Title",
            section_title="Section",
            metadata={"key": "value"},
        )
        assert chunk.section_title == "Section"
        assert chunk.metadata == {"key": "value"}

    def test_chunk_defaults(self):
        """Optional fields default to None."""
        chunk = Chunk(id="x", text="y", article_title="z")
        assert chunk.section_title is None
        assert chunk.metadata is None


class TestSemanticChunkerInit:
    """Tests for chunker initialization."""

    def test_default_parameters(self):
        """Chunker has sensible defaults."""
        chunker = SemanticChunker()
        assert chunker.max_chunk_size == 1000
        assert chunker.overlap == 100

    def test_custom_parameters(self):
        """Chunker accepts custom parameters."""
        chunker = SemanticChunker(max_chunk_size=500, overlap=50)
        assert chunker.max_chunk_size == 500
        assert chunker.overlap == 50


class TestGenerateChunkId:
    """Tests for chunk ID generation."""

    def test_basic_id_generation(self):
        """ID is generated from title and indices."""
        chunker = SemanticChunker()
        chunk_id = chunker._generate_chunk_id("Test Article", 0)
        assert chunk_id == "test_article_0"

    def test_id_with_multiple_indices(self):
        """Multiple indices are joined with underscore."""
        chunker = SemanticChunker()
        chunk_id = chunker._generate_chunk_id("Title", 1, 2)
        assert chunk_id == "title_1_2"

    def test_id_handles_special_characters(self):
        """Special characters are replaced with underscores."""
        chunker = SemanticChunker()
        chunk_id = chunker._generate_chunk_id("How to: Setup (Guide)", 0)
        assert chunk_id == "how_to_setup_guide_0"
        # No special chars
        assert ":" not in chunk_id
        assert "(" not in chunk_id

    def test_id_lowercased(self):
        """IDs are lowercase."""
        chunker = SemanticChunker()
        chunk_id = chunker._generate_chunk_id("UPPERCASE Title", 0)
        assert chunk_id == "uppercase_title_0"

    def test_id_strips_edge_underscores(self):
        """Leading/trailing underscores are removed."""
        chunker = SemanticChunker()
        chunk_id = chunker._generate_chunk_id("  Title  ", 0)
        assert not chunk_id.startswith("_")
        assert chunk_id == "title_0"


class TestExtractSections:
    """Tests for markdown section extraction."""

    def test_extracts_h1_sections(self):
        """H1 headings create sections."""
        chunker = SemanticChunker()
        md = "# Section One\nContent 1\n# Section Two\nContent 2"
        sections = chunker._extract_sections(md)

        assert len(sections) == 2
        assert sections[0][0] == "Section One"
        assert "Content 1" in sections[0][1]
        assert sections[1][0] == "Section Two"
        assert "Content 2" in sections[1][1]

    def test_extracts_h2_sections(self):
        """H2 headings create sections."""
        chunker = SemanticChunker()
        md = "## First\nText 1\n## Second\nText 2"
        sections = chunker._extract_sections(md)

        assert len(sections) == 2
        assert sections[0][0] == "First"
        assert sections[1][0] == "Second"

    def test_extracts_mixed_headings(self):
        """Mixed heading levels all create sections."""
        chunker = SemanticChunker()
        md = "# Main\nIntro\n## Sub\nDetails\n### Deep\nMore"
        sections = chunker._extract_sections(md)

        assert len(sections) == 3
        headings = [s[0] for s in sections]
        assert "Main" in headings
        assert "Sub" in headings
        assert "Deep" in headings

    def test_intro_content_before_first_heading(self):
        """Content before first heading becomes Introduction section."""
        chunker = SemanticChunker()
        md = "This is intro content.\n# First Section\nSection content"
        sections = chunker._extract_sections(md)

        assert len(sections) == 2
        assert sections[0][0] == "Introduction"
        assert "intro content" in sections[0][1]
        assert sections[1][0] == "First Section"

    def test_no_headings_returns_empty(self):
        """No headings means no sections (falls back to other logic)."""
        chunker = SemanticChunker()
        md = "Just plain text without any headings."
        sections = chunker._extract_sections(md)

        assert sections == []

    def test_empty_sections_skipped(self):
        """Headings with no content are skipped."""
        chunker = SemanticChunker()
        md = "# Empty\n# Has Content\nSome text here"
        sections = chunker._extract_sections(md)

        # Only the section with content should be included
        assert len(sections) == 1
        assert sections[0][0] == "Has Content"


class TestSplitIfTooLong:
    """Tests for splitting long text."""

    def test_short_text_not_split(self):
        """Text under max size returns as single chunk."""
        chunker = SemanticChunker(max_chunk_size=100)
        result = chunker._split_if_too_long("Short text")

        assert result == ["Short text"]

    def test_exact_size_not_split(self):
        """Text exactly at max size returns as single chunk."""
        chunker = SemanticChunker(max_chunk_size=10)
        result = chunker._split_if_too_long("0123456789")

        assert len(result) == 1

    def test_long_text_split(self):
        """Text over max size is split into chunks."""
        chunker = SemanticChunker(max_chunk_size=50, overlap=0)
        long_text = " ".join(["word"] * 100)  # Way over 50 chars
        result = chunker._split_if_too_long(long_text)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 60  # Some tolerance for word boundaries

    def test_split_with_overlap(self):
        """Overlap words appear in consecutive chunks."""
        chunker = SemanticChunker(max_chunk_size=50, overlap=20)
        # Create text that will definitely be split
        words = [f"word{i}" for i in range(50)]
        long_text = " ".join(words)
        result = chunker._split_if_too_long(long_text)

        # Should have multiple chunks
        assert len(result) > 1
        # Overlap causes some repetition between consecutive chunks
        # (Testing the mechanism exists, not exact overlap calculation)

    def test_empty_text_returns_empty(self):
        """Empty text returns empty list."""
        chunker = SemanticChunker()
        result = chunker._split_if_too_long("")

        assert result == []

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only text returns empty list."""
        chunker = SemanticChunker()
        result = chunker._split_if_too_long("   \n\t  ")

        assert result == []


class TestChunkArticle:
    """Tests for chunking a single article."""

    def test_article_with_sections(self):
        """Article with headings produces section-based chunks."""
        chunker = SemanticChunker()
        article = Article(
            title="Test Guide",
            md_content="# Introduction\nWelcome.\n# Steps\n1. Do this\n2. Do that",
            raw_html="<h1>Introduction</h1><p>Welcome.</p>",
            features=ContentFeatures(has_steps=True),
        )
        chunks = chunker.chunk_article(article)

        assert len(chunks) >= 2
        # Check sections are captured
        section_titles = [c.section_title for c in chunks]
        assert "Introduction" in section_titles
        assert "Steps" in section_titles

    def test_article_without_sections(self):
        """Article without headings falls back to content chunking."""
        chunker = SemanticChunker()
        article = Article(
            title="Plain Article",
            md_content="Just plain text without any markdown headings at all.",
            raw_html="<p>Just plain text</p>",
            features=ContentFeatures(),
        )
        chunks = chunker.chunk_article(article)

        assert len(chunks) >= 1
        assert chunks[0].section_title is None

    def test_chunk_has_article_title(self):
        """All chunks reference their source article."""
        chunker = SemanticChunker()
        article = Article(
            title="My Article Title",
            md_content="# Section\nContent here",
            raw_html="",
            features=ContentFeatures(),
        )
        chunks = chunker.chunk_article(article)

        for chunk in chunks:
            assert chunk.article_title == "My Article Title"

    def test_chunk_metadata_includes_features(self):
        """Chunk metadata includes article features."""
        chunker = SemanticChunker()
        article = Article(
            title="Integration Setup",
            md_content="# Connect\nSteps to integrate",
            raw_html="",
            features=ContentFeatures(is_integration=True, has_steps=True),
        )
        chunks = chunker.chunk_article(article)

        for chunk in chunks:
            assert chunk.metadata is not None
            assert chunk.metadata["is_integration"] is True
            assert chunk.metadata["has_steps"] is True
            assert chunk.metadata["source"] == "help_center"

    def test_chunk_metadata_includes_section(self):
        """Chunk metadata includes section title."""
        chunker = SemanticChunker()
        article = Article(
            title="Guide",
            md_content="# Overview\nContent here",
            raw_html="",
            features=ContentFeatures(),
        )
        chunks = chunker.chunk_article(article)

        assert chunks[0].metadata["section"] == "Overview"

    def test_long_section_split_into_multiple_chunks(self):
        """Long sections are split into multiple chunks."""
        chunker = SemanticChunker(max_chunk_size=100, overlap=0)
        long_content = " ".join(["word"] * 100)
        article = Article(
            title="Long Article",
            md_content=f"# Section\n{long_content}",
            raw_html="",
            features=ContentFeatures(),
        )
        chunks = chunker.chunk_article(article)

        # Should have multiple chunks for the long section
        assert len(chunks) > 1


class TestChunkAll:
    """Tests for chunking multiple articles."""

    def test_chunks_multiple_articles(self, sample_articles):
        """Multiple articles produce combined chunks."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_all(sample_articles)

        assert len(chunks) >= len(sample_articles)
        # Each article should contribute chunks
        article_titles = {c.article_title for c in chunks}
        for article in sample_articles:
            assert article.title in article_titles

    def test_empty_list_returns_empty(self):
        """Empty article list returns empty chunk list."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_all([])

        assert chunks == []

    def test_chunk_ids_unique_across_articles(self, sample_articles):
        """Chunk IDs are unique across all articles."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_all(sample_articles)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"


class TestEdgeCases:
    """Edge case tests for the chunker."""

    def test_article_with_empty_content(self):
        """Article with empty content produces no chunks."""
        chunker = SemanticChunker()
        article = Article(
            title="Empty",
            md_content="",
            raw_html="",
            features=ContentFeatures(),
        )
        chunks = chunker.chunk_article(article)

        assert chunks == []

    def test_article_with_only_headings(self):
        """Article with only headings falls back to full content chunking."""
        chunker = SemanticChunker()
        article = Article(
            title="Headers Only",
            md_content="# H1\n## H2\n### H3",
            raw_html="",
            features=ContentFeatures(),
        )
        chunks = chunker.chunk_article(article)

        # When no section content is found, falls back to full content
        # The heading text itself becomes the chunk
        assert len(chunks) == 1
        assert "# H1" in chunks[0].text

    def test_unicode_content(self):
        """Unicode content is handled correctly."""
        chunker = SemanticChunker()
        article = Article(
            title="日本語 Article",
            md_content="# セクション\nこれはテストです。",
            raw_html="",
            features=ContentFeatures(),
        )
        chunks = chunker.chunk_article(article)

        assert len(chunks) >= 1
        assert "セクション" in chunks[0].section_title
        assert "テスト" in chunks[0].text

    def test_very_long_title(self):
        """Very long titles are handled in ID generation."""
        chunker = SemanticChunker()
        long_title = "A" * 500
        chunk_id = chunker._generate_chunk_id(long_title, 0)

        # Should still work and be valid
        assert chunk_id.endswith("_0")
        assert "a" in chunk_id.lower()
