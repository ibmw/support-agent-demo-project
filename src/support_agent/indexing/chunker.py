"""
Semantic chunking for Help Center articles in markdown format.
"""

import re
from dataclasses import dataclass

from .parser import Article


@dataclass
class Chunk:
    """Represents a chunk of content from an article."""

    id: str
    text: str
    article_title: str
    section_title: str | None = None
    metadata: dict | None = None


class SemanticChunker:
    """
    Chunk articles semantically by markdown sections.

    Strategy: Split by markdown headings (#, ##, ###) to preserve context.
    Falls back to paragraph-based chunking if no headings found.
    """

    # Regex to match markdown headings (# Title, ## Subtitle, etc.)
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_article(self, article: Article) -> list[Chunk]:
        """Split an article into semantic chunks."""
        chunks = []

        # Extract sections from markdown
        sections = self._extract_sections(article.md_content)

        # Base metadata from article features
        base_metadata = {
            "source": "help_center",
            "article_title": article.title,
            **article.features.to_dict(),
        }

        if sections:
            # Chunk by sections
            for i, (section_title, section_content) in enumerate(sections):
                chunk_texts = self._split_if_too_long(section_content)
                for j, text in enumerate(chunk_texts):
                    chunk_id = self._generate_chunk_id(article.title, i, j)
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=text,
                            article_title=article.title,
                            section_title=section_title,
                            metadata={
                                **base_metadata,
                                "section": section_title,
                            },
                        )
                    )
        else:
            # Fall back to full content chunking
            chunk_texts = self._split_if_too_long(article.md_content)
            for i, text in enumerate(chunk_texts):
                chunk_id = self._generate_chunk_id(article.title, i)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=text,
                        article_title=article.title,
                        metadata=base_metadata,
                    )
                )

        return chunks

    def _generate_chunk_id(self, title: str, *indices: int) -> str:
        """Generate a unique chunk ID from title and indices."""
        # Clean title for ID: lowercase, replace spaces/special chars with underscore
        clean_title = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        suffix = "_".join(str(i) for i in indices)
        return f"{clean_title}_{suffix}"

    def _extract_sections(self, md_content: str) -> list[tuple[str, str]]:
        """Extract sections based on markdown headings."""
        sections = []

        # Find all headings with their positions
        headings = list(self.HEADING_PATTERN.finditer(md_content))

        if not headings:
            return []

        # Extract content before first heading (if any)
        first_heading_start = headings[0].start()
        if first_heading_start > 0:
            intro_content = md_content[:first_heading_start].strip()
            if intro_content:
                sections.append(("Introduction", intro_content))

        # Extract each section
        for i, match in enumerate(headings):
            heading_text = match.group(2).strip()

            # Find the end of this section (start of next heading or end of content)
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(md_content)

            content = md_content[start:end].strip()
            if content:
                sections.append((heading_text, content))

        return sections

    def _split_if_too_long(self, text: str) -> list[str]:
        """Split text if it exceeds max_chunk_size."""
        if len(text) <= self.max_chunk_size:
            return [text] if text.strip() else []

        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > self.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Add overlap from previous chunk
                overlap_words = (
                    current_chunk[-self.overlap // 10 :] if self.overlap else []
                )
                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_all(self, articles: list[Article]) -> list[Chunk]:
        """Chunk all articles."""
        all_chunks = []
        for article in articles:
            all_chunks.extend(self.chunk_article(article))
        return all_chunks
