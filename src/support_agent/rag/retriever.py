"""
Knowledge base retriever for querying ChromaDB.
"""

import hashlib
from dataclasses import dataclass, field

import chromadb

from ..clients import get_openai_client
from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__, component="retriever")


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""

    text: str
    article_title: str
    section_title: str | None = None
    score: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Get formatted source reference."""
        if self.section_title:
            return f"{self.article_title} > {self.section_title}"
        return self.article_title


class KnowledgeBaseRetriever:
    """
    Retrieve relevant chunks from the knowledge base.

    Uses ChromaDB for vector search with OpenAI embeddings.
    """

    def __init__(
        self,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ):
        """
        Initialize the retriever.

        Args:
            top_k: Number of results to return (default from settings)
            score_threshold: Minimum similarity score (0-1, higher is more similar)
        """
        self.top_k = top_k or settings.retrieval_top_k
        self.score_threshold = score_threshold

        self.openai_client = get_openai_client()
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Retriever initialized",
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            collection=settings.chroma_collection_name,
        )

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query

        Returns:
            List of RetrievalResult objects, sorted by relevance
        """
        # Create query hash for log correlation across services
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        log = logger.bind(
            query_preview=query[:100],
            query_hash=query_hash,
            query_length=len(query),
        )

        # Check if collection is empty
        if self.collection.count() == 0:
            log.debug("Collection is empty, returning no results")
            return []

        # Get query embedding
        query_embedding = self.openai_client.embed_query(query)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Parse results
        retrieval_results = []
        if results["documents"] and results["documents"][0]:
            # Re-arrange data structure to be more readable
            documents = results["documents"][0]
            metadatas = (
                results["metadatas"][0]
                if results["metadatas"]
                else [{}] * len(documents)
            )
            distances = (
                results["distances"][0]
                if results["distances"]
                else [0.0] * len(documents)
            )

            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Convert distance to similarity score (cosine distance to similarity)
                score = 1 - distance

                # Apply score threshold if set
                if self.score_threshold is not None and score < self.score_threshold:
                    continue

                retrieval_results.append(
                    RetrievalResult(
                        text=doc,
                        article_title=metadata.get("article_title", "Unknown"),
                        section_title=metadata.get("section"),
                        score=score,
                        metadata=metadata,
                    )
                )

        log.debug(
            "Retrieval completed",
            results_count=len(retrieval_results),
            top_score=round(retrieval_results[0].score, 3)
            if retrieval_results
            else None,
        )

        return retrieval_results

    def retrieve_with_context(self, query: str) -> str:
        """
        Retrieve and format context for LLM consumption.

        Args:
            query: The search query

        Returns:
            Formatted context string for the LLM
        """
        results = self.retrieve(query)

        if not results:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.source
            context_parts.append(f"[Source {i}: {source}]\n{result.text}\n")

        return "\n---\n".join(context_parts)

    def get_sources(self, query: str) -> list[str]:
        """
        Get list of source references for a query.

        Args:
            query: The search query

        Returns:
            List of source strings (article titles or sections)
        """
        results = self.retrieve(query)
        return [result.source for result in results]
