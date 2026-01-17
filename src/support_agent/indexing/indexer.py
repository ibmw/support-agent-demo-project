"""
ChromaDB indexer for the knowledge base.
"""

import random
import re
import time

import chromadb
from openai import APIConnectionError, OpenAI, RateLimitError

from ..config import settings
from .chunker import Chunk


class KnowledgeBaseIndexer:
    """Index chunks into ChromaDB with OpenAI embeddings."""

    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding(self, text: str, max_retries: int = 5) -> list[float]:
        """
        Generate embedding for a single text with retry logic.

        Implements exponential backoff for rate limit and connection error handling.

        Args:
            text: The text to embed
            max_retries: Maximum number of retry attempts (default: 5)

        Returns:
            Embedding vector as list of floats

        Raises:
            RateLimitError: If rate limits are exceeded after all retries
            APIConnectionError: If connection errors persist after all retries
        """
        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    model=settings.embedding_model, input=text
                )
                return response.data[0].embedding
            except (RateLimitError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    raise

                wait_time = self._extract_wait_time(e) or self._calculate_backoff(
                    attempt
                )
                time.sleep(wait_time)

        raise APIConnectionError("Max retries exceeded")

    def _get_embeddings_batch(
        self, texts: list[str], max_retries: int = 5
    ) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts with retry logic.

        Implements exponential backoff for rate limit and connection error handling.

        Args:
            texts: List of texts to embed
            max_retries: Maximum number of retry attempts (default: 5)

        Returns:
            List of embedding vectors

        Raises:
            RateLimitError: If rate limits are exceeded after all retries
            APIConnectionError: If connection errors persist after all retries
        """
        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    model=settings.embedding_model, input=texts
                )
                return [item.embedding for item in response.data]
            except (RateLimitError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    raise

                wait_time = self._extract_wait_time(e) or self._calculate_backoff(
                    attempt
                )
                time.sleep(wait_time)

        raise APIConnectionError("Max retries exceeded")

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> None:
        """Index all chunks into ChromaDB."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = [chunk.id for chunk in batch]
            texts = [chunk.text for chunk in batch]
            metadatas = [chunk.metadata or {} for chunk in batch]

            embeddings = self._get_embeddings_batch(texts)

            self.collection.upsert(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
            )

            print(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    def get_collection_stats(self) -> dict:
        """Get statistics about the indexed collection. Used for debugging and monitoring."""
        return {"name": self.collection.name, "count": self.collection.count()}

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.chroma_client.delete_collection(settings.chroma_collection_name)
        self.collection = self.chroma_client.create_collection(
            name=settings.chroma_collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _calculate_backoff(
        self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
    ) -> float:
        """
        Calculate exponential backoff delay with jitter.

        Args:
            attempt: Current attempt number (0-indexed)
            base_delay: Base delay in seconds (default: 1.0)
            max_delay: Maximum delay in seconds (default: 60.0)

        Returns:
            Delay in seconds
        """
        delay = min(base_delay * (2**attempt), max_delay)
        jitter = random.uniform(0, delay * 0.25)
        return delay + jitter

    def _extract_wait_time(self, error: Exception) -> float | None:
        """
        Extract wait time from error message if available.

        Args:
            error: The exception (RateLimitError or APIConnectionError)

        Returns:
            Wait time in seconds, or None if not found
        """
        error_msg = str(error)
        match = re.search(
            r"try again in (\d+)\s*(?:s|second|seconds)", error_msg, re.IGNORECASE
        )
        if match:
            return float(match.group(1))
        return None
