"""
ChromaDB indexer for the knowledge base.
"""

import time

import chromadb

from ..clients import get_openai_client
from ..config import settings
from ..logging import get_logger
from .chunker import Chunk

logger = get_logger(__name__, component="indexer")


class KnowledgeBaseIndexer:
    """Index chunks into ChromaDB with OpenAI embeddings."""

    def __init__(self):
        self.openai_client = get_openai_client()
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            "Indexer initialized",
            collection=settings.chroma_collection_name,
            persist_dir=str(settings.chroma_persist_dir),
        )

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> None:
        """Index all chunks into ChromaDB."""
        total = len(chunks)
        logger.info("Starting indexing", total_chunks=total, batch_size=batch_size)

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]

            ids = [chunk.id for chunk in batch]
            texts = [chunk.text for chunk in batch]
            metadatas = [chunk.metadata or {} for chunk in batch]

            embeddings = self.openai_client.embed_texts(texts)

            self.collection.upsert(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
            )

            indexed = min(i + batch_size, total)
            logger.debug("Batch indexed", indexed=indexed, total=total)
            print(f"Indexed {indexed}/{total} chunks")

            # Small delay between batches to avoid rate limits
            if indexed < total:
                time.sleep(1.0)

        logger.info("Indexing completed", total_chunks=total)

    def get_collection_stats(self) -> dict:
        """Get statistics about the indexed collection."""
        stats = {"name": self.collection.name, "count": self.collection.count()}
        logger.debug("Collection stats retrieved", **stats)
        return stats

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        logger.warning(
            "Clearing collection", collection=settings.chroma_collection_name
        )
        self.chroma_client.delete_collection(settings.chroma_collection_name)
        self.collection = self.chroma_client.create_collection(
            name=settings.chroma_collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection cleared and recreated")
