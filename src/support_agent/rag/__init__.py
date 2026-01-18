"""RAG (Retrieval-Augmented Generation) module for the support agent."""

from .retriever import KnowledgeBaseRetriever, RetrievalResult

__all__ = ["KnowledgeBaseRetriever", "RetrievalResult"]
