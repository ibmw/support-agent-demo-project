"""
CrewAI tools for the RAG pipeline.
"""

from crewai.tools import tool

from .retriever import KnowledgeBaseRetriever

# Create a module-level retriever instance for the tool
_retriever: KnowledgeBaseRetriever | None = None


def get_retriever() -> KnowledgeBaseRetriever:
    """Get or create the retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeBaseRetriever()
    return _retriever


def set_retriever(retriever: KnowledgeBaseRetriever) -> None:
    """Set a custom retriever instance (useful for testing)."""
    global _retriever
    _retriever = retriever


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """
    Search the help center knowledge base for relevant information.

    Use this tool to find answers to customer questions by searching
    through help articles, FAQs, and documentation.

    Args:
        query: The search query describing what information you need

    Returns:
        Relevant context from the knowledge base with source references
    """
    retriever = get_retriever()
    return retriever.retrieve_with_context(query)


@tool("get_article_sources")
def get_article_sources(query: str) -> str:
    """
    Get a list of relevant article sources for a query.

    Use this tool to identify which help articles are relevant
    to a customer's question without retrieving the full content.

    Args:
        query: The search query

    Returns:
        Comma-separated list of relevant article sources
    """
    retriever = get_retriever()
    sources = retriever.get_sources(query)
    if not sources:
        return "No relevant articles found."
    return ", ".join(sources)
