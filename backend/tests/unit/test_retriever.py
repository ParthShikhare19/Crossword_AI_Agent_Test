"""
Unit tests for the semantic retrieval layer.

These tests verify candidate retrieval behavior. They intentionally do not
test document precedence yet because precedence belongs to the separate
metadata-aware reranking layer.
"""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.rag.retriever import KnowledgeRetriever


def test_retrieve_candidates_preserves_source_metadata() -> None:
    """
    Retrieved results must retain enough metadata to identify their source.
    """

    mock_vectorstore = MagicMock()

    mock_vectorstore.similarity_search_with_score.return_value = [
        (
            Document(
                page_content=(
                    "Customers may return eligible products "
                    "within 30 calendar days of delivery."
                ),
                metadata={
                    "chunk_id": "returns:001",
                    "filename": "01-returns-policy-current.md",
                    "heading": "Standard return window",
                    "status": "active",
                    "audience": "customer",
                    "policy_authority": "official",
                },
            ),
            0.15,
        )
    ]

    retriever = KnowledgeRetriever(mock_vectorstore)

    results = retriever.retrieve_candidates(
        "How long can I return an unused backpack?"
    )

    assert len(results) == 1

    result = results[0]

    assert result.filename == "01-returns-policy-current.md"
    assert result.heading == "Standard return window"
    assert result.metadata["status"] == "active"
    assert result.metadata["policy_authority"] == "official"


def test_empty_query_is_rejected() -> None:
    """
    An empty retrieval query should fail before calling the vector store.
    """

    mock_vectorstore = MagicMock()

    retriever = KnowledgeRetriever(mock_vectorstore)

    try:
        retriever.retrieve_candidates("")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected empty query to raise ValueError."
        )