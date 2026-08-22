"""
Unit tests for the embedding service.

The test verifies that the configured embedding model can generate vectors
and that repeated calls reuse the same model instance.
"""

from app.rag.embeddings import get_embedding_model


def test_embedding_model_can_embed_text() -> None:
    """
    The embedding service should successfully convert text into a vector.
    """

    embeddings = get_embedding_model()

    vector = embeddings.embed_query(
        "What is the return policy?"
    )

    assert isinstance(vector, list)
    assert len(vector) > 0


def test_embedding_model_is_cached() -> None:
    """
    The embedding model should be instantiated only once per process.

    This protects the application from repeatedly loading the relatively
    expensive Sentence Transformer model.
    """

    first = get_embedding_model()
    second = get_embedding_model()

    assert first is second