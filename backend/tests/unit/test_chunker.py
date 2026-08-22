"""
Unit tests for the knowledge-base chunking pipeline.

These tests verify that important source information is retained during
chunking. Losing metadata or headings at this stage would make reliable
source citation and document-precedence decisions impossible later.
"""

from app.rag.chunker import chunk_document
from app.rag.document_loader import KnowledgeDocument


def test_chunk_preserves_filename_and_metadata() -> None:
    """
    Source identity and front-matter metadata must survive chunking.
    """

    document = KnowledgeDocument(
        filename="example-policy.md",
        content=(
            "# Example Policy\n\n"
            "## Return Window\n\n"
            "Customers may return eligible products within 30 days."
        ),
        metadata={
            "document_id": "TEST-001",
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
        },
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.filename == "example-policy.md"
    assert chunk.heading == "Return Window"

    assert chunk.metadata["document_id"] == "TEST-001"
    assert chunk.metadata["status"] == "active"
    assert chunk.metadata["audience"] == "customer"


def test_chunk_contains_heading_context() -> None:
    """
    The heading should be included in searchable chunk content.

    This gives the embedding model additional semantic context and makes
    heading-specific retrieval more reliable.
    """

    document = KnowledgeDocument(
        filename="shipping.md",
        content=(
            "# Shipping\n\n"
            "## Canada delivery estimate\n\n"
            "Canada deliveries take 5–9 business days after dispatch."
        ),
        metadata={},
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1

    assert "Canada delivery estimate" in chunks[0].content


def test_large_section_is_split() -> None:
    """
    Oversized sections should be divided into multiple retrieval chunks.
    """

    long_content = "A" * 3500

    document = KnowledgeDocument(
        filename="large.md",
        content=f"# Large Document\n\n{long_content}",
        metadata={},
    )

    chunks = chunk_document(
        document,
        max_characters=1000,
        overlap_characters=100,
    )

    assert len(chunks) > 1

    # Every generated chunk should have a stable source identity.
    assert all(chunk.filename == "large.md" for chunk in chunks)


def test_invalid_overlap_is_rejected() -> None:
    """
    Overlap equal to or larger than the chunk size could cause an invalid
    or non-progressing chunking loop, so it should fail fast.
    """

    document = KnowledgeDocument(
        filename="example.md",
        content="# Example\n\nSome content.",
        metadata={},
    )

    try:
        chunk_document(
            document,
            max_characters=100,
            overlap_characters=100,
        )
    except ValueError as exc:
        assert "overlap_characters" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for invalid overlap configuration"
        )