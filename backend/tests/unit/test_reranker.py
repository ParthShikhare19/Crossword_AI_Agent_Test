"""
Tests for metadata-aware document precedence.

These tests verify that deterministic source-selection rules prevent
superseded and internal documents from becoming customer-facing authority.

The tests also verify that semantic relevance remains the primary ranking
signal among eligible documents.
"""

from app.models.retrieval import RetrievedChunk
from app.rag.reranker import (
    get_authoritative_candidates,
    rerank_candidates,
)


def make_chunk(
    filename: str,
    heading: str,
    score: float,
    metadata: dict,
) -> RetrievedChunk:
    """
    Create a retrieval result for testing.

    The score represents the FAISS distance, where a lower value means
    stronger semantic similarity.
    """

    return RetrievedChunk(
        chunk_id=f"{filename}:001",
        content="Test content",
        filename=filename,
        heading=heading,
        score=score,
        metadata=metadata,
    )


def test_active_official_customer_policy_is_preferred() -> None:
    """
    Active official customer content should outrank legacy content.
    """

    current = make_chunk(
        filename="01-returns-policy-current.md",
        heading="Standard return window",
        score=1.03,
        metadata={
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
            "customer_answering": True,
        },
    )

    legacy = make_chunk(
        filename="02-returns-policy-legacy.md",
        heading="Return window",
        score=1.01,
        metadata={
            "status": "superseded",
            "audience": "customer",
            "policy_authority": "official",
            "customer_answering": True,
        },
    )

    results = rerank_candidates(
        [legacy, current]
    )

    assert results[0].chunk.filename == (
        "01-returns-policy-current.md"
    )

    authoritative = get_authoritative_candidates(
        [legacy, current]
    )

    assert len(authoritative) == 1

    assert authoritative[0].chunk.filename == (
        "01-returns-policy-current.md"
    )


def test_internal_document_is_not_authoritative() -> None:
    """
    Internal content must not become customer-facing evidence merely because
    it has a high semantic similarity score.
    """

    internal = make_chunk(
        filename="14-internal-content-migration-notes.md",
        heading="Unapproved legacy copy",
        score=0.5,
        metadata={
            "status": "draft",
            "audience": "internal",
            "policy_authority": "none",
            "customer_answering": False,
        },
    )

    authoritative = get_authoritative_candidates(
        [internal]
    )

    assert authoritative == []


def test_customer_answering_false_is_rejected() -> None:
    """
    Explicit customer_answering=False is a hard exclusion signal.
    """

    chunk = make_chunk(
        filename="internal.md",
        heading="Internal guidance",
        score=0.1,
        metadata={
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
            "customer_answering": False,
        },
    )

    results = get_authoritative_candidates(
        [chunk]
    )

    assert results == []


def test_semantic_relevance_ranks_eligible_sources() -> None:
    """
    Among eligible sources with equal authority, the more semantically
    relevant passage should receive the higher combined score.

    This prevents authority metadata from becoming the dominant ranking
    signal when multiple documents are equally authoritative.
    """

    highly_relevant = make_chunk(
        filename="01-returns-policy-current.md",
        heading="Standard return window",
        score=0.50,
        metadata={
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
            "customer_answering": True,
        },
    )

    less_relevant = make_chunk(
        filename="07-warranty.md",
        heading="Warranty periods",
        score=0.80,
        metadata={
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
            "customer_answering": True,
        },
    )

    results = rerank_candidates(
        [less_relevant, highly_relevant]
    )

    assert results[0].chunk.filename == (
        "01-returns-policy-current.md"
    )

    assert (
        results[0].combined_score
        > results[1].combined_score
    )


def test_ineligible_document_cannot_rank_above_eligible_document() -> None:
    """
    A superseded document with excellent semantic similarity must never
    outrank an eligible current source.
    """

    superseded = make_chunk(
        filename="02-returns-policy-legacy.md",
        heading="Return window",
        score=0.01,
        metadata={
            "status": "superseded",
            "audience": "customer",
            "policy_authority": "official",
            "customer_answering": True,
        },
    )

    current = make_chunk(
        filename="01-returns-policy-current.md",
        heading="Standard return window",
        score=1.00,
        metadata={
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
            "customer_answering": True,
        },
    )

    results = rerank_candidates(
        [superseded, current]
    )

    assert results[0].chunk.filename == (
        "01-returns-policy-current.md"
    )

    assert results[1].eligible is False