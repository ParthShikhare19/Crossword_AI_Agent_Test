"""
Tests for authoritative-source conflict detection.

The tests verify that the detector:
    - identifies multiple authoritative sources for the same topic
    - ignores superseded sources
    - does not confuse unrelated policies with conflicts
    - handles a single authoritative source safely
"""

from app.models.retrieval import RetrievedChunk
from app.rag.conflict_detector import detect_conflicts
from app.rag.reranker import RerankedResult


def make_result(
    filename: str,
    topic: str,
    eligible: bool = True,
) -> RerankedResult:
    """
    Create a deterministic reranked result for testing.
    """

    chunk = RetrievedChunk(
        chunk_id=f"{filename}:001",
        content="Example content",
        filename=filename,
        heading="Example heading",
        score=0.5,
        metadata={
            "topic": topic,
        },
    )

    return RerankedResult(
        chunk=chunk,
        authority_score=90,
        relevance_score=0.8,
        combined_score=1.7,
        eligible=eligible,
        reason="active, customer-facing, official",
    )


def test_multiple_authoritative_sources_same_topic_are_conflict() -> None:
    """
    Two different authoritative sources addressing the same topic should
    trigger a conservative conflict report.
    """

    source_a = make_result(
        filename="12-breeze-tumbler-product-card.md",
        topic="breeze tumbler",
    )

    source_b = make_result(
        filename="05-breeze-tumbler-care-policy.md",
        topic="breeze tumbler",
    )

    report = detect_conflicts(
        [source_a, source_b]
    )

    assert report.has_conflict is True

    assert set(report.sources) == {
        "12-breeze-tumbler-product-card.md",
        "05-breeze-tumbler-care-policy.md",
    }


def test_unrelated_authoritative_documents_are_not_conflict() -> None:
    """
    Different authoritative topics should not be considered contradictory.
    """

    returns = make_result(
        filename="01-returns-policy-current.md",
        topic="returns",
    )

    warranty = make_result(
        filename="07-warranty.md",
        topic="warranty",
    )

    report = detect_conflicts(
        [returns, warranty]
    )

    assert report.has_conflict is False
    assert report.sources == []


def test_single_authoritative_source_is_not_conflict() -> None:
    """A single authoritative source is safe from a conflict perspective."""

    source = make_result(
        filename="01-returns-policy-current.md",
        topic="returns",
    )

    report = detect_conflicts([source])

    assert report.has_conflict is False


def test_ineligible_sources_are_ignored() -> None:
    """
    Superseded or internal documents should not create a conflict with an
    active authoritative document.
    """

    current = make_result(
        filename="01-returns-policy-current.md",
        topic="returns",
        eligible=True,
    )

    legacy = make_result(
        filename="02-returns-policy-legacy.md",
        topic="returns",
        eligible=False,
    )

    report = detect_conflicts(
        [current, legacy]
    )

    assert report.has_conflict is False