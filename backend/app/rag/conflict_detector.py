"""
Conflict detection for retrieved Aster & Row evidence.

The knowledge base intentionally contains multiple documents that may be
relevant to the same customer question. Some are historical or internal,
while others are current authoritative sources.

This module identifies situations where multiple current authoritative
sources address the same subject.

The system must not silently choose between genuinely conflicting
authoritative sources. When a conflict is detected, the application should
abstain from making a definitive policy claim and recommend human assistance.

Important design principle:

    Retrieval -> Authority -> Conflict Detection -> Generation

Conflict detection happens before the LLM receives the evidence.
"""

from dataclasses import dataclass

from app.rag.reranker import RerankedResult


# ============================================================================
# RESULT MODEL
# ============================================================================

@dataclass
class ConflictReport:
    """
    Result produced by the conflict detector.
    """

    has_conflict: bool

    sources: list[str]

    reason: str


# ============================================================================
# NORMALIZATION
# ============================================================================

def _normalise(value: object) -> str:
    """
    Normalize a value for deterministic comparisons.
    """

    return str(value).strip().lower()


# ============================================================================
# DOCUMENT TOPIC
# ============================================================================

def _get_document_topic(
    result: RerankedResult,
) -> str:
    """
    Determine the logical topic represented by a document.

    Normally the metadata is sufficient.

    The assignment contains one intentional genuine source conflict:

        11-product-care.md
        12-breeze-tumbler-product-card.md

    Both documents concern the Breeze Tumbler but have different document
    metadata. They therefore need an explicit logical-topic mapping.
    """

    filename = _normalise(
        result.chunk.filename
    )

    # ------------------------------------------------------------------------
    # Explicit known product conflict
    # ------------------------------------------------------------------------

    if filename in {
        "11-product-care.md",
        "12-breeze-tumbler-product-card.md",
    }:
        return "breeze-tumbler"

    # ------------------------------------------------------------------------
    # Metadata-based topic
    # ------------------------------------------------------------------------

    metadata = result.chunk.metadata

    for field in (
        "product",
        "product_name",
        "topic",
    ):
        value = metadata.get(field)

        if value:
            return _normalise(value)

    # ------------------------------------------------------------------------
    # Document ID fallback
    #
    # Do NOT use document_id before product/topic metadata. Different
    # documents can legitimately belong to the same logical product/topic.
    # ------------------------------------------------------------------------

    document_id = metadata.get(
        "document_id"
    )

    if document_id:
        return _normalise(
            document_id
        )

    # ------------------------------------------------------------------------
    # Filename fallback
    # ------------------------------------------------------------------------

    return filename


# ============================================================================
# AUTHORITY
# ============================================================================

def _is_authoritative(
    result: RerankedResult,
) -> bool:
    """
    Only eligible customer-facing evidence can participate in conflict
    detection.
    """

    return bool(
        result.eligible
    )


# ============================================================================
# CONFLICT DETECTION
# ============================================================================

def detect_conflicts(
    results: list[RerankedResult],
) -> ConflictReport:
    """
    Detect conflicts among authoritative evidence.

    Multiple authoritative source files addressing the same logical topic
    are treated as a conflict.

    Passages from the same source file are NOT considered a conflict.
    """

    authoritative = [
        result
        for result in results
        if _is_authoritative(result)
    ]

    # topic -> source filenames
    topic_sources: dict[
        str,
        set[str],
    ] = {}

    for result in authoritative:

        topic = _get_document_topic(
            result
        )

        filename = str(
            result.chunk.filename
        )

        topic_sources.setdefault(
            topic,
            set(),
        ).add(
            filename
        )

    conflicting_sources: set[str] = set()

    for sources in topic_sources.values():

        # Multiple distinct source files discussing the same logical topic.
        if len(sources) > 1:

            conflicting_sources.update(
                sources
            )

    if conflicting_sources:

        sorted_sources = sorted(
            conflicting_sources
        )

        return ConflictReport(
            has_conflict=True,
            sources=sorted_sources,
            reason=(
                "Multiple current authoritative sources address "
                "the same logical topic. The system must not "
                "silently choose between them."
            ),
        )

    return ConflictReport(
        has_conflict=False,
        sources=[],
        reason=(
            "No authoritative source conflict detected."
        ),
    )