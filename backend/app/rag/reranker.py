"""
Metadata-aware reranking for the Aster & Row knowledge base.

FAISS provides semantic similarity, but semantic similarity alone cannot
determine whether a document is authoritative for a customer-facing answer.

The supplied corpus intentionally contains:
    - active policies
    - superseded policies
    - internal documents
    - draft content
    - product information
    - potentially conflicting authoritative documents

This module applies deterministic business rules after vector retrieval.

The reranker combines two independent signals:

    1. Semantic relevance
       How closely the passage matches the user's question.

    2. Document authority
       Whether the source is approved and appropriate for customer answers.

Important design principle:

    Authority determines whether a document is eligible for customer use.
    Semantic relevance primarily determines the ranking among eligible
    documents.

This prevents an unrelated authoritative document from outranking a
highly relevant authoritative document.

Keeping these decisions deterministic makes retrieval behavior testable
and prevents an LLM from silently choosing between conflicting policies.
"""

from dataclasses import dataclass

from app.models.retrieval import RetrievedChunk


@dataclass
class RerankedResult:
    """
    Represents a candidate after metadata-aware evaluation.

    Attributes:
        chunk:
            Original retrieved passage.

        authority_score:
            Deterministic score representing source authority.

        relevance_score:
            Normalized semantic relevance derived from the FAISS distance.

        combined_score:
            Final deterministic ranking score.

        eligible:
            Whether the source may be used as customer-facing evidence.

        reason:
            Explanation of the authority decision for debugging and logs.
    """

    chunk: RetrievedChunk
    authority_score: int
    relevance_score: float
    combined_score: float
    eligible: bool
    reason: str


# ---------------------------------------------------------------------------
# Metadata authority scoring
# ---------------------------------------------------------------------------
#
# These scores are used primarily to determine whether a source is suitable
# for customer-facing answers.
#
# The actual ranking of eligible documents is driven primarily by semantic
# relevance, not by these authority values.
# ---------------------------------------------------------------------------

STATUS_SCORES = {
    "active": 40,
    "superseded": -100,
    "draft": -100,
}

AUDIENCE_SCORES = {
    "customer": 20,
    "internal": -50,
}

AUTHORITY_SCORES = {
    "official": 30,
    "none": -30,
}


# ---------------------------------------------------------------------------
# Ranking weights
# ---------------------------------------------------------------------------
#
# Semantic relevance receives the majority of the ranking weight.
#
# Authority is still included as a secondary signal, but only after the
# source passes the eligibility threshold.
#
# This ensures:
#
#     relevant official policy
#             >
#     unrelated official document
#
# while still ensuring that:
#
#     superseded/internal content
#             =
#     never eligible
# ---------------------------------------------------------------------------

RELEVANCE_WEIGHT = 0.80
AUTHORITY_WEIGHT = 0.20

ELIGIBILITY_THRESHOLD = 50


def _normalise(value: object) -> str:
    """
    Convert metadata values into a predictable lowercase representation.
    """

    return str(value).strip().lower()


def _calculate_relevance_score(
    distance: float,
) -> float:
    """
    Convert the FAISS distance into a normalized relevance value.

    FAISS returns distance rather than a conventional "higher is better"
    similarity score for this index.

    Lower distance therefore indicates a stronger semantic match.

    The transformation keeps the value bounded between zero and one.
    """

    if distance < 0:
        distance = 0

    return 1.0 / (1.0 + distance)


def calculate_authority_score(
    chunk: RetrievedChunk,
) -> tuple[int, str]:
    """
    Calculate a deterministic authority score for a retrieved passage.

    Authority is used to establish whether the passage is safe and
    appropriate for customer-facing answers.

    It is intentionally separate from semantic relevance so that an
    authoritative document that is unrelated to the customer's question
    cannot dominate retrieval ranking.
    """

    metadata = chunk.metadata

    status = _normalise(
        metadata.get("status", "")
    )

    audience = _normalise(
        metadata.get("audience", "")
    )

    policy_authority = _normalise(
        metadata.get("policy_authority", "")
    )

    customer_answering = metadata.get(
        "customer_answering",
        True,
    )

    score = 0
    reasons: list[str] = []

    # -----------------------------------------------------------------------
    # Document lifecycle status
    # -----------------------------------------------------------------------

    status_score = STATUS_SCORES.get(
        status,
        0,
    )

    score += status_score

    if status == "active":
        reasons.append("active")

    elif status == "superseded":
        reasons.append("superseded")

    elif status == "draft":
        reasons.append("draft")

    # -----------------------------------------------------------------------
    # Intended audience
    # -----------------------------------------------------------------------

    audience_score = AUDIENCE_SCORES.get(
        audience,
        0,
    )

    score += audience_score

    if audience == "customer":
        reasons.append("customer-facing")

    elif audience == "internal":
        reasons.append("internal")

    # -----------------------------------------------------------------------
    # Policy authority
    # -----------------------------------------------------------------------

    authority_score = AUTHORITY_SCORES.get(
        policy_authority,
        0,
    )

    score += authority_score

    if policy_authority == "official":
        reasons.append("official")

    # -----------------------------------------------------------------------
    # Explicit customer-answering restriction
    # -----------------------------------------------------------------------

    if customer_answering is False:
        score -= 100

        reasons.append(
            "not approved for customer answering"
        )

    return score, ", ".join(reasons)


def _calculate_combined_score(
    relevance_score: float,
    authority_score: int,
    eligible: bool,
) -> float:
    """
    Calculate the final ranking score.

    Eligibility is evaluated first.

    Ineligible documents receive negative infinity so that they can never
    accidentally become customer-facing evidence regardless of their
    semantic similarity.

    For eligible documents:

        80% semantic relevance
        20% authority

    This makes semantic relevance the primary ordering signal while still
    allowing authority to act as a secondary precedence signal.
    """

    if not eligible:
        return float("-inf")

    # Convert the authority score into a bounded-ish secondary signal.

    normalized_authority = authority_score / 100.0

    return (
        relevance_score * RELEVANCE_WEIGHT
        + normalized_authority * AUTHORITY_WEIGHT
    )


def rerank_candidates(
    candidates: list[RetrievedChunk],
) -> list[RerankedResult]:
    """
    Combine semantic relevance and document authority.

    The reranking process follows two stages:

        Stage 1:
            Determine whether a document is eligible for customer-facing
            use based on deterministic metadata rules.

        Stage 2:
            Rank eligible documents primarily by semantic relevance.

    This separation is important for reliability.

    An authoritative document that is unrelated to the user's question
    should not outrank a highly relevant authoritative document merely
    because both sources have identical authority metadata.
    """

    results: list[RerankedResult] = []

    for candidate in candidates:

        # ---------------------------------------------------------------
        # Step 1: Determine source authority.
        # ---------------------------------------------------------------

        authority_score, reason = (
            calculate_authority_score(
                candidate
            )
        )

        # ---------------------------------------------------------------
        # Step 2: Calculate semantic relevance.
        # ---------------------------------------------------------------

        relevance_score = (
            _calculate_relevance_score(
                candidate.score
            )
        )

        # ---------------------------------------------------------------
        # Step 3: Determine customer-facing eligibility.
        #
        # This remains intentionally deterministic.
        # ---------------------------------------------------------------

        eligible = (
            authority_score
            >= ELIGIBILITY_THRESHOLD
        )

        # ---------------------------------------------------------------
        # Step 4: Calculate the final ranking score.
        #
        # Ineligible documents receive -inf and therefore cannot be
        # selected as customer-facing evidence.
        # ---------------------------------------------------------------

        combined_score = (
            _calculate_combined_score(
                relevance_score=relevance_score,
                authority_score=authority_score,
                eligible=eligible,
            )
        )

        results.append(
            RerankedResult(
                chunk=candidate,
                authority_score=authority_score,
                relevance_score=relevance_score,
                combined_score=combined_score,
                eligible=eligible,
                reason=reason,
            )
        )

    # Highest combined score appears first.

    results.sort(
        key=lambda result: result.combined_score,
        reverse=True,
    )

    return results


def get_authoritative_candidates(
    candidates: list[RetrievedChunk],
) -> list[RerankedResult]:
    """
    Return only sources eligible for customer-facing evidence.

    This helper provides a simple deterministic boundary between the
    retrieval/reranking stage and the evidence-selection stage.
    """

    return [
        result
        for result in rerank_candidates(candidates)
        if result.eligible
    ]