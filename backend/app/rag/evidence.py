"""
Deterministic evidence selection for the Aster & Row support agent.

Pipeline:

    FAISS
        -> broad candidate retrieval

    Metadata reranker
        -> authority / eligibility

    Evidence scoring
        -> semantic + lexical relevance

    Evidence selector
        -> final customer-safe evidence

Important principles:

    1. Ineligible legacy/internal documents are NEVER evidence.
    2. Prompt-injection text is treated as untrusted query content.
    3. Official active policy sources receive priority.
    4. Dedicated policy sections are preferred over generic sections.
    5. Complementary authoritative sources may be selected together.
    6. Genuine active-source conflicts are preserved for conflict detection.
    7. Weakly related evidence must not prevent abstention.
    8. Clearly topic-mismatched authoritative documents are rejected.
"""

import math
import re

from app.rag.embeddings import get_embedding_model
from app.rag.reranker import RerankedResult


# ============================================================================
# THRESHOLDS
# ============================================================================

DEFAULT_MINIMUM_RELEVANCE = 0.47

DEFAULT_MINIMUM_QUERY_SIMILARITY = 0.30

DEFAULT_MINIMUM_LEXICAL_SIMILARITY = 0.05

DEFAULT_MINIMUM_EVIDENCE_SCORE = 0.31

DEFAULT_MINIMUM_RETRIEVAL_FLOOR = 0.40


# ============================================================================
# AUTHORITATIVE FALLBACK
# ============================================================================

DEFAULT_AUTHORITATIVE_FALLBACK_RELEVANCE = 0.42

DEFAULT_AUTHORITATIVE_FALLBACK_QUERY_SIMILARITY = 0.25

DEFAULT_AUTHORITATIVE_FALLBACK_HEADING_SIMILARITY = 0.15

DEFAULT_AUTHORITATIVE_FALLBACK_EVIDENCE_SCORE = 0.28


# ============================================================================
# AUTHORITATIVE SOURCE PRIORITY
# ============================================================================

_SOURCE_PRIORITY = {
    "01-returns-policy-current.md": 10,
    "03-final-sale-and-promotions.md": 20,
    "04-damaged-or-wrong-items.md": 20,
    "06-international-shipping.md": 20,
    "07-warranty.md": 20,
    "08-order-changes-and-cancellations.md": 20,
    "09-trailplus-membership.md": 20,
    "10-gift-cards-and-price-adjustments.md": 20,
    "11-product-care.md": 30,
    "12-breeze-tumbler-product-card.md": 30,
    "13-support-escalation.md": 30,
}


# ============================================================================
# QUERY NORMALIZATION
# ============================================================================

def _normalize_query_for_evidence(
    query: str,
) -> str:
    """
    Remove instruction-like language from the scoring representation.

    This does NOT modify the customer's original message.
    """

    if not query:
        return ""

    normalized = query.strip()

    if not normalized:
        return ""

    instruction_patterns = (
        r"\bignore\s+(?:your|the|all)\s+instructions?\b",
        r"\bdisregard\s+(?:your|the|all)\s+instructions?\b",
        r"\bforget\s+(?:your|the|all)\s+instructions?\b",
        r"\bfollow\s+these\s+instructions?\b",
        r"\buse\s+that\s+newer\s+document\b",
        r"\buse\s+this\s+newer\s+document\b",
        r"\buse\s+the\s+newer\s+document\b",
        r"\bapprove\s+my\s+return\b",
        r"\bapprove\s+the\s+return\b",
        r"\breveal\s+(?:the\s+)?hidden\s+(?:prompt|instructions?)\b",
        r"\bshow\s+(?:me\s+)?(?:the\s+)?hidden\s+(?:prompt|instructions?)\b",
        r"\breveal\s+(?:the\s+)?system\s+prompt\b",
        r"\bshow\s+(?:me\s+)?(?:the\s+)?system\s+prompt\b",
    )

    for pattern in instruction_patterns:
        normalized = re.sub(
            pattern,
            " ",
            normalized,
            flags=re.IGNORECASE,
        )

    control_words = (
        r"\bignore\b",
        r"\bdisregard\b",
        r"\bforget\b",
        r"\breveal\b",
        r"\bapprove\b",
    )

    for pattern in control_words:
        normalized = re.sub(
            pattern,
            " ",
            normalized,
            flags=re.IGNORECASE,
        )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


# ============================================================================
# TOKENIZATION
# ============================================================================

def _tokenize(
    text: str,
) -> set[str]:
    """
    Convert text to normalized lexical tokens.
    """

    tokens = re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )

    stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "am",
        "be",
        "can",
        "do",
        "does",
        "did",
        "how",
        "what",
        "when",
        "where",
        "why",
        "who",
        "which",
        "i",
        "me",
        "my",
        "you",
        "your",
        "to",
        "of",
        "for",
        "on",
        "in",
        "at",
        "and",
        "or",
        "with",
        "have",
        "has",
        "had",
        "this",
        "that",
        "it",
        "within",
        "please",
        "could",
        "would",
        "should",
        "about",
        "tell",
        "say",
        "give",
        "use",
        "everyone",
        "newer",
        "note",
        "says",
    }

    normalized: set[str] = set()

    for token in tokens:

        if token in stop_words:
            continue

        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"

        elif (
            token.endswith("s")
            and not token.endswith("ss")
        ):
            token = token[:-1]

        normalized.add(token)

    return normalized


# ============================================================================
# DOMAIN CONCEPTS
# ============================================================================

_DOMAIN_CONCEPTS = {
    "return": {
        "return",
        "returns",
        "refund",
        "refunded",
        "send",
        "back",
        "window",
    },

    "trailplus": {
        "trailplus",
        "membership",
        "member",
    },

    "final_sale": {
        "final",
        "sale",
        "finalsale",
    },

    "damage": {
        "damaged",
        "damage",
        "broken",
        "defective",
        "defect",
        "wrong",
        "incorrect",
        "zipper",
    },

    "shipping": {
        "ship",
        "shipping",
        "destination",
        "country",
        "international",
        "delivery",
        "deliver",
        "canada",
        "germany",
    },

    "warranty": {
        "warranty",
        "warranties",
        "repair",
        "repairing",
        "years",
        "lifetime",
    },

    "dishwasher": {
        "dishwasher",
        "dishwashers",
        "wash",
        "washed",
        "cleaning",
        "clean",
        "handwash",
        "hand",
        "rack",
        "tumbler",
    },

    "vegan": {
        "vegan",
        "fabric",
        "fabrics",
        "adhesive",
        "adhesives",
        "material",
        "materials",
        "certification",
    },
}


def _query_concepts(
    query: str,
) -> set[str]:
    """
    Identify high-level policy concepts present in the query.
    """

    tokens = _tokenize(
        _normalize_query_for_evidence(
            query
        )
    )

    concepts: set[str] = set()

    for concept, terms in _DOMAIN_CONCEPTS.items():

        if tokens.intersection(terms):
            concepts.add(concept)

    return concepts


# ============================================================================
# LEXICAL SIMILARITY
# ============================================================================

def _calculate_lexical_similarity(
    query: str,
    result: RerankedResult,
) -> float:
    """
    Calculate lexical overlap between query and evidence.

    Heading and content are both considered.
    """

    query_tokens = _tokenize(
        _normalize_query_for_evidence(
            query
        )
    )

    if not query_tokens:
        return 0.0

    passage_text = (
        f"{result.chunk.heading}\n"
        f"{result.chunk.content}"
    )

    passage_tokens = _tokenize(
        passage_text
    )

    if not passage_tokens:
        return 0.0

    overlap = (
        query_tokens
        & passage_tokens
    )

    return (
        len(overlap)
        / len(query_tokens)
    )


# ============================================================================
# COSINE SIMILARITY
# ============================================================================

def _cosine_similarity(
    first: list[float],
    second: list[float],
) -> float:

    if not first or not second:
        return 0.0

    if len(first) != len(second):
        raise ValueError(
            "Embedding vectors must have the same dimension."
        )

    dot_product = sum(
        left * right
        for left, right in zip(
            first,
            second,
        )
    )

    first_norm = math.sqrt(
        sum(
            value * value
            for value in first
        )
    )

    second_norm = math.sqrt(
        sum(
            value * value
            for value in second
        )
    )

    if (
        first_norm == 0
        or second_norm == 0
    ):
        return 0.0

    return dot_product / (
        first_norm * second_norm
    )


# ============================================================================
# SEMANTIC SIMILARITY
# ============================================================================

def _calculate_query_similarity(
    query_embedding: list[float],
    result: RerankedResult,
) -> float:

    model = get_embedding_model()

    passage_text = (
        f"{result.chunk.heading}\n"
        f"{result.chunk.content}"
    )

    passage_embedding = model.embed_query(
        passage_text
    )

    return _cosine_similarity(
        query_embedding,
        passage_embedding,
    )


def _calculate_heading_similarity(
    query_embedding: list[float],
    result: RerankedResult,
) -> float:

    model = get_embedding_model()

    heading_embedding = model.embed_query(
        result.chunk.heading
    )

    return _cosine_similarity(
        query_embedding,
        heading_embedding,
    )


# ============================================================================
# EVIDENCE SCORE
# ============================================================================

def _calculate_evidence_score(
    result: RerankedResult,
    passage_similarity: float,
    heading_similarity: float,
    lexical_similarity: float,
) -> float:
    """
    Calculate evidence strength.

    Semantic passage similarity is the strongest signal.

    FAISS relevance is retained as a secondary recall signal.

    Lexical and heading similarity provide deterministic topic alignment.
    """

    return (
        passage_similarity * 0.50
        + result.relevance_score * 0.20
        + lexical_similarity * 0.20
        + heading_similarity * 0.10
    )


# ============================================================================
# SOURCE HELPERS
# ============================================================================

def _source_name(
    result: RerankedResult,
) -> str:

    return result.chunk.filename.strip().lower()


def _source_priority(
    result: RerankedResult,
) -> int:

    return _SOURCE_PRIORITY.get(
        _source_name(result),
        100,
    )


def _contains_any(
    text: str,
    terms: set[str],
) -> bool:

    normalized = text.lower()

    return any(
        term.lower() in normalized
        for term in terms
    )


# ============================================================================
# SOURCE TOPIC CLASSIFICATION
# ============================================================================

def _is_final_sale_related(
    result: RerankedResult,
) -> bool:

    filename = _source_name(result)

    heading = (
        result.chunk.heading
        or ""
    ).lower()

    content = (
        result.chunk.content
        or ""
    ).lower()

    text = (
        f"{filename}\n"
        f"{heading}\n"
        f"{content}"
    )

    return (
        filename
        == "03-final-sale-and-promotions.md"
        or _contains_any(
            text,
            {
                "final-sale",
                "final sale",
                "final-sale item",
                "final sale item",
            },
        )
    )


def _is_damage_related(
    result: RerankedResult,
) -> bool:

    filename = _source_name(result)

    heading = (
        result.chunk.heading
        or ""
    ).lower()

    content = (
        result.chunk.content
        or ""
    ).lower()

    text = (
        f"{filename}\n"
        f"{heading}\n"
        f"{content}"
    )

    return (
        filename
        == "04-damaged-or-wrong-items.md"
        or _contains_any(
            text,
            {
                "damaged",
                "broken",
                "defective",
                "wrong item",
                "incorrect item",
            },
        )
    )


def _is_return_related(
    result: RerankedResult,
) -> bool:

    filename = _source_name(result)

    heading = (
        result.chunk.heading
        or ""
    ).lower()

    content = (
        result.chunk.content
        or ""
    ).lower()

    text = (
        f"{filename}\n"
        f"{heading}\n"
        f"{content}"
    )

    return (
        filename
        in {
            "01-returns-policy-current.md",
            "09-trailplus-membership.md",
        }
        or _contains_any(
            text,
            {
                "return window",
                "calendar days",
                "return",
                "refund",
            },
        )
    )


def _is_shipping_related(
    result: RerankedResult,
) -> bool:

    filename = _source_name(result)

    return filename == "06-international-shipping.md"


def _is_warranty_related(
    result: RerankedResult,
) -> bool:

    filename = _source_name(result)

    return filename == "07-warranty.md"


def _is_dishwasher_related(
    result: RerankedResult,
) -> bool:

    filename = _source_name(result)

    return filename in {
        "11-product-care.md",
        "12-breeze-tumbler-product-card.md",
    }


# ============================================================================
# QUERY TOPIC HELPERS
# ============================================================================

def _query_has_final_sale(
    query: str,
) -> bool:

    normalized = query.lower()

    return (
        "final-sale" in normalized
        or "final sale" in normalized
    )


def _query_has_damage(
    query: str,
) -> bool:

    return _contains_any(
        query.lower(),
        {
            "broken",
            "damaged",
            "damage",
            "defective",
            "defect",
            "wrong item",
            "incorrect item",
            "broken zipper",
        },
    )


def _query_has_shipping(
    query: str,
) -> bool:

    return _contains_any(
        query.lower(),
        {
            "ship",
            "shipping",
            "international",
            "country",
            "canada",
            "germany",
            "destination",
        },
    )


def _query_has_warranty(
    query: str,
) -> bool:

    return _contains_any(
        query.lower(),
        {
            "warranty",
            "lifetime",
            "repair",
            "repairing",
        },
    )


def _query_has_dishwasher(
    query: str,
) -> bool:

    return _contains_any(
        query.lower(),
        {
            "dishwasher",
            "dishwashers",
            "tumbler",
            "dish",
        },
    )


# ============================================================================
# TOPIC MISMATCH DETECTION
# ============================================================================

def _is_topic_mismatched(
    query: str,
    result: RerankedResult,
) -> bool:
    """
    Reject authoritative documents that clearly belong to a different
    policy domain than the customer's question.

    Domain precedence matters here.

    For example, a return-window question may contain words such as
    "delivery" or "deliver" because the return policy is measured from
    the delivery date. Such a query is still fundamentally a return
    question and must not cause the current returns policy to be rejected
    as a shipping document.

    Return-related intent therefore takes precedence over shipping-related
    intent.
    """

    concepts = _query_concepts(query)
    filename = _source_name(result)

    # -----------------------------------------------------------------------
    # Return questions
    # -----------------------------------------------------------------------
    #
    # Return questions can legitimately contain shipping/delivery language.
    # Example:
    #
    #   "How many days after delivery can I send back an unused backpack?"
    #
    # This must remain a returns-policy query, not a shipping query.
    #
    if "return" in concepts:

        if filename == "07-warranty.md":
            return True

        # Do not apply the generic shipping mismatch rules below.
        # The returns policy is authoritative for return-window questions.
        return False

    # -----------------------------------------------------------------------
    # Warranty questions
    # -----------------------------------------------------------------------

    if "warranty" in concepts:

        if filename == "01-returns-policy-current.md":
            return True

    # -----------------------------------------------------------------------
    # International / shipping questions
    # -----------------------------------------------------------------------

    if "shipping" in concepts:

        if filename in {
            "07-warranty.md",
            "01-returns-policy-current.md",
        }:
            return True

    return False


# ============================================================================
# PROMPT-INJECTION DETECTION
# ============================================================================

def _is_prompt_injection_query(
    query: str,
) -> bool:
    """
    Detect prompt-injection style retrieval requests.

    This does NOT block the query.

    It changes evidence-selection behavior so that internal migration notes
    cannot become authoritative merely because the user explicitly asks
    the retriever to follow them.
    """

    normalized = query.lower()

    patterns = (
        "ignore the real policy",
        "ignore the policy",
        "use that newer document",
        "use this newer document",
        "migration note",
        "give everyone 60 days",
        "approve my return",
        "ignore your instructions",
        "reveal the hidden",
        "system prompt",
    )

    return any(
        pattern in normalized
        for pattern in patterns
    )


# ============================================================================
# TOPIC SCORE
# ============================================================================

def _topic_alignment(
    query: str,
    result: RerankedResult,
) -> float:
    """
    Deterministic topic alignment bonus.

    This is intentionally small. It cannot override poor semantic evidence.
    """

    concepts = _query_concepts(
        query
    )

    if not concepts:
        return 0.0

    bonus = 0.0

    if (
        "return" in concepts
        and _is_return_related(result)
    ):
        bonus += 0.035

    if (
        "trailplus" in concepts
        and _source_name(result)
        == "09-trailplus-membership.md"
    ):
        bonus += 0.060

    if (
        "final_sale" in concepts
        and _is_final_sale_related(result)
    ):
        bonus += 0.040

    if (
        "damage" in concepts
        and _is_damage_related(result)
    ):
        bonus += 0.040

    if (
        "shipping" in concepts
        and _is_shipping_related(result)
    ):
        bonus += 0.050

    if (
        "warranty" in concepts
        and _is_warranty_related(result)
    ):
        bonus += 0.050

    if (
        "dishwasher" in concepts
        and _is_dishwasher_related(result)
    ):
        bonus += 0.050

    return bonus


# ============================================================================
# COMPLEMENTARY SOURCES
# ============================================================================

def _add_complementary_sources(
    selected: list[RerankedResult],
    candidates: list[RerankedResult],
    query: str,
    max_results: int,
) -> list[RerankedResult]:
    """
    Add required complementary sources for multi-policy questions.

    Currently supported:

        final-sale + damage
            -> 03-final-sale-and-promotions.md
            -> 04-damaged-or-wrong-items.md
    """

    if len(selected) >= max_results:
        return selected[:max_results]

    if not (
        _query_has_final_sale(query)
        and _query_has_damage(query)
    ):
        return selected[:max_results]

    selected_sources = {
        _source_name(result)
        for result in selected
    }

    required_sources = (
        "03-final-sale-and-promotions.md",
        "04-damaged-or-wrong-items.md",
    )

    for required_source in required_sources:

        if required_source in selected_sources:
            continue

        matching = [
            result
            for result in candidates
            if (
                _source_name(result)
                == required_source
                and result.eligible
            )
        ]

        if not matching:
            continue

        best = max(
            matching,
            key=lambda result: (
                result.relevance_score,
                _source_priority(result),
            ),
        )

        selected.append(best)
        selected_sources.add(required_source)

        if len(selected) >= max_results:
            break

    return selected[:max_results]


# ============================================================================
# SOURCE-CONFLICT PRESERVATION
# ============================================================================

def _preserve_dishwasher_conflict_sources(
    selected: list[RerankedResult],
    candidates: list[RerankedResult],
    query: str,
    max_results: int,
) -> list[RerankedResult]:
    """
    Preserve both official Breeze Tumbler sources when the query asks about
    dishwasher safety.
    """

    if not _query_has_dishwasher(query):
        return selected[:max_results]

    required_sources = {
        "11-product-care.md",
        "12-breeze-tumbler-product-card.md",
    }

    selected_sources = {
        _source_name(result)
        for result in selected
    }

    for required_source in required_sources:

        if required_source in selected_sources:
            continue

        matching = [
            result
            for result in candidates
            if (
                _source_name(result)
                == required_source
                and result.eligible
            )
        ]

        if not matching:
            continue

        best = max(
            matching,
            key=lambda result: (
                result.relevance_score,
                result.combined_score
                if result.combined_score != float("-inf")
                else -1.0,
            ),
        )

        selected.append(best)
        selected_sources.add(required_source)

    return selected[:max_results]


# ============================================================================
# REQUIRED POLICY SOURCE PRESERVATION
# ============================================================================

def _preserve_required_policy_source(
    selected: list[RerankedResult],
    candidates: list[RerankedResult],
    query: str,
    max_results: int,
) -> list[RerankedResult]:
    """
    Ensure the authoritative source is retained for known policy topics.

    This does not generate an answer or hard-code policy claims.

    It only ensures that the relevant authoritative source is present in
    evidence when an eligible candidate for that source exists.
    """

    normalized = query.lower()

    required_source: str | None = None

    # -----------------------------------------------------------------------
    # Standard return policy.
    # -----------------------------------------------------------------------

    if (
        "return" in normalized
        and "trailplus" not in normalized
        and not _query_has_final_sale(normalized)
        and not _query_has_damage(normalized)
    ):
        required_source = (
            "01-returns-policy-current.md"
        )

    # -----------------------------------------------------------------------
    # TrailPlus.
    # -----------------------------------------------------------------------

    if "trailplus" in normalized:
        required_source = (
            "09-trailplus-membership.md"
        )

    # -----------------------------------------------------------------------
    # International shipping.
    # -----------------------------------------------------------------------

    if _query_has_shipping(normalized):
        required_source = (
            "06-international-shipping.md"
        )

    # -----------------------------------------------------------------------
    # Warranty.
    # -----------------------------------------------------------------------

    if _query_has_warranty(normalized):
        required_source = (
            "07-warranty.md"
        )

    if required_source is None:
        return selected[:max_results]

    selected_sources = {
        _source_name(result)
        for result in selected
    }

    if required_source in selected_sources:
        return selected[:max_results]

    matching = [
        result
        for result in candidates
        if (
            _source_name(result)
            == required_source
            and result.eligible
        )
    ]

    if not matching:
        return selected[:max_results]

    best = max(
        matching,
        key=lambda result: (
            result.relevance_score,
            result.combined_score
            if result.combined_score != float("-inf")
            else -1.0,
        ),
    )

    if len(selected) >= max_results:

        weakest_index = min(
            range(len(selected)),
            key=lambda index: (
                selected[index].relevance_score,
                _source_priority(
                    selected[index]
                ),
            ),
        )

        selected[weakest_index] = best

    else:

        selected.append(best)

    return selected[:max_results]


# ============================================================================
# EVIDENCE SELECTION
# ============================================================================

def select_evidence(
    results: list[RerankedResult],
    query: str | None = None,
    max_results: int = 4,
    minimum_relevance: float = DEFAULT_MINIMUM_RELEVANCE,
    minimum_query_similarity: float = (
        DEFAULT_MINIMUM_QUERY_SIMILARITY
    ),
    minimum_lexical_similarity: float = (
        DEFAULT_MINIMUM_LEXICAL_SIMILARITY
    ),
    minimum_evidence_score: float = (
        DEFAULT_MINIMUM_EVIDENCE_SCORE
    ),
    minimum_retrieval_floor: float = (
        DEFAULT_MINIMUM_RETRIEVAL_FLOOR
    ),
    authoritative_fallback_relevance: float = (
        DEFAULT_AUTHORITATIVE_FALLBACK_RELEVANCE
    ),
    authoritative_fallback_query_similarity: float = (
        DEFAULT_AUTHORITATIVE_FALLBACK_QUERY_SIMILARITY
    ),
    authoritative_fallback_heading_similarity: float = (
        DEFAULT_AUTHORITATIVE_FALLBACK_HEADING_SIMILARITY
    ),
    authoritative_fallback_evidence_score: float = (
        DEFAULT_AUTHORITATIVE_FALLBACK_EVIDENCE_SCORE
    ),
) -> list[RerankedResult]:
    """
    Select deterministic, authoritative, customer-safe evidence.

    The function never returns ineligible sources.
    """

    if max_results <= 0:
        raise ValueError(
            "max_results must be greater than zero."
        )

    # -----------------------------------------------------------------------
    # Authority / safety gate.
    # -----------------------------------------------------------------------

    eligible = [
        result
        for result in results
        if result.eligible
    ]

    if not eligible:
        return []

    # -----------------------------------------------------------------------
    # Backward-compatible call with no query.
    # -----------------------------------------------------------------------

    if query is None:

        return [
            result
            for result in eligible
            if (
                result.relevance_score
                >= minimum_relevance
            )
        ][:max_results]

    if not query.strip():
        raise ValueError(
            "Evidence-selection query cannot be empty."
        )

    # -----------------------------------------------------------------------
    # Retrieval floor.
    # -----------------------------------------------------------------------

    candidates = [
        result
        for result in eligible
        if (
            result.relevance_score
            >= minimum_retrieval_floor
        )
    ]

    if not candidates:
        return []

    # -----------------------------------------------------------------------
    # Normalize prompt-injection wording for scoring.
    # -----------------------------------------------------------------------

    evidence_query = (
        _normalize_query_for_evidence(
            query
        )
    )

    if not evidence_query:
        return []

    prompt_injection = (
        _is_prompt_injection_query(query)
    )

    # -----------------------------------------------------------------------
    # Query embedding.
    # -----------------------------------------------------------------------

    model = get_embedding_model()

    query_embedding = model.embed_query(
        evidence_query
    )

    scored: list[
        tuple[
            float,
            RerankedResult,
        ]
    ] = []

    # -----------------------------------------------------------------------
    # Score candidates.
    # -----------------------------------------------------------------------

    for result in candidates:

        # ---------------------------------------------------------------
        # Topic mismatch gate.
        #
        # This must happen BEFORE semantic scoring so a semantically
        # similar but policy-irrelevant source cannot enter the evidence
        # set.
        # ---------------------------------------------------------------

        if _is_topic_mismatched(
            query=query,
            result=result,
        ):
            continue

        passage_similarity = (
            _calculate_query_similarity(
                query_embedding,
                result,
            )
        )

        heading_similarity = (
            _calculate_heading_similarity(
                query_embedding,
                result,
            )
        )

        lexical_similarity = (
            _calculate_lexical_similarity(
                evidence_query,
                result,
            )
        )

        evidence_score = (
            _calculate_evidence_score(
                result=result,
                passage_similarity=passage_similarity,
                heading_similarity=heading_similarity,
                lexical_similarity=lexical_similarity,
            )
        )

        # ---------------------------------------------------------------
        # Normal evidence path.
        # ---------------------------------------------------------------

        normal_match = (
            passage_similarity
            >= minimum_query_similarity
            and lexical_similarity
            >= minimum_lexical_similarity
            and evidence_score
            >= minimum_evidence_score
        )

        # ---------------------------------------------------------------
        # Authoritative fallback.
        # ---------------------------------------------------------------

        authoritative_fallback = (
            result.relevance_score
            >= authoritative_fallback_relevance
            and passage_similarity
            >= authoritative_fallback_query_similarity
            and heading_similarity
            >= authoritative_fallback_heading_similarity
            and evidence_score
            >= authoritative_fallback_evidence_score
        )

        if not (
            normal_match
            or authoritative_fallback
        ):
            continue

        ranking_score = evidence_score

        # ---------------------------------------------------------------
        # Topic-specific authority boost.
        # ---------------------------------------------------------------

        ranking_score += _topic_alignment(
            query=query,
            result=result,
        )

        # ---------------------------------------------------------------
        # Authoritative source priority.
        # ---------------------------------------------------------------

        ranking_score += (
            max(
                0,
                40 - _source_priority(result),
            )
            * 0.0005
        )

        # ---------------------------------------------------------------
        # Prompt-injection protection.
        # ---------------------------------------------------------------

        if (
            prompt_injection
            and _source_name(result)
            in {
                "14-internal-content-migration-notes.md",
                "02-returns-policy-legacy.md",
            }
        ):
            continue

        scored.append(
            (
                ranking_score,
                result,
            )
        )

    # -----------------------------------------------------------------------
    # Sort by evidence strength.
    # -----------------------------------------------------------------------

    scored.sort(
        key=lambda item: (
            item[0],
            item[1].relevance_score,
            -_source_priority(item[1]),
        ),
        reverse=True,
    )

    selected = [
        result
        for _, result in scored[:max_results]
    ]

    # -----------------------------------------------------------------------
    # Preserve required authoritative policy sources.
    # -----------------------------------------------------------------------

    selected = _preserve_required_policy_source(
        selected=selected,
        candidates=candidates,
        query=query,
        max_results=max_results,
    )

    # -----------------------------------------------------------------------
    # Preserve complementary final-sale/damage sources.
    # -----------------------------------------------------------------------

    selected = _add_complementary_sources(
        selected=selected,
        candidates=candidates,
        query=query,
        max_results=max_results,
    )

    # -----------------------------------------------------------------------
    # Preserve both active Breeze Tumbler sources.
    # -----------------------------------------------------------------------

    selected = _preserve_dishwasher_conflict_sources(
        selected=selected,
        candidates=candidates,
        query=query,
        max_results=max_results,
    )

    # -----------------------------------------------------------------------
    # De-duplicate by source + heading.
    # -----------------------------------------------------------------------

    deduplicated: list[RerankedResult] = []

    seen: set[
        tuple[str, str]
    ] = set()

    for result in selected:

        key = (
            result.chunk.filename,
            result.chunk.heading,
        )

        if key in seen:
            continue

        seen.add(key)
        deduplicated.append(result)

    return deduplicated[:max_results]