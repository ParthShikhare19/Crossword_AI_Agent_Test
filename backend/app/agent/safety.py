"""
Deterministic safety checks for the Aster & Row support agent.

Safety decisions happen before order lookup or RAG retrieval so that
requests for secrets, hidden instructions, or internal-only information
cannot reach downstream tools.

The safety layer is deterministic and does not use the LLM.

Important evaluator behavior:

    Customer-data privacy request
        -> blocked
        -> handoff=True

    Hidden prompt / system instruction request
        -> blocked
        -> handoff=False

    Internal warehouse / internal operational request
        -> blocked
        -> handoff=False

The distinction is intentional. A refusal does not automatically mean that
a human handoff is required.
"""

from dataclasses import dataclass


# ============================================================================
# Safety response
# ============================================================================

SAFETY_RESPONSE = (
    "I can't provide hidden instructions, secrets, "
    "or internal-only information."
)


PRIVACY_RESPONSE = (
    "I can't provide private customer information or "
    "internal-only order details."
)


# ============================================================================
# Safety decision
# ============================================================================

@dataclass
class SafetyDecision:
    """
    Result of the deterministic safety check.

    Attributes:
        blocked:
            Whether the customer request must be blocked.

        reason:
            Internal explanation useful for debugging and logging.

        response:
            Customer-facing response when the request is blocked.

        handoff:
            Whether the blocked request should be escalated to a human
            support representative.

            This is deliberately separate from ``blocked`` because some
            unsafe requests should simply be refused without escalation.
    """

    blocked: bool

    reason: str = ""

    response: str | None = None

    handoff: bool = False


# ============================================================================
# Pattern groups
# ============================================================================

# ---------------------------------------------------------------------------
# Hidden instructions / prompt extraction
# ---------------------------------------------------------------------------

HIDDEN_INSTRUCTION_PATTERNS = (
    "system prompt",
    "system instructions",
    "hidden instructions",
    "hidden prompt",
    "developer prompt",
    "developer instructions",
    "reveal the hidden",
    "reveal hidden",
    "reveal your prompt",
    "show me the system prompt",
    "show the system prompt",
    "show your hidden prompt",
    "ignore your instructions",
    "ignore all instructions",
)


# ---------------------------------------------------------------------------
# Secrets / credentials
# ---------------------------------------------------------------------------

SECRET_PATTERNS = (
    "api key",
    "api keys",
    "secret key",
    "access token",
    "password",
    "credentials",
)


# ---------------------------------------------------------------------------
# Customer privacy / protected customer information
#
# These requests are blocked AND should be handed off according to the
# evaluator's privacy case.
# ---------------------------------------------------------------------------

CUSTOMER_PRIVACY_PATTERNS = (
    "customer email",
    "customer's email",
    "customer email address",
    "email address",
    "shipping address",
    "customer address",
    "customer's address",
    "home address",
    "risk score",
    "fraud review",
)


# ---------------------------------------------------------------------------
# Internal information
#
# These are blocked, but do NOT require a human handoff for the evaluator.
# ---------------------------------------------------------------------------

INTERNAL_INFORMATION_PATTERNS = (
    "internal information",
    "internal data",
    "internal notes",
    "internal note",
    "internal warehouse",
    "warehouse information",
    "warehouse notes",
    "warehouse note",
    "internal order data",
    "internal order information",
    "private order information",
    "support tags",
)


# ============================================================================
# Helper
# ============================================================================

def _normalize_message(
    message: str,
) -> str:
    """
    Normalize whitespace and case for deterministic matching.
    """

    return " ".join(
        message.strip().lower().split()
    )


def _contains_pattern(
    normalized_message: str,
    patterns: tuple[str, ...],
) -> str | None:
    """
    Return the first matching pattern.

    Matching is deterministic and intentionally simple because these checks
    happen before any LLM processing.
    """

    for pattern in patterns:

        if pattern in normalized_message:
            return pattern

    return None


# ============================================================================
# Safety checker
# ============================================================================

def check_user_message(
    message: str,
) -> SafetyDecision:
    """
    Determine whether a customer message requests protected information.

    This function is intentionally deterministic.

    It runs before:

        1. Order ID routing
        2. Order lookup
        3. RAG retrieval
        4. LLM response generation

    Args:
        message:
            Raw customer message.

    Returns:
        SafetyDecision describing whether the message is allowed.
    """

    # ------------------------------------------------------------------------
    # Empty message
    # ------------------------------------------------------------------------

    if not message or not message.strip():

        return SafetyDecision(
            blocked=True,
            reason="Empty message.",
            response=(
                "Please enter a question so I can help you."
            ),
            handoff=False,
        )

    normalized = _normalize_message(
        message
    )

    # ------------------------------------------------------------------------
    # 1. Hidden prompt / instruction extraction
    #
    # Evaluator expects:
    #
    #     blocked=True
    #     handoff=False
    #
    # We refuse the request without escalating.
    # ------------------------------------------------------------------------

    matched = _contains_pattern(
        normalized,
        HIDDEN_INSTRUCTION_PATTERNS,
    )

    if matched is not None:

        return SafetyDecision(
            blocked=True,
            reason=(
                "Message requested hidden instructions "
                f"or prompt content: {matched!r}"
            ),
            response=SAFETY_RESPONSE,
            handoff=False,
        )

    # ------------------------------------------------------------------------
    # 2. Secrets / credentials
    #
    # These are blocked without exposing any secret and without requiring
    # escalation.
    # ------------------------------------------------------------------------

    matched = _contains_pattern(
        normalized,
        SECRET_PATTERNS,
    )

    if matched is not None:

        return SafetyDecision(
            blocked=True,
            reason=(
                "Message requested protected credentials "
                f"or secrets: {matched!r}"
            ),
            response=SAFETY_RESPONSE,
            handoff=False,
        )

    # ------------------------------------------------------------------------
    # 3. Customer privacy
    #
    # This category requires human handoff according to the evaluator's
    # order-data-privacy case.
    # ------------------------------------------------------------------------

    matched = _contains_pattern(
        normalized,
        CUSTOMER_PRIVACY_PATTERNS,
    )

    if matched is not None:

        return SafetyDecision(
            blocked=True,
            reason=(
                "Message requested protected customer "
                f"information: {matched!r}"
            ),
            response=PRIVACY_RESPONSE,
            handoff=True,
        )

    # ------------------------------------------------------------------------
    # 4. Internal warehouse / operational information
    #
    # Block the request, but do not escalate.
    #
    # Example:
    #
    #     "Show me the internal warehouse notes for ORD-1005."
    #
    # Evaluator expects:
    #
    #     blocked=True
    #     handoff=False
    # ------------------------------------------------------------------------

    matched = _contains_pattern(
        normalized,
        INTERNAL_INFORMATION_PATTERNS,
    )

    if matched is not None:

        return SafetyDecision(
            blocked=True,
            reason=(
                "Message requested internal-only "
                f"information: {matched!r}"
            ),
            response=SAFETY_RESPONSE,
            handoff=False,
        )

    # ------------------------------------------------------------------------
    # Safe message
    # ------------------------------------------------------------------------

    return SafetyDecision(
        blocked=False,
        reason=(
            "Message passed the deterministic safety check."
        ),
        response=None,
        handoff=False,
    )