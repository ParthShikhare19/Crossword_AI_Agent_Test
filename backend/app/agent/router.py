"""
Deterministic intent routing for the Aster & Row support agent.

The router decides which application capability should handle a customer
message before the LLM generates a response.

Supported routes:

    ORDER
        Customer is asking about an order and provides an order ID.

    RAG
        Customer is asking about company policies, products, shipping,
        warranty, returns, membership, or other supplied knowledge.

    CLARIFICATION
        The request requires information that is not currently available,
        such as an order ID.

    SAFETY
        The request attempts to access hidden instructions, secrets, or
        internal-only information.

The router intentionally uses deterministic rules for high-risk decisions.
This prevents the model from accidentally receiving sensitive data or
inventing tool usage.
"""

import re
from dataclasses import dataclass
from enum import Enum

from app.agent.safety import check_user_message


class Intent(str, Enum):
    """
    Supported application-level intents.
    """

    ORDER = "order"
    RAG = "rag"
    CLARIFICATION = "clarification"
    SAFETY = "safety"


@dataclass
class RouteDecision:
    """
    Result of application-level intent classification.

    Attributes:
        intent:
            Selected application route.

        order_id:
            Normalized order ID when one is present.

        reason:
            Explanation useful for debugging and observability.
    """

    intent: Intent

    order_id: str | None = None

    reason: str = ""


# ---------------------------------------------------------------------------
# Order ID
# ---------------------------------------------------------------------------

ORDER_ID_PATTERN = re.compile(
    r"\bORD-\d+\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Order-related keywords
# ---------------------------------------------------------------------------

ORDER_KEYWORDS = (
    "order",
    "shipment",
    "shipped",
    "tracking",
    "track",
    "where is",
    "order status",
    "delivery status",
    "shipment status",
)

# ---------------------------------------------------------------------------
# Follow-up phrases
# ---------------------------------------------------------------------------

ORDER_FOLLOW_UP_KEYWORDS = (
    "when will it arrive",
    "when will it",
    "when should it arrive",
    "when should it",
    "where is it",
    "where is that",
    "has it shipped",
    "is it shipped",
    "when does it arrive",
    "when is it arriving",
    "what is the delivery date",
    "what's the delivery date",
    "delivery date",
    "eta",
    "tracking number",
    "track it",
)


# ---------------------------------------------------------------------------
# Order ID extraction
# ---------------------------------------------------------------------------

def extract_order_id(
    message: str,
) -> str | None:
    """
    Extract an order ID from a customer message.

    The function only recognizes the expected ORD-<number> format.

    Examples:

        "Where is ORD-1007?"
            -> "ORD-1007"

        "ord-1007 status"
            -> "ORD-1007"

        "Where is my order?"
            -> None
    """

    if not message:
        return None

    match = ORDER_ID_PATTERN.search(
        message
    )

    if match is None:
        return None

    return match.group(0).upper()


# ---------------------------------------------------------------------------
# Order keyword detection
# ---------------------------------------------------------------------------

def _mentions_order(
    normalized_message: str,
) -> bool:
    """
    Determine whether the message explicitly refers to an order.

    Matching is performed using word boundaries so that words such as
    "ordered" do not accidentally match the keyword "order", and
    "arrived" does not accidentally match "arrive".
    """

    for keyword in ORDER_KEYWORDS:
        pattern = rf"\b{re.escape(keyword)}\b"

        if re.search(
            pattern,
            normalized_message,
        ):
            return True

    return False


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

def is_order_follow_up(
    message: str,
) -> bool:
    """
    Determine whether a message looks like an order-related follow-up.

    This function does NOT decide whether an order actually exists.

    It only identifies linguistic patterns that commonly refer back to a
    previously discussed order.

    Examples:

        "When will it arrive?"
            -> True

        "Where is it?"
            -> True

        "Has it shipped?"
            -> True

        "How long is the return window?"
            -> False
    """

    if not message:
        return False

    normalized = " ".join(
        message.strip().lower().split()
    )

    return any(
        phrase in normalized
        for phrase in ORDER_FOLLOW_UP_KEYWORDS
    )


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def route_message(
    message: str,
) -> RouteDecision:
    """
    Determine the application route for a customer message.

    Safety is checked before any other routing decision.

    Order-related messages without an order ID are routed to clarification
    rather than allowing the system to guess an order.

    All remaining company-specific questions use the RAG pipeline.

    Note:
        Conversation-dependent order follow-ups are intentionally exposed
        through `is_order_follow_up()`. The router itself remains
        deterministic and stateless.
    """

    # -----------------------------------------------------------------------
    # Safety first
    # -----------------------------------------------------------------------

    safety = check_user_message(
        message
    )

    if safety.blocked:
        return RouteDecision(
            intent=Intent.SAFETY,
            reason=(
                "Message was blocked by the deterministic "
                "safety layer."
            ),
        )

    # -----------------------------------------------------------------------
    # Normalize input
    # -----------------------------------------------------------------------

    normalized = (
        message.strip().lower()
        if message
        else ""
    )

    # -----------------------------------------------------------------------
    # Extract order ID
    # -----------------------------------------------------------------------

    order_id = extract_order_id(
        message
    )

    # -----------------------------------------------------------------------
    # Detect order language
    # -----------------------------------------------------------------------

    mentions_order = _mentions_order(
        normalized
    )

    # -----------------------------------------------------------------------
    # Explicit order ID + order language
    # -----------------------------------------------------------------------

    if order_id and mentions_order:
        return RouteDecision(
            intent=Intent.ORDER,
            order_id=order_id,
            reason=(
                "Order-related request contains a valid order ID."
            ),
        )

    # -----------------------------------------------------------------------
    # Valid order ID even when wording is unusual
    #
    # If the customer explicitly gives an order ID, the safest behavior is
    # to treat it as an order request rather than sending it through RAG.
    # -----------------------------------------------------------------------

    if order_id:
        return RouteDecision(
            intent=Intent.ORDER,
            order_id=order_id,
            reason=(
                "Message contains a valid order ID; "
                "route to order lookup."
            ),
        )

    # -----------------------------------------------------------------------
    # Order-related message without an order ID
    # -----------------------------------------------------------------------

    if mentions_order:
        return RouteDecision(
            intent=Intent.CLARIFICATION,
            reason=(
                "The customer appears to be asking about an order, "
                "but no order ID was supplied."
            ),
        )

    # -----------------------------------------------------------------------
    # Conversation-dependent follow-ups are handled by the agent after
    # checking session history.
    #
    # Standalone follow-ups remain RAG here because the router has no
    # conversation state.
    # -----------------------------------------------------------------------

    return RouteDecision(
        intent=Intent.RAG,
        reason=(
            "No order lookup is required; "
            "route to knowledge-base retrieval."
        ),
    )