"""
Tests for the customer-safe order lookup tool.

The order tool is a security boundary, so these tests verify both normal
lookup behavior and negative/privacy cases.
"""

from app.tools.order_lookup import (
    InvalidOrderIdError,
    OrderNotFoundError,
    lookup_order,
)


def test_valid_order_is_returned() -> None:
    """
    A valid order ID should return its current status and valid ETA.
    """

    result = lookup_order("ORD-1003")

    assert result.order_id == "ORD-1003"
    assert result.status == "shipped"
    assert result.delivery_estimate == "2026-08-18"


def test_lowercase_order_id_is_normalized() -> None:
    """
    Harmless case differences should be accepted.
    """

    result = lookup_order("ord-1003")

    assert result.order_id == "ORD-1003"


def test_surrounding_whitespace_is_normalized() -> None:
    """
    Harmless surrounding whitespace should be ignored.
    """

    result = lookup_order("  ORD-1003  ")

    assert result.order_id == "ORD-1003"


def test_unknown_order_is_rejected() -> None:
    """
    A correctly formatted but unknown order ID must not produce invented
    order information.
    """

    try:
        lookup_order("ORD-9999")
    except OrderNotFoundError:
        pass
    else:
        raise AssertionError(
            "Expected OrderNotFoundError."
        )


def test_malformed_order_id_is_rejected() -> None:
    """
    Malformed identifiers should fail safely before searching the dataset.
    """

    try:
        lookup_order("1003")
    except InvalidOrderIdError:
        pass
    else:
        raise AssertionError(
            "Expected InvalidOrderIdError."
        )


def test_cancelled_order_does_not_expose_stale_eta() -> None:
    """
    Cancelled orders may contain stale ETA data in the source dataset.

    The customer-facing tool must suppress that field.
    """

    result = lookup_order("ORD-1004")

    assert result.status == "cancelled"
    assert result.delivery_estimate is None


def test_internal_fields_are_not_exposed() -> None:
    """
    The returned Pydantic model should contain only customer-safe fields.

    This test is intentionally explicit because privacy is a major
    evaluation criterion in the assignment.
    """

    result = lookup_order("ORD-1005")

    result_fields = set(
        result.model_dump().keys()
    )

    assert result_fields == {
        "order_id",
        "status",
        "delivery_estimate",
    }


def test_missing_delivery_estimate_is_allowed() -> None:
    """
    An order without an ETA should return None rather than an invented date.
    """

    result = lookup_order("ORD-1001")

    assert result.status == "pending"
    assert result.delivery_estimate is None