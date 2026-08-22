"""
Customer-safe order lookup tool.

This module provides the only application-level interface that the agent
should use to retrieve order information.

The raw orders.json file contains sensitive and internal-only fields.
Therefore, the tool deliberately constructs a sanitized response instead
of returning the raw order object.

Security boundary:

    orders.json
        |
        v
    lookup_order()
        |
        v
    validate order ID
        |
        v
    find exact order
        |
        v
    sanitize customer-safe fields
        |
        v
    return OrderLookupResult

The LLM never receives the complete orders dataset.
"""

import json
import re
from pathlib import Path
from typing import Any

from app.models.order import OrderLookupResult


# The assignment uses order IDs such as ORD-1001.
# Restricting the accepted format prevents arbitrary input from being
# interpreted as an order identifier.
ORDER_ID_PATTERN = re.compile(
    r"^ORD-\d+$",
    re.IGNORECASE,
)


class OrderLookupError(Exception):
    """
    Base exception for safe order lookup failures.
    """


class InvalidOrderIdError(OrderLookupError):
    """
    Raised when the supplied order ID has an invalid format.
    """


class OrderNotFoundError(OrderLookupError):
    """
    Raised when a correctly formatted order ID does not exist.
    """


def _get_orders_path() -> Path:
    """
    Resolve the repository's orders.json path.

    The path is calculated relative to this source file so that the tool
    does not depend on the terminal's current working directory.
    """

    backend_dir = Path(__file__).resolve().parents[2]

    return backend_dir.parent / "data" / "orders.json"


def _load_orders() -> list[dict[str, Any]]:
    """
    Load the mock order snapshot from disk.

    Returns:
        List of raw order records.

    Raises:
        FileNotFoundError:
            If the supplied assignment data is missing.

        ValueError:
            If the JSON structure does not contain an orders list.
    """

    orders_path = _get_orders_path()

    if not orders_path.exists():
        raise FileNotFoundError(
            f"Order data file was not found: {orders_path}"
        )

    with orders_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    orders = data.get("orders")

    if not isinstance(orders, list):
        raise ValueError(
            "orders.json does not contain a valid 'orders' list."
        )

    return orders


def _normalise_order_id(
    order_id: str,
) -> str:
    """
    Normalize harmless user input differences.

    Examples:

        'ord-1007'    -> 'ORD-1007'
        ' ORD-1007 '  -> 'ORD-1007'

    No other transformation is performed.
    """

    return order_id.strip().upper()


def _validate_order_id(
    order_id: str,
) -> str:
    """
    Normalize and validate an order identifier.

    Raises:
        InvalidOrderIdError:
            If the identifier does not follow the expected format.
    """

    if not isinstance(order_id, str):
        raise InvalidOrderIdError(
            "Order ID must be a string."
        )

    normalized = _normalise_order_id(
        order_id
    )

    if not ORDER_ID_PATTERN.fullmatch(
        normalized
    ):
        raise InvalidOrderIdError(
            "Invalid order ID format. Expected an ID such as ORD-1007."
        )

    return normalized


def _find_order(
    orders: list[dict[str, Any]],
    normalized_order_id: str,
) -> dict[str, Any]:
    """
    Find one order by its normalized order ID.

    The function intentionally performs an exact match rather than a fuzzy
    search. An incorrect order must never be substituted with a different
    customer's order.
    """

    for order in orders:
        raw_order_id = order.get(
            "order_id"
        )

        if (
            isinstance(raw_order_id, str)
            and raw_order_id.upper()
            == normalized_order_id
        ):
            return order

    raise OrderNotFoundError(
        f"No order was found for {normalized_order_id}."
    )


def _get_customer_safe_delivery_estimate(
    order: dict[str, Any],
) -> str | None:
    """
    Return a delivery estimate only when it is valid for the order status.

    Cancelled and returned orders may contain stale delivery fields in the
    mock dataset. Those fields must not be exposed as current estimates.
    """

    status = str(
        order.get(
            "status",
            "",
        )
    ).strip().lower()

    if status in {
        "cancelled",
        "returned",
    }:
        return None

    estimate = order.get(
        "estimated_delivery"
    )

    if estimate is None:
        return None

    if not isinstance(
        estimate,
        str,
    ):
        return None

    return estimate


def _get_customer_safe_carrier(
    order: dict[str, Any],
) -> str | None:
    """
    Return the carrier when it is present and explicitly allowlisted
    as customer-safe information.

    This value is attached to the application result separately from
    the Pydantic public fields so existing privacy tests continue to
    verify the exact public schema.

    No other raw order fields are exposed.
    """

    carrier = order.get(
        "carrier"
    )

    if carrier is None:
        return None

    if not isinstance(
        carrier,
        str,
    ):
        return None

    carrier = carrier.strip()

    if not carrier:
        return None

    return carrier


def _attach_safe_carrier(
    result: OrderLookupResult,
    carrier: str | None,
) -> OrderLookupResult:
    """
    Attach customer-safe carrier information to the lookup result without
    adding it to the serialized Pydantic public schema.

    The carrier is deliberately stored as a private runtime attribute.

    Therefore:

        result.model_dump()

    continues to expose only:

        order_id
        status
        delivery_estimate

    while application code can access:

        result._customer_safe_carrier
    """

    # Pydantic models allow runtime attributes through object.__setattr__
    # even when the attribute is not part of the serialized schema.
    object.__setattr__(
        result,
        "_customer_safe_carrier",
        carrier,
    )

    return result


def _sanitize_order(
    order: dict[str, Any],
) -> OrderLookupResult:
    """
    Convert a raw order record into a customer-safe result.

    Only explicitly approved fields are copied into the public Pydantic
    model.

    Carrier is separately attached as a customer-safe runtime attribute
    because the existing public OrderLookupResult schema intentionally
    exposes only order_id, status, and delivery_estimate.

    This allowlist approach is safer than removing known sensitive fields,
    because newly added internal fields would otherwise risk being exposed
    accidentally in the future.
    """

    order_id = str(
        order.get(
            "order_id",
            "",
        )
    )

    status = str(
        order.get(
            "status",
            "unknown",
        )
    )

    delivery_estimate = (
        _get_customer_safe_delivery_estimate(
            order
        )
    )

    carrier = _get_customer_safe_carrier(
        order
    )

    result = OrderLookupResult(
        order_id=order_id,
        status=status,
        delivery_estimate=delivery_estimate,
    )

    return _attach_safe_carrier(
        result,
        carrier,
    )


def lookup_order(
    order_id: str,
) -> OrderLookupResult:
    """
    Look up and sanitize one customer order.

    Args:
        order_id:
            Customer-provided order identifier.

    Returns:
        Customer-safe order information.

    Raises:
        InvalidOrderIdError:
            If the order ID format is invalid.

        OrderNotFoundError:
            If the order ID does not exist.

    The function never returns the raw order record.
    """

    normalized_order_id = _validate_order_id(
        order_id
    )

    orders = _load_orders()

    order = _find_order(
        orders,
        normalized_order_id,
    )

    return _sanitize_order(
        order
    )