"""
Customer-safe international shipping lookup tool.

This tool provides structured access to the active international shipping
policy.

Security boundary:

    knowledge-base/06-international-shipping.md
        |
        v
    lookup_shipping()
        |
        v
    validate destination
        |
        v
    return customer-safe shipping information

The tool exposes only policy information that is explicitly present in
the active customer-facing international shipping document.
"""

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ShippingLookupError(Exception):
    """Base exception for shipping lookup failures."""


class UnsupportedDestinationError(ShippingLookupError):
    """
    Raised when the requested destination is not supported.
    """


class ShippingPolicyNotFoundError(ShippingLookupError):
    """
    Raised when the authoritative shipping policy cannot be found.
    """


# ---------------------------------------------------------------------------
# Customer-safe result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShippingLookupResult:
    """
    Customer-safe international shipping information.

    Only information explicitly supported by the active international
    shipping policy is returned.
    """

    destination: str
    supported: bool
    delivery_estimate: str | None = None
    processing_time: str | None = None
    duties_and_taxes: str | None = None
    return_postage: str | None = None
    direct_exchanges: bool | None = None


# ---------------------------------------------------------------------------
# Policy path
# ---------------------------------------------------------------------------


def _get_shipping_policy_path() -> Path:
    """
    Resolve the active international shipping policy.

    The path is calculated relative to this source file so the tool does not
    depend on the current terminal working directory.
    """

    backend_dir = Path(__file__).resolve().parents[2]

    return (
        backend_dir.parent
        / "knowledge-base"
        / "06-international-shipping.md"
    )


def _ensure_policy_exists() -> Path:
    """
    Verify that the authoritative shipping policy exists.
    """

    policy_path = _get_shipping_policy_path()

    if not policy_path.exists():
        raise ShippingPolicyNotFoundError(
            f"International shipping policy was not found: "
            f"{policy_path}"
        )

    return policy_path


# ---------------------------------------------------------------------------
# Destination normalization
# ---------------------------------------------------------------------------


def _normalize_destination(
    destination: str,
) -> str:
    """
    Normalize harmless destination variations.

    Examples:

        "Canada"  -> "Canada"
        "canada"  -> "Canada"
        " CANADA " -> "Canada"
    """

    if not isinstance(destination, str):
        raise UnsupportedDestinationError(
            "Destination must be provided as text."
        )

    normalized = destination.strip().lower()

    if normalized in {
        "canada",
        "ca",
        "can",
    }:
        return "Canada"

    return destination.strip()


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------


def lookup_shipping(
    destination: str,
) -> ShippingLookupResult:
    """
    Look up international shipping information for a destination.

    The current active policy supports only Canada.

    Args:
        destination:
            Country or destination requested by the customer.

    Returns:
        Customer-safe shipping policy information.

    Raises:
        UnsupportedDestinationError:
            If the requested destination is not supported.

        ShippingPolicyNotFoundError:
            If the authoritative shipping policy is missing.

    The function does not return raw document contents.
    """

    _ensure_policy_exists()

    normalized_destination = _normalize_destination(
        destination
    )

    if normalized_destination != "Canada":
        raise UnsupportedDestinationError(
            "Aster & Row currently ships internationally only to Canada."
        )

    return ShippingLookupResult(
        destination="Canada",
        supported=True,
        delivery_estimate=(
            "5-9 business days after dispatch"
        ),
        processing_time=(
            "1-2 business days before dispatch"
        ),
        duties_and_taxes=(
            "Import duties, taxes, and brokerage charges "
            "are not prepaid by Aster & Row. The recipient "
            "is responsible for charges assessed by Canadian "
            "authorities or the carrier."
        ),
        return_postage=(
            "Aster & Row does not provide prepaid labels "
            "for ordinary Canadian change-of-mind returns. "
            "The customer is responsible for return postage."
        ),
        direct_exchanges=False,
    )