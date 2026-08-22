"""
Pydantic models for customer-safe order information.

The raw orders.json dataset may contain internal-only fields. These models
represent only information that is approved for use in customer-facing
responses.

Keeping a separate public model prevents accidental exposure of fields such
as customer email, address, internal notes, or risk scores.
"""

from pydantic import BaseModel


class OrderLookupResult(BaseModel):
    """
    Sanitized order information returned by the order lookup tool.
    """

    order_id: str

    status: str

    # Delivery information may be unavailable for some orders.
    delivery_estimate: str | None = None