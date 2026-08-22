"""
Pydantic schemas for chat requests and responses.

The API layer uses these models to validate incoming requests and to ensure
that responses returned to the frontend follow a predictable structure.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request payload sent by the frontend for each chat message.
    """

    session_id: str = Field(
        ...,
        min_length=1,
        description="Identifier used to maintain conversation context.",
    )

    message: str = Field(
        ...,
        min_length=1,
        description="Current user message.",
    )


class SourceReference(BaseModel):
    """
    Identifies the knowledge-base passage used to support an answer.

    Filename and heading are intentionally exposed because the assignment
    requires customer-facing source references for policy/product answers.
    """

    filename: str
    heading: str


class ChatResponse(BaseModel):
    """
    Standard response returned by the support-agent API.
    """

    answer: str

    sources: list[SourceReference] = Field(default_factory=list)

    # True when the agent cannot safely resolve the request and recommends
    # assistance from a human support representative.
    handoff: bool = False