"""
HTTP API for the Aster & Row support agent.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agent.agent import (
    AgentResult,
    SupportAgent,
    create_llm,
)
from app.rag.retriever import create_retriever


router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
)


# ===========================================================================
# REQUEST MODEL
# ===========================================================================

class ChatRequest(BaseModel):
    """
    Incoming customer chat request.
    """

    session_id: str = Field(
        ...,
        min_length=1,
    )

    message: str = Field(
        ...,
        min_length=1,
    )


# ===========================================================================
# RESPONSE MODEL
# ===========================================================================

class ChatResponse(BaseModel):
    """
    Customer-facing API response.

    `order` is structured separately so the frontend does not need to
    parse order information from natural-language LLM output.
    """

    answer: str

    sources: list[dict[str, str]]

    handoff: bool

    intent: str

    order: dict | None = None


# ===========================================================================
# AGENT INSTANCE
# ===========================================================================

agent = SupportAgent(
    retriever=create_retriever(),
    llm=create_llm(),
)


# ===========================================================================
# CHAT ENDPOINT
# ===========================================================================

@router.post(
    "",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Process a customer message.
    """

    result: AgentResult = agent.handle_message(
        session_id=request.session_id,
        user_message=request.message,
    )

    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        handoff=result.handoff,
        intent=result.intent,
        order=result.order,
    )