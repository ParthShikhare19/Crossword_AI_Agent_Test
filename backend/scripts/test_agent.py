"""
Manual end-to-end test for the Aster & Row support agent.

This script exercises the real application components:

    Router
        ↓
    FAISS retrieval
        ↓
    Metadata reranking
        ↓
    Evidence selection
        ↓
    Conflict detection
        ↓
    Groq generation

It is intentionally kept as a manual diagnostic script rather than a
pytest test because it makes real model/API calls.
"""

from app.agent.agent import (
    SupportAgent,
    create_llm,
)
from app.rag.retriever import (
    create_retriever,
)


def main() -> None:
    """
    Run one real customer-support interaction.
    """

    print("=" * 70)
    print("ASTER & ROW SUPPORT AGENT")
    print("=" * 70)

    # Load the persisted FAISS index.
    retriever = create_retriever()

    # Create the configured Groq model.
    llm = create_llm()

    # Inject both dependencies into the application agent.
    agent = SupportAgent(
        retriever=retriever,
        llm=llm,
    )

    session_id = "manual-test-1"

    user_message = (
        "How long does a regular customer have to return "
        "an unused backpack?"
    )

    print()
    print(f"Customer: {user_message}")
    print()
    print("Processing...")
    print()

    result = agent.handle_message(
        session_id=session_id,
        user_message=user_message,
    )

    print("-" * 70)
    print("ANSWER")
    print("-" * 70)
    print(result.answer)

    print()
    print("-" * 70)
    print("SOURCES")
    print("-" * 70)

    if result.sources:
        for source in result.sources:
            print(
                f"- {source['filename']} "
                f"| {source['heading']}"
            )
    else:
        print("No sources.")

    print()
    print("-" * 70)
    print(f"Intent: {result.intent}")
    print(f"Human handoff: {result.handoff}")
    print("-" * 70)


if __name__ == "__main__":
    main()