"""
Conversation-aware query construction.

Follow-up questions are often too short to retrieve correctly on their own.

For example:

    User: Do you ship internationally?
    User: What about Canada?

The second message must be interpreted together with the relevant previous
turn.

This module creates a retrieval query from the current message and a small
amount of recent conversation context.

The original customer message is preserved separately so that the final
response still addresses exactly what the customer asked.
"""

from app.agent.memory import conversation_memory


def build_retrieval_query(
    session_id: str,
    current_message: str,
) -> str:
    """
    Build a retrieval query using recent conversation context.

    Only recent user messages are included. Assistant responses are excluded
    because they may contain generated claims that should not become
    retrieval authority.
    """

    history = conversation_memory.get_messages(
        session_id
    )

    previous_user_messages = [
        message.content
        for message in history
        if message.role == "user"
    ][-3:]

    if not previous_user_messages:
        return current_message

    context = "\n".join(
        previous_user_messages
    )

    return (
        "Relevant conversation context:\n"
        f"{context}\n\n"
        "Current customer question:\n"
        f"{current_message}"
    )