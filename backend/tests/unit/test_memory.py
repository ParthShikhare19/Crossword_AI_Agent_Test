"""
Unit tests for session-based conversation memory.
"""

from app.agent.memory import ConversationMemory


def test_messages_are_preserved_within_session() -> None:
    """Messages should remain available to later turns."""

    memory = ConversationMemory()

    memory.add_message(
        "session-1",
        "user",
        "Where is ORD-1007?",
    )

    memory.add_message(
        "session-1",
        "assistant",
        "Your order is delayed.",
    )

    messages = memory.get_messages(
        "session-1"
    )

    assert len(messages) == 2
    assert messages[0].content == "Where is ORD-1007?"
    assert messages[1].content == "Your order is delayed."


def test_sessions_are_isolated() -> None:
    """
    Conversation context must never leak between sessions.
    """

    memory = ConversationMemory()

    memory.add_message(
        "session-1",
        "user",
        "Where is ORD-1007?",
    )

    session_two_messages = memory.get_messages(
        "session-2"
    )

    assert session_two_messages == []


def test_old_messages_are_trimmed() -> None:
    """The configured history limit should be respected."""

    memory = ConversationMemory(
        max_messages=2
    )

    memory.add_message(
        "session-1",
        "user",
        "First message",
    )

    memory.add_message(
        "session-1",
        "assistant",
        "First response",
    )

    memory.add_message(
        "session-1",
        "user",
        "Second message",
    )

    messages = memory.get_messages(
        "session-1"
    )

    assert len(messages) == 2
    assert messages[0].content == "First response"
    assert messages[1].content == "Second message"


def test_clear_session_removes_context() -> None:
    """Clearing a session should remove its conversation history."""

    memory = ConversationMemory()

    memory.add_message(
        "session-1",
        "user",
        "Where is my order?",
    )

    memory.clear_session("session-1")

    assert memory.get_messages(
        "session-1"
    ) == []