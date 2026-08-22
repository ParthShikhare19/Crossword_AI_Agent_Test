"""
Session-based conversation memory for the support agent.

The assignment requires multi-turn conversations while ensuring that one
customer session does not leak context into another session.

This implementation uses an in-memory store because the take-home
assignment does not require distributed deployment or persistent sessions.

The memory layer stores only conversation messages. It does not store
secrets, raw order records, or internal knowledge-base content.
"""

from dataclasses import dataclass, field


@dataclass
class ConversationMessage:
    """
    Represents one message in a conversation.
    """

    role: str
    content: str


@dataclass
class ConversationSession:
    """
    Stores the conversation history for one session.
    """

    messages: list[ConversationMessage] = field(
        default_factory=list
    )


class ConversationMemory:
    """
    In-memory session store.

    Each session_id maps to an independent conversation. This prevents
    unrelated conversations from sharing context.
    """

    def __init__(
        self,
        max_messages: int = 12,
    ) -> None:
        """
        Initialize the conversation store.

        Args:
            max_messages:
                Maximum number of recent messages retained per session.

        A bounded history prevents unrelated old details from being carried
        indefinitely into future requests.
        """

        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than zero."
            )

        self.max_messages = max_messages

        self._sessions: dict[
            str,
            ConversationSession,
        ] = {}

    def _get_or_create_session(
        self,
        session_id: str,
    ) -> ConversationSession:
        """
        Return an existing session or create a new one.
        """

        if not session_id.strip():
            raise ValueError(
                "session_id cannot be empty."
            )

        if session_id not in self._sessions:
            self._sessions[session_id] = (
                ConversationSession()
            )

        return self._sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to a session.

        Only the most recent configured number of messages are retained.
        """

        if not content.strip():
            return

        session = self._get_or_create_session(
            session_id
        )

        session.messages.append(
            ConversationMessage(
                role=role,
                content=content,
            )
        )

        # Keep only recent context. This prevents unrelated old details
        # from being carried forever.
        session.messages = session.messages[
            -self.max_messages:
        ]

    def get_messages(
        self,
        session_id: str,
    ) -> list[ConversationMessage]:
        """
        Return a copy of the current session history.
        """

        if not session_id.strip():
            raise ValueError(
                "session_id cannot be empty."
            )

        session = self._sessions.get(
            session_id
        )

        if session is None:
            return []

        return list(session.messages)

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        """
        Delete all conversation context for a session.
        """

        self._sessions.pop(
            session_id,
            None,
        )


# Shared application-level memory instance.

conversation_memory = ConversationMemory()