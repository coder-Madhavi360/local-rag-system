from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, List, Mapping

logger = logging.getLogger(__name__)

CHAT_HISTORY_KEY = "messages"
VALID_CHAT_ROLES = {
    "user",
    "assistant",
}


# =========================
# Session Chat History
# =========================


def _timestamp() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _valid_message(
    message: Any,
) -> bool:
    return (
        isinstance(
            message,
            Mapping,
        )
        and message.get(
            "role"
        )
        in VALID_CHAT_ROLES
        and isinstance(
            message.get(
                "content"
            ),
            str,
        )
    )


def _normalize_message(
    message: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "role": str(
            message.get(
                "role"
            )
        ),
        "content": str(
            message.get(
                "content",
                "",
            )
        ),
        "timestamp": str(
            message.get(
                "timestamp"
            )
            or _timestamp()
        ),
    }


def initialize_chat_history(
    session_state: Any,
) -> List[dict[str, str]]:
    """Create or repair Streamlit session chat history.

    Streamlit reruns the script on every interaction, so chat messages must
    live in ``st.session_state`` to survive the current app session. Keeping a
    validated history also preserves multi-turn RAG context for display and for
    restoring ConversationBufferMemory if memory state is reset.

    Future extension points:
    - Database-backed history.
    - Multi-user history namespaces.
    - Export/import chat transcripts.
    - Persistent long-term storage across app restarts.
    """

    raw_history = session_state.get(
        CHAT_HISTORY_KEY,
        [],
    )

    if not isinstance(
        raw_history,
        list,
    ):
        logger.warning(
            "Corrupted chat history state detected. Resetting history."
        )
        raw_history = []

    history = [
        _normalize_message(
            message
        )
        for message in raw_history
        if _valid_message(
            message
        )
    ]

    session_state[CHAT_HISTORY_KEY] = history

    logger.info(
        "Chat history initialized with %s messages.",
        len(history),
    )
    print(
        f"[chat_history] Loaded messages: {len(history)}"
    )

    return history


def load_chat_history(
    session_state: Any,
) -> List[dict[str, str]]:
    """Load validated chat messages from the current Streamlit session."""

    if CHAT_HISTORY_KEY not in session_state:
        return initialize_chat_history(
            session_state
        )

    history = session_state.get(
        CHAT_HISTORY_KEY,
        [],
    )

    if not isinstance(
        history,
        list,
    ):
        return initialize_chat_history(
            session_state
        )

    valid_history = [
        _normalize_message(
            message
        )
        for message in history
        if _valid_message(
            message
        )
    ]

    if len(valid_history) != len(history):
        logger.warning(
            "Dropped invalid chat history messages while loading history."
        )
        session_state[CHAT_HISTORY_KEY] = valid_history

    logger.info(
        "Loaded %s chat history messages.",
        len(valid_history),
    )

    return valid_history


def save_chat_message(
    session_state: Any,
    role: str,
    content: str,
) -> dict[str, str] | None:
    """Append one timestamped chat message to session history."""

    if role not in VALID_CHAT_ROLES:
        logger.warning(
            "Ignoring chat message with invalid role: %s",
            role,
        )
        return None

    if not isinstance(
        content,
        str,
    ):
        logger.warning(
            "Ignoring chat message with non-string content."
        )
        return None

    history = load_chat_history(
        session_state
    )
    message = {
        "role": role,
        "content": content,
        "timestamp": _timestamp(),
    }

    history.append(
        message
    )
    session_state[CHAT_HISTORY_KEY] = history

    logger.info(
        "Saved %s chat message. Total stored messages: %s",
        role,
        len(history),
    )
    print(
        f"[chat_history] Saved {role} message | total messages: {len(history)}"
    )

    return message


def clear_chat_history(
    session_state: Any,
) -> None:
    """Clear chat display history for the current Streamlit session."""

    session_state[CHAT_HISTORY_KEY] = []
    logger.info(
        "Cleared chat history."
    )
    print(
        "[chat_history] Cleared history"
    )
