from __future__ import annotations

import logging
from typing import Any, List

try:
    from langchain.memory import ConversationBufferMemory
except (ImportError, ModuleNotFoundError):
    from langchain_classic.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)

MEMORY_SESSION_KEY = "conversation_memory"


# =========================
# Conversation Memory
# =========================


def initialize_memory(
    session_state: Any,
) -> ConversationBufferMemory:
    """Create or return Streamlit session-scoped conversational memory.

    RAG retrieval only sees the current query unless we pass prior turns into
    query reformulation. ConversationBufferMemory keeps previous user questions
    and assistant answers local to the current Streamlit session, so follow-up
    questions can be rewritten with context before metadata filtering, hybrid
    retrieval, reranking, and answer generation run.

    Future extension points:
    - Persistent memory across app restarts.
    - Vector memory for long-running conversations.
    - Summarization memory for large histories.
    - User-specific memory in multi-user deployments.
    """

    if MEMORY_SESSION_KEY not in session_state:
        session_state[MEMORY_SESSION_KEY] = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=False,
        )
        logger.info(
            "Initialized new session conversation memory."
        )

    return session_state[MEMORY_SESSION_KEY]


def _history_from_buffer(
    buffer: str,
) -> List[dict[str, str]]:
    messages: List[dict[str, str]] = []

    for line in (
        buffer or ""
    ).splitlines():
        if line.startswith("Human: "):
            messages.append(
                {
                    "role": "user",
                    "content": line.removeprefix(
                        "Human: "
                    ),
                }
            )
        elif line.startswith("AI: "):
            messages.append(
                {
                    "role": "assistant",
                    "content": line.removeprefix(
                        "AI: "
                    ),
                }
            )

    return messages


def get_chat_history(
    memory: ConversationBufferMemory,
) -> List[dict[str, str]]:
    """Return memory as role/content messages for query reformulation."""

    try:
        variables = memory.load_memory_variables(
            {}
        )
        chat_history = variables.get(
            "chat_history",
            "",
        )

        if isinstance(chat_history, str):
            messages = _history_from_buffer(
                chat_history
            )
        else:
            messages = [
                {
                    "role": getattr(
                        message,
                        "type",
                        "user",
                    ),
                    "content": getattr(
                        message,
                        "content",
                        "",
                    ),
                }
                for message in chat_history
            ]

        logger.info(
            "Current memory state contains %s stored messages.",
            len(messages),
        )
        print(
            f"[conversation_memory] Current memory messages: {len(messages)}"
        )

        return messages

    except Exception:
        logger.exception(
            "Failed to read conversation memory."
        )
        return []


def add_to_memory(
    memory: ConversationBufferMemory,
    user_input: str,
    assistant_output: str,
) -> None:
    """Store one completed conversation turn in memory."""

    if not user_input:
        return

    try:
        memory.save_context(
            {
                "input": user_input,
            },
            {
                "output": assistant_output or "",
            },
        )

        stored_turns = len(
            get_chat_history(
                memory
            )
        ) // 2
        logger.info(
            "Stored conversation turn. Total stored turns: %s",
            stored_turns,
        )
        print(
            f"[conversation_memory] Stored turns: {stored_turns}"
        )

    except Exception:
        logger.exception(
            "Failed to add conversation turn to memory."
        )


def clear_memory(
    memory: ConversationBufferMemory,
) -> None:
    """Reset the current session memory safely."""

    try:
        memory.clear()
        logger.info(
            "Cleared session conversation memory."
        )
        print(
            "[conversation_memory] Cleared memory"
        )
    except Exception:
        logger.exception(
            "Failed to clear conversation memory."
        )
