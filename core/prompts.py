from __future__ import annotations

from typing import Iterable, Mapping


MAX_HISTORY_MESSAGES = 6
MAX_MESSAGE_CHARS = 500


def format_chat_history(
    messages: Iterable[Mapping[str, str]] | None,
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> str:
    """Format recent chat turns for lightweight query reformulation."""

    if not messages:
        return ""

    formatted_messages: list[str] = []

    for message in list(messages)[-max_messages:]:
        role = message.get(
            "role",
            "user",
        )
        content = message.get(
            "content",
            "",
        )

        if not content:
            continue

        formatted_messages.append(
            f"{role}: {content[:MAX_MESSAGE_CHARS]}"
        )

    return "\n".join(
        formatted_messages
    )


def build_query_reformulation_prompt(
    query: str,
    chat_history: str,
) -> str:
    """Build a prompt that turns follow-up questions into standalone queries.

    Reformulation helps retrieval because vector search and BM25 only see the
    current query. Follow-ups such as "what about security?" or "tell more
    about it" need prior conversational context converted into explicit search
    terms before hybrid retrieval runs.
    """

    return f"""
Rewrite the latest question as a standalone search query for document retrieval.
Use the chat history only to resolve references like it, its, that topic, next
topic, or what about. Preserve the user's intent. Do not answer the question.
If the latest question is already standalone, return it unchanged.

Chat history:
{chat_history}

Latest question:
{query}

Standalone query:
""".strip()
