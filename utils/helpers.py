import hashlib
import re
from pathlib import Path

MAX_CONTEXT_CHARS = 3200
MAX_SOURCE_CHARS = 850


def hash_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def safe_filename(filename):

    name = Path(filename).name

    return re.sub(
        r"[^A-Za-z0-9._ -]",
        "_",
        name
    )


def format_sources(docs):
    """
    Format retrieved document sources cleanly.
    """

    formatted_sources = []

    for doc in docs:
        metadata = doc.metadata

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", None)
        chunk = metadata.get("chunk", None)

        source_name = source.split("/")[-1]

        if page is not None:
            formatted = f"{source_name} — Page {page}"
        elif chunk is not None:
            formatted = f"{source_name} — Chunk {chunk}"
        else:
            formatted = source_name

        if formatted not in formatted_sources:
            formatted_sources.append(formatted)

    return formatted_sources


def _clean_context_text(text):

    cleaned_text = re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()

    return cleaned_text


def _trim_to_sentence(text, max_chars):

    if len(text) <= max_chars:
        return text

    trimmed_text = text[:max_chars].rsplit(
        " ",
        1,
    )[0].strip()

    sentence_end = max(
        trimmed_text.rfind("."),
        trimmed_text.rfind("?"),
        trimmed_text.rfind("!"),
    )

    if sentence_end >= max_chars * 0.6:
        return trimmed_text[: sentence_end + 1]

    return f"{trimmed_text}..."


def build_cited_context(
    documents,
    max_context_chars=MAX_CONTEXT_CHARS,
    max_source_chars=MAX_SOURCE_CHARS,
):
    """
    Build compact evidence lines for citation-aware local generation.

    Small seq2seq models often copy standalone labels such as "[Source 1]".
    Keeping the label and evidence on one line makes the citation pattern
    easier to follow while preserving the document order from retrieval or
    reranking.
    """

    context_parts = []
    used_chars = 0

    for i, doc in enumerate(documents or [], start=1):
        content = _clean_context_text(
            getattr(
                doc,
                "page_content",
                "",
            )
        )

        if not content:
            continue

        remaining_chars = max_context_chars - used_chars

        if remaining_chars <= 0:
            break

        source_text = _trim_to_sentence(
            content,
            min(
                max_source_chars,
                remaining_chars,
            ),
        )

        evidence_line = f"Source {i}: {source_text}"

        context_parts.append(
            evidence_line
        )
        used_chars += len(
            evidence_line
        )

    return "\n".join(
        context_parts
    )
