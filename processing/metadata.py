from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable, List, Mapping

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

ANY_SHEET = "__any_sheet__"
SUPPORTED_METADATA_FILTERS = {
    "source",
    "page",
    "sheet",
}


# =========================
# Query Filter Parsing
# =========================


def _clean_value(
    value: str,
) -> str:
    return value.strip().strip("\"'.,;:()[]{}")


def extract_metadata_filters(
    query: str,
) -> dict[str, Any]:
    """Extract lightweight metadata filters from natural language queries.

    Metadata-aware retrieval lets the app narrow candidate chunks before the
    hybrid retriever scores them. Filtering by filename, page, or sheet reduces
    unrelated context and improves precision while keeping the current RAG
    pipeline and returned LangChain Document objects unchanged.

    Future extension points:
    - User-selected filters from the UI.
    - Advanced metadata search with ranges, collections, and tags.
    - Document collections that map friendly names to multiple sources.
    """

    if not isinstance(query, str) or not query.strip():
        return {}

    filters: dict[str, Any] = {}
    normalized_query = query.strip()

    page_match = re.search(
        r"\b(?:from|on|at|in|page)\s+page\s+(\d+)\b|\bpage\s+(\d+)\b",
        normalized_query,
        flags=re.IGNORECASE,
    )
    if page_match:
        page_value = next(
            group
            for group in page_match.groups()
            if group
        )
        try:
            page_number = int(
                page_value
            )
            if page_number > 0:
                filters["page"] = page_number
        except ValueError:
            logger.warning(
                "Ignoring invalid page filter parsed from query: %r",
                page_value,
            )

    sheet_match = re.search(
        r"\b(?:sheet|worksheet|tab)\s+([A-Za-z0-9][\w ._-]*?)(?=\s+(?:and|with|where|for|from|in|on|about)\b|[?.!,;:]|$)",
        normalized_query,
        flags=re.IGNORECASE,
    )
    if sheet_match:
        sheet_name = _clean_value(
            sheet_match.group(1)
        )
        if sheet_name:
            filters["sheet"] = sheet_name
    elif re.search(
        r"\b(?:excel|spreadsheet|worksheets?|sheets?)\b",
        normalized_query,
        flags=re.IGNORECASE,
    ):
        filters["sheet"] = ANY_SHEET

    source_extension = (
        r"(?:pdf|docx|xlsx|xls|txt|md|csv|json|py|html|xml|log)"
    )
    cue_source_pattern = (
        r"\b(?:from|in|inside|within|use|using|only in|search only in|retrieve from)"
        rf"\s+(?:only\s+)?([A-Za-z0-9][\w .()_-]*\.{source_extension})\b"
    )
    bare_source_pattern = (
        rf"\b([A-Za-z0-9][\w()._-]*\.{source_extension})\b"
    )

    cue_candidates = [
        re.sub(
            r"^(?:page\s+\d+\s+in\s+)",
            "",
            _clean_value(
                match.group(1)
            ),
            flags=re.IGNORECASE,
        )
        for match in re.finditer(
            cue_source_pattern,
            normalized_query,
            flags=re.IGNORECASE,
        )
    ]
    bare_candidates = [
        _clean_value(
            match.group(1)
        )
        for match in re.finditer(
            bare_source_pattern,
            normalized_query,
            flags=re.IGNORECASE,
        )
    ]
    source_candidates = cue_candidates or bare_candidates

    if source_candidates:
        source_name = min(
            (
                candidate
                for candidate in source_candidates
                if candidate
            ),
            key=lambda candidate: (
                len(
                    candidate.split()
                ),
                len(
                    candidate
                ),
            ),
            default="",
        )
        if source_name:
            filters["source"] = source_name

    logger.info(
        "Detected metadata filters: %s",
        filters,
    )
    print(
        f"[metadata_filter] Detected filters: {filters or 'none'}"
    )

    return filters


# =========================
# Filter Resolution
# =========================


def _metadata_value(
    document: Document,
    key: str,
) -> Any:
    metadata = document.metadata or {}
    return metadata.get(
        key
    )


def _source_matches(
    metadata_source: Any,
    requested_source: str,
) -> bool:
    if metadata_source is None:
        return False

    source_text = str(
        metadata_source
    )
    requested_text = requested_source.strip()

    return (
        source_text.casefold() == requested_text.casefold()
        or Path(source_text).name.casefold() == requested_text.casefold()
    )


def _value_matches(
    metadata_value: Any,
    requested_value: Any,
) -> bool:
    if requested_value == ANY_SHEET:
        return metadata_value is not None and str(
            metadata_value
        ).strip() != ""

    if metadata_value is None:
        return False

    if isinstance(requested_value, int):
        try:
            return int(
                metadata_value
            ) == requested_value
        except (TypeError, ValueError):
            return False

    return str(
        metadata_value
    ).casefold() == str(
        requested_value
    ).casefold()


def _document_matches_filters(
    document: Document,
    filters: Mapping[str, Any],
) -> bool:
    for key, requested_value in filters.items():
        if key not in SUPPORTED_METADATA_FILTERS:
            continue

        metadata_value = _metadata_value(
            document,
            key,
        )

        if key == "source":
            if not _source_matches(
                metadata_value,
                str(requested_value),
            ):
                return False
            continue

        if not _value_matches(
            metadata_value,
            requested_value,
        ):
            return False

    return True


def _resolved_filter_values(
    documents: Iterable[Document],
    filters: Mapping[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}

    for key, requested_value in filters.items():
        if key not in SUPPORTED_METADATA_FILTERS:
            logger.warning(
                "Ignoring unsupported metadata filter key: %s",
                key,
            )
            continue

        if key == "source":
            for document in documents:
                metadata_source = _metadata_value(
                    document,
                    "source",
                )
                if _source_matches(
                    metadata_source,
                    str(requested_value),
                ):
                    resolved["source"] = metadata_source
                    break
            continue

        if key == "sheet" and requested_value == ANY_SHEET:
            sheet_names = sorted(
                {
                    str(
                        _metadata_value(
                            document,
                            "sheet",
                        )
                    )
                    for document in documents
                    if _metadata_value(
                        document,
                        "sheet",
                    )
                    is not None
                    and str(
                        _metadata_value(
                            document,
                            "sheet",
                        )
                    ).strip()
                }
            )
            if sheet_names:
                resolved["sheet"] = sheet_names
            continue

        for document in documents:
            metadata_value = _metadata_value(
                document,
                key,
            )
            if _value_matches(
                metadata_value,
                requested_value,
            ):
                resolved[key] = metadata_value
                break

    return resolved


def apply_metadata_filter(
    documents: Iterable[Document] | None,
    filters: Mapping[str, Any] | None,
) -> List[Document]:
    """Filter LangChain Documents by metadata while preserving all metadata.

    BM25 has no native metadata filter, so the compatible approach is to filter
    the candidate Document list before building the BM25 index. Chroma receives
    an equivalent metadata ``where`` clause when possible, keeping semantic and
    keyword retrieval aligned.
    """

    if documents is None:
        return []

    document_list = list(
        documents
    )

    if not filters:
        return document_list

    safe_filters = {
        key: value
        for key, value in filters.items()
        if key in SUPPORTED_METADATA_FILTERS and value not in (None, "")
    }

    if not safe_filters:
        return document_list

    filtered_documents = [
        document
        for document in document_list
        if isinstance(
            document,
            Document,
        )
        and _document_matches_filters(
            document,
            safe_filters,
        )
    ]

    logger.info(
        "Applied metadata filters %s: %s/%s documents matched.",
        safe_filters,
        len(filtered_documents),
        len(document_list),
    )
    print(
        "[metadata_filter] Applied filters: "
        f"{safe_filters} | matched {len(filtered_documents)}/{len(document_list)} chunks"
    )

    return filtered_documents


def build_chroma_where_filter(
    documents: Iterable[Document] | None,
    filters: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a ChromaDB metadata filter from validated chunk metadata."""

    if documents is None or not filters:
        return None

    document_list = [
        document
        for document in documents
        if isinstance(
            document,
            Document,
        )
    ]
    if not document_list:
        return None

    resolved = _resolved_filter_values(
        document_list,
        filters,
    )
    if not resolved:
        logger.info(
            "No existing document metadata matched requested filters: %s",
            filters,
        )
        return None

    clauses: list[dict[str, Any]] = []

    for key, value in resolved.items():
        if isinstance(value, list):
            clauses.append(
                {
                    key: {
                        "$in": value,
                    }
                }
            )
        else:
            clauses.append(
                {
                    key: {
                        "$eq": value,
                    }
                }
            )

    if not clauses:
        return None

    if len(clauses) == 1:
        chroma_filter = clauses[0]
    else:
        chroma_filter = {
            "$and": clauses
        }

    logger.info(
        "Built Chroma metadata filter: %s",
        chroma_filter,
    )
    print(
        f"[metadata_filter] Chroma where filter: {chroma_filter}"
    )

    return chroma_filter
