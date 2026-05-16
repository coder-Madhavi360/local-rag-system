from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, List, Sequence

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_SCORE_KEY = "reranker_score"


# =========================
# Validation
# =========================


def _safe_top_k(
    top_k: int,
) -> int:
    try:
        return max(
            1,
            int(top_k),
        )
    except (TypeError, ValueError):
        logger.warning(
            "Invalid reranking top_k=%r received. Falling back to top_k=4.",
            top_k,
        )
        return 4


def _valid_documents(
    documents: Sequence[Document] | Iterable[Document] | None,
) -> List[Document]:
    if documents is None:
        return []

    valid_documents: List[Document] = []

    for index, document in enumerate(documents):
        if not isinstance(document, Document):
            logger.warning(
                "Skipping non-Document item at index %s while reranking.",
                index,
            )
            continue

        if not isinstance(document.page_content, str):
            logger.warning(
                "Skipping document with non-string page_content at index %s.",
                index,
            )
            continue

        if not document.page_content.strip():
            continue

        valid_documents.append(
            document
        )

    return valid_documents


def _copy_with_reranker_score(
    document: Document,
    score: float,
) -> Document:
    metadata = dict(
        document.metadata or {}
    )
    metadata[RERANKER_SCORE_KEY] = float(
        score
    )

    return Document(
        page_content=document.page_content,
        metadata=metadata,
        id=document.id,
    )


# =========================
# Load Reranker
# =========================


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load and cache the HuggingFace CrossEncoder reranker model."""

    logger.info(
        "Loading CrossEncoder reranker model: %s",
        RERANKER_MODEL,
    )

    return CrossEncoder(
        RERANKER_MODEL
    )


# =========================
# Rerank Documents
# =========================


def rerank_documents(
    query: str,
    documents: Sequence[Document] | Iterable[Document] | None,
    top_k: int = 4,
) -> List[Document]:
    """Rerank retrieved chunks with a HuggingFace CrossEncoder.

    Retrieval finds a candidate set of chunks using semantic search, keyword
    search, or hybrid retrieval. Reranking is a second precision step: a
    cross-encoder reads the query and each candidate chunk together, then
    assigns a relevance score to each pair.

    This is different from vector retrieval, which compares precomputed
    embeddings, and from BM25 retrieval, which compares sparse keyword tokens.
    Because the cross-encoder evaluates the full query-document pair, it can
    put the most useful chunks first before the LLM sees the context. Better
    ordering improves answer quality and reduces hallucinations by giving the
    model stronger evidence up front.

    Future extension points:
    - Apply query reformulation before building reranking pairs.
    - Add contextual compression after reranking.
    - Include metadata-aware reranking features when source/page filters exist.
    - Combine reranker scores with confidence scoring or conversational memory.
    """

    if not query or not query.strip():
        return []

    top_k_value = _safe_top_k(
        top_k
    )
    valid_documents = _valid_documents(
        documents
    )

    if not valid_documents:
        logger.info(
            "No valid documents supplied for reranking."
        )
        return []

    try:
        reranker = get_reranker()

        pairs = [
            (
                query,
                document.page_content,
            )
            for document in valid_documents
        ]

        scores = reranker.predict(
            pairs
        )

        scored_documents = [
            _copy_with_reranker_score(
                document,
                float(score),
            )
            for document, score in zip(
                valid_documents,
                scores,
            )
        ]

        ranked_documents = sorted(
            scored_documents,
            key=lambda document: document.metadata.get(
                RERANKER_SCORE_KEY,
                float("-inf"),
            ),
            reverse=True,
        )

        logger.info(
            "Reranked %s documents and returning top %s.",
            len(ranked_documents),
            top_k_value,
        )

        return ranked_documents[:top_k_value]

    except Exception:
        logger.exception(
            "Reranking failed. Returning unreordered retrieved documents."
        )

        return valid_documents[:top_k_value]
