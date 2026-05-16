from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Sequence

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)


class _EmptyBM25Vectorizer:
    """No-op vectorizer used when BM25 receives no valid documents."""

    def get_top_n(
        self,
        query_tokens: List[str],
        documents: List[Document],
        n: int = 4,
    ) -> List[Document]:
        return []


# =========================
# BM25 Keyword Retrieval
# =========================


def preprocess_text(text: str) -> List[str]:
    """Tokenize text for BM25 sparse retrieval.

    BM25 is a keyword-based sparse retrieval algorithm. Unlike semantic
    retrieval, which compares dense embedding vectors by meaning, BM25 scores
    documents by exact token overlap, term frequency, and inverse document
    frequency. This makes it useful for matching names, IDs, acronyms, error
    messages, and other precise words that embeddings may blur together.
    """

    if not isinstance(text, str):
        return []

    return text.lower().split()


def _validate_documents(
    documents: Iterable[Document] | None,
) -> List[Document]:
    """Return only usable LangChain Documents while preserving metadata."""

    if documents is None:
        return []

    valid_documents: List[Document] = []

    for index, document in enumerate(documents):
        if not isinstance(document, Document):
            logger.warning(
                "Skipping non-Document item at index %s while building BM25 retriever.",
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

        valid_documents.append(document)

    return valid_documents


def build_bm25_retriever(
    documents: Sequence[Document] | Iterable[Document] | None,
    k: int = 4,
    preprocess_func: Callable[[str], List[str]] = preprocess_text,
) -> BM25Retriever:
    """Build a standalone BM25 retriever for exact keyword search.

    The returned retriever keeps validated Document objects, including their
    metadata and IDs, so it can later be combined with vector retrieval in
    EnsembleRetriever, hybrid retrieval pipelines, or reranking stages.
    """

    try:
        top_k = max(1, int(k))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid BM25 k=%r received. Falling back to k=4.",
            k,
        )
        top_k = 4

    valid_documents = _validate_documents(
        documents
    )

    if not valid_documents:
        logger.info(
            "No valid documents supplied for BM25 retriever; returning an empty retriever."
        )

        # BM25Retriever.from_documents cannot build an index from an empty
        # corpus, so this no-op vectorizer keeps the public retriever interface
        # available while safely returning no keyword matches.
        return BM25Retriever(
            vectorizer=_EmptyBM25Vectorizer(),
            docs=[],
            k=top_k,
            preprocess_func=preprocess_func,
        )

    try:
        from rank_bm25 import BM25Okapi

        tokenized_documents = [
            preprocess_func(document.page_content)
            for document in valid_documents
        ]

        retriever = BM25Retriever(
            vectorizer=BM25Okapi(
                tokenized_documents
            ),
            docs=valid_documents,
            k=top_k,
            preprocess_func=preprocess_func,
        )

        return retriever

    except Exception:
        logger.exception(
            "Failed to build BM25 retriever."
        )
        raise


# =========================
# Example Usage
# =========================

# from retrieval.bm25_retriever import build_bm25_retriever
#
# bm25_retriever = build_bm25_retriever(chunks, k=4)
# keyword_docs = bm25_retriever.invoke("exact error code or keyword")
#
# Future integration examples:
# - Combine with Chroma retriever using EnsembleRetriever.
# - Pass BM25 results into hybrid retrieval before reranking.
# - Rerank keyword + semantic results with a CrossEncoder.
