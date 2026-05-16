from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence

from langchain_core.documents import Document

try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    from langchain_classic.retrievers import EnsembleRetriever

from retrieval.bm25_retriever import build_bm25_retriever

logger = logging.getLogger(__name__)

DEFAULT_HYBRID_WEIGHTS = [0.5, 0.5]


# =========================
# Validation
# =========================


def _safe_top_k(
    k: int,
) -> int:
    try:
        return max(
            1,
            int(k),
        )
    except (TypeError, ValueError):
        logger.warning(
            "Invalid hybrid retrieval k=%r received. Falling back to k=4.",
            k,
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
                "Skipping non-Document item at index %s while building hybrid retriever.",
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


def _deduplicate_documents(
    documents: Iterable[Document],
) -> List[Document]:
    unique_documents: List[Document] = []
    seen_keys: set[tuple[str | None, str]] = set()

    for document in documents:
        source = document.metadata.get(
            "source"
        )
        key = (
            source,
            document.page_content,
        )

        if key in seen_keys:
            continue

        seen_keys.add(
            key
        )
        unique_documents.append(
            document
        )

    return unique_documents


# =========================
# Hybrid Retriever
# =========================


def build_hybrid_retriever(
    vector_store,
    documents: Sequence[Document] | Iterable[Document] | None,
    k: int = 4,
) -> Optional[EnsembleRetriever]:
    """Build a hybrid retriever from Chroma semantic search and BM25.

    Semantic retrieval uses ChromaDB embeddings to find chunks with similar
    meaning, even when the query and document use different wording. Keyword
    retrieval uses BM25 sparse retrieval to reward exact token matches, which
    is especially useful for names, acronyms, file terms, IDs, and error text.

    Hybrid retrieval improves RAG robustness by blending both signals. Better
    retrieved context reduces the chance that the LLM has to guess, which helps
    reduce hallucinations while preserving semantic understanding.
    """

    top_k = _safe_top_k(
        k
    )
    valid_documents = _valid_documents(
        documents
    )

    if vector_store is None:
        logger.info(
            "Cannot build hybrid retriever because vector_store is None."
        )
        return None

    if not valid_documents:
        logger.info(
            "Cannot build hybrid retriever because no valid documents were supplied."
        )
        return None

    try:
        # Chroma semantic retrieval keeps embedding-based meaning search.
        # Future extension: add metadata filters through search_kwargs.
        chroma_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k,
            },
        )

        # BM25 keyword retrieval preserves exact-match sparse retrieval.
        bm25_retriever = build_bm25_retriever(
            valid_documents,
            k=top_k,
        )

        return EnsembleRetriever(
            retrievers=[
                chroma_retriever,
                bm25_retriever,
            ],
            weights=DEFAULT_HYBRID_WEIGHTS,
        )

    except Exception:
        logger.exception(
            "Failed to build hybrid retriever."
        )
        return None


def hybrid_retrieve(
    question: str,
    vector_store,
    chunks: Sequence[Document] | Iterable[Document] | None,
    top_k: int,
) -> List[Document]:
    """Retrieve LangChain Documents using Chroma + BM25 hybrid search.

    The output remains a plain list of Documents, so the existing context
    viewer, confidence scoring, reranker, and answer generation pipeline can
    keep working without UI or app-level rewrites.

    Future extension points:
    - Apply query reformulation before retrieval.
    - Add Chroma metadata filtering to search_kwargs.
    - Send hybrid results to a reranker for final ordering.
    - Attach confidence metadata after retrieval or reranking.
    - Incorporate conversational memory into the retrieval query.
    """

    if not question or not question.strip():
        return []

    top_k_value = _safe_top_k(
        top_k
    )

    hybrid_retriever = build_hybrid_retriever(
        vector_store=vector_store,
        documents=chunks,
        k=top_k_value,
    )

    if hybrid_retriever is None:
        return []

    try:
        retrieved_documents = hybrid_retriever.invoke(
            question
        )
    except Exception:
        logger.exception(
            "Hybrid retrieval failed."
        )
        return []

    unique_documents = _deduplicate_documents(
        retrieved_documents
    )

    logger.info(
        "Hybrid retrieval returned %s unique documents.",
        len(unique_documents),
    )

    return unique_documents[:top_k_value]
