from __future__ import annotations

import logging
import re
from typing import Iterable, List, Mapping, Optional, Sequence

from langchain_core.documents import Document

try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    from langchain_classic.retrievers import EnsembleRetriever

from core.llm import get_generator
from core.prompts import (
    build_query_reformulation_prompt,
    format_chat_history,
)
from processing.metadata import (
    apply_metadata_filter,
    extract_metadata_filters,
)
from retrieval.bm25_retriever import build_bm25_retriever
from retrieval.vectorstore import apply_metadata_filter as apply_chroma_filter

logger = logging.getLogger(__name__)

DEFAULT_HYBRID_WEIGHTS = [0.5, 0.5]
MAX_REFORMULATED_QUERY_CHARS = 300


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
# Query Reformulation
# =========================


def _clean_reformulated_query(
    generated_text: str,
    fallback_query: str,
) -> str:
    reformulated_query = (
        generated_text or ""
    ).strip()

    reformulated_query = re.sub(
        r"^(standalone query|standalone question|query|question):\s*",
        "",
        reformulated_query,
        flags=re.IGNORECASE,
    )

    reformulated_query = reformulated_query.splitlines()[0].strip()
    reformulated_query = reformulated_query.strip(
        "\"' "
    )

    if not reformulated_query:
        return fallback_query

    if len(reformulated_query) > MAX_REFORMULATED_QUERY_CHARS:
        logger.warning(
            "Discarding overly long reformulated query."
        )
        return fallback_query

    return reformulated_query


def reformulate_query(
    query: str,
    chat_history: Iterable[Mapping[str, str]] | None = None,
) -> str:
    """Rewrite follow-up questions into standalone retrieval queries.

    Conversational users often ask ambiguous follow-ups such as "what about
    security?" Hybrid retrieval works best when BM25 and Chroma receive a query
    with explicit terms, so this lightweight step uses the existing local LLM
    to resolve references from recent chat history before retrieval.
    """

    if not query or not query.strip():
        return ""

    original_query = query.strip()
    formatted_history = format_chat_history(
        chat_history
    )

    if not formatted_history:
        logger.info(
            "No chat history found. Using original query for retrieval."
        )
        print(
            f"[query_reformulation] Original Query: {original_query}"
        )
        print(
            f"[query_reformulation] Reformulated Query: {original_query}"
        )
        return original_query

    try:
        prompt = build_query_reformulation_prompt(
            query=original_query,
            chat_history=formatted_history,
        )

        tokenizer, model = get_generator()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            do_sample=False,
        )

        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        reformulated_query = _clean_reformulated_query(
            generated_text,
            original_query,
        )

    except Exception:
        logger.exception(
            "Query reformulation failed. Falling back to original query."
        )
        reformulated_query = original_query

    logger.info(
        "Original query: %s | Reformulated query: %s",
        original_query,
        reformulated_query,
    )
    print(
        f"[query_reformulation] Original Query: {original_query}"
    )
    print(
        f"[query_reformulation] Reformulated Query: {reformulated_query}"
    )

    return reformulated_query


# =========================
# Hybrid Retriever
# =========================


def build_hybrid_retriever(
    vector_store,
    documents: Sequence[Document] | Iterable[Document] | None,
    k: int = 4,
    metadata_filters: Mapping[str, object] | None = None,
) -> Optional[EnsembleRetriever]:
    """Build a hybrid retriever from Chroma semantic search and BM25.

    Semantic retrieval uses ChromaDB embeddings to find chunks with similar
    meaning, even when the query and document use different wording. Keyword
    retrieval uses BM25 sparse retrieval to reward exact token matches, which
    is especially useful for names, acronyms, file terms, IDs, and error text.

    Hybrid retrieval improves RAG robustness by blending both signals. Metadata
    filtering now runs before retrieval, so document-specific prompts narrow the
    candidate set without changing the reranker, confidence score, or UI.
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
        search_kwargs = apply_chroma_filter(
            search_kwargs={
                "k": top_k,
            },
            metadata_filters=metadata_filters,
            documents=valid_documents,
        )

        # Chroma semantic retrieval keeps embedding-based meaning search while
        # using native metadata filtering when a source, page, or sheet is
        # detected in the query.
        chroma_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        # BM25 has no Chroma-style metadata where clause, so we filter the
        # Document corpus first and then build the same keyword retriever.
        bm25_documents = apply_metadata_filter(
            valid_documents,
            metadata_filters,
        )

        if metadata_filters and not bm25_documents:
            logger.info(
                "Metadata filters matched no BM25 documents: %s",
                metadata_filters,
            )
            return None

        # BM25 keyword retrieval preserves exact-match sparse retrieval.
        bm25_retriever = build_bm25_retriever(
            bm25_documents,
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
    - User-selected filters from the UI.
    - Advanced metadata search across document collections.
    - Send hybrid results to a reranker for final ordering.
    - Attach confidence metadata after retrieval or reranking.
    - Incorporate conversational memory into the retrieval query.
    """

    if not question or not question.strip():
        return []

    top_k_value = _safe_top_k(
        top_k
    )
    metadata_filters = extract_metadata_filters(
        question
    )

    filtered_chunks = apply_metadata_filter(
        chunks,
        metadata_filters,
    )

    if metadata_filters and not filtered_chunks:
        logger.info(
            "No chunks matched metadata filters: %s",
            metadata_filters,
        )
        return []

    hybrid_retriever = build_hybrid_retriever(
        vector_store=vector_store,
        documents=filtered_chunks,
        k=top_k_value,
        metadata_filters=metadata_filters,
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
