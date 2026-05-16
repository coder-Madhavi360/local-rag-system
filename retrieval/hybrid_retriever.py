from langchain_community.retrievers import BM25Retriever
import logging

# Chroma returns distance scores where lower is more relevant.
# Tune this value against your embedding model/data if relevant chunks are
# filtered too aggressively or unrelated chunks still pass through.
CHROMA_DISTANCE_THRESHOLD = 1.0

logger = logging.getLogger(__name__)

# =========================
# BM25 Retriever
# =========================

def build_bm25_retriever(
    chunks,
    top_k,
):

    retriever = (
        BM25Retriever.from_documents(
            chunks
        )
    )

    retriever.k = top_k

    return retriever

# =========================
# Hybrid Retrieval
# =========================

def hybrid_retrieve(
    question,
    vector_store,
    chunks,
    top_k,
):

    if vector_store is None:
        return []

    semantic_results = (
        vector_store.similarity_search_with_score(
            question,
            k=max(top_k * 3, top_k),
        )
    )

    semantic_docs = []

    for index, (doc, distance) in enumerate(
        semantic_results,
        start=1,
    ):

        source = doc.metadata.get(
            "source",
            "unknown",
        )

        logger.info(
            "Chroma result %s | distance=%.4f | source=%s",
            index,
            distance,
            source,
        )

        print(
            f"[retrieval] Chroma result {index}: "
            f"distance={distance:.4f}, source={source}"
        )

        # Chroma distance is inverse relevance: lower is better, higher is
        # weaker. Only chunks at or below this threshold are allowed into the
        # RAG context so unrelated questions do not force random documents into
        # the answer generation step.
        if distance <= CHROMA_DISTANCE_THRESHOLD:
            doc.metadata["chroma_distance"] = distance
            semantic_docs.append(doc)

    if not semantic_docs:
        logger.info(
            "No Chroma results passed relevance threshold %.4f",
            CHROMA_DISTANCE_THRESHOLD,
        )

        print(
            "[retrieval] No Chroma results passed relevance "
            f"threshold {CHROMA_DISTANCE_THRESHOLD:.4f}"
        )

        return []

    bm25 = build_bm25_retriever(
        chunks,
        top_k,
    )

    bm25_docs = bm25.invoke(
        question
    )

    relevant_keys = {
        doc.page_content[:100]
        for doc in semantic_docs
    }

    bm25_docs = [
        doc
        for doc in bm25_docs
        if doc.page_content[:100] in relevant_keys
    ]

    combined = (
        semantic_docs + bm25_docs
    )

    unique_docs = []

    seen = set()

    for doc in combined:

        key = (
            doc.page_content[:100]
        )

        if key not in seen:

            seen.add(key)

            unique_docs.append(doc)

    return unique_docs[:top_k]
