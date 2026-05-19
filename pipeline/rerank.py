from retrieval.reranker import rerank_documents


def rerank_documents_stage(query, documents, top_k):
    """Rerank retrieved documents with the existing CrossEncoder reranker."""

    reranked_documents = rerank_documents(
        query=query,
        documents=documents,
        top_k=top_k,
    )

    return reranked_documents


def rerank(query, documents, top_k=4):
    """Backward-compatible alias for the rerank stage."""

    return rerank_documents_stage(
        query=query,
        documents=documents,
        top_k=top_k,
    )
