from retrieval.hybrid_retriever import hybrid_retrieve


def retrieve_documents(
    query,
    vector_store,
    chunks,
    top_k,
):
    """Retrieve candidate documents with the existing hybrid retriever."""

    documents = hybrid_retrieve(
        question=query,
        vector_store=vector_store,
        chunks=chunks,
        top_k=top_k,
    )

    return documents


def retrieve(query, vector_store, chunks, top_k):
    """Backward-compatible alias for the retrieve stage."""

    return retrieve_documents(
        query=query,
        vector_store=vector_store,
        chunks=chunks,
        top_k=top_k,
    )
