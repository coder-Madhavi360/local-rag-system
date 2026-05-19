from retrieval.hybrid_retriever import reformulate_query


def rewrite_query(query: str, chat_history=None) -> str:
    """Rewrite a user query into the retrieval query used by the RAG pipeline."""

    rewritten_query = reformulate_query(
        query=query,
        chat_history=chat_history,
    )

    return rewritten_query
