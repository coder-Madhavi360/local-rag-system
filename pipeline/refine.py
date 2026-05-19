from retrieval.confidence import confidence_score


def refine_documents(documents, top_k=4):
    """Filter the reranked context while preserving existing document ordering."""

    refined_documents = documents[:top_k]

    return refined_documents


def score_confidence(documents, question):
    """Calculate confidence using the existing scoring function."""

    return confidence_score(
        documents,
        question,
    )


def refine_context(documents, top_k=4):
    """Backward-compatible alias for the refine stage."""

    return refine_documents(
        documents=documents,
        top_k=top_k,
    )
