import math

from langchain_core.documents import Document

from retrieval.reranker import RERANKER_SCORE_KEY


# =========================
# Confidence Score
# =========================


def _extract_score(
    item,
) -> float | None:
    if isinstance(item, Document):
        score = item.metadata.get(
            RERANKER_SCORE_KEY
        )
    elif isinstance(item, tuple) and len(item) >= 2:
        score = item[1]
    else:
        score = None

    try:
        return float(
            score
        )
    except (TypeError, ValueError):
        return None

def confidence_score(
    reranked_docs,
    question,
):

    if not reranked_docs:
        return 0

    # =========================
    # Extract reranker scores
    # =========================

    scores = [
        score
        for score in (
            _extract_score(item)
            for item in reranked_docs
        )
        if score is not None
    ]

    if not scores:
        return 0

    # =========================
    # Average reranker score
    # =========================

    avg_score = (
        sum(scores) / len(scores)
    )

    reranker_confidence = (
        1 / (
            1 + math.exp(-avg_score)
        )
    )

    # =========================
    # Keyword overlap
    # =========================

    question_words = set(
        question.lower().split()
    )

    document_words = set()

    for doc in reranked_docs:

        if isinstance(doc, tuple):
            document = doc[0]
        else:
            document = doc

        content = (
            document.page_content.lower()
        )

        document_words.update(
            content.split()
        )

    overlap = (
        len(
            question_words.intersection(
                document_words
            )
        )
        / max(len(question_words), 1)
    )

    # =========================
    # Retrieved chunk strength
    # =========================

    chunk_strength = min(
        len(reranked_docs) / 5,
        1.0
    )

    # =========================
    # Final confidence
    # =========================

    final_confidence = (
        (0.5 * reranker_confidence)
        + (0.3 * overlap)
        + (0.2 * chunk_strength)
    )

    return int(
        max(
            0,
            min(final_confidence * 100, 100)
        )
    )