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

    top_score = max(
        scores
    )

    confidence = (
        1 / (
            1 + math.exp(-top_score)
        )
    )

    return int(confidence * 100)
