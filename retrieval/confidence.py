import math

# =========================
# Confidence Score
# =========================

def confidence_score(
    reranked_docs,
    question,
):

    if not reranked_docs:
        return 0

    scores = [
        score
        for _, score in reranked_docs
    ]

    top_score = scores[0]

    confidence = (
        1 / (
            1 + math.exp(-top_score)
        )
    )

    return int(confidence * 100)