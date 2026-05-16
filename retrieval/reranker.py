from sentence_transformers import (
    CrossEncoder,
)

RERANKER_MODEL = (
    "BAAI/bge-reranker-base"
)

# =========================
# Load Reranker
# =========================

def get_reranker():

    return CrossEncoder(
        RERANKER_MODEL
    )

# =========================
# Rerank Documents
# =========================

def rerank_documents(
    question,
    documents,
    top_k,
):

    if not documents:
        return []

    reranker = get_reranker()

    pairs = [
        (
            question,
            doc.page_content,
        )
        for doc in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked[:top_k]