from pipeline.rewrite import rewrite_query
from pipeline.retrieve import retrieve_documents
from pipeline.rerank import rerank_documents_stage
from pipeline.refine import refine_documents, score_confidence
from pipeline.generate import generate_answer_stage


def run_rag_pipeline(
    query,
    vector_store,
    chunks,
    top_k,
    chat_history=None,
):
    """Run the full local RAG pipeline without handling any Streamlit UI."""

    # 1. Rewrite
    retrieval_query = rewrite_query(
        query=query,
        chat_history=chat_history,
    )

    # 2. Retrieve
    retrieved_docs = retrieve_documents(
        query=retrieval_query,
        vector_store=vector_store,
        chunks=chunks,
        top_k=top_k,
    )

    if not retrieved_docs:
        return {
            "answer": "No relevant information found in uploaded documents.",
            "retrieval_query": retrieval_query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "final_docs": [],
            "confidence": 0,
        }

    # 3. Rerank
    reranked_docs = rerank_documents_stage(
        query=retrieval_query,
        documents=retrieved_docs,
        top_k=top_k,
    )

    # 4. Refine
    final_docs = refine_documents(
        documents=reranked_docs,
        top_k=top_k,
    )

    # 5. Score
    confidence = score_confidence(
        documents=final_docs,
        question=query,
    )

    # 6. Generate
    answer = generate_answer_stage(
        question=query,
        documents=final_docs,
    )

    return {
        "answer": answer,
        "retrieval_query": retrieval_query,
        "retrieved_docs": retrieved_docs,
        "reranked_docs": reranked_docs,
        "final_docs": final_docs,
        "confidence": confidence,
    }
