from retrieval.vectorstore import build_vector_store
from core.memory import add_to_memory


def insert_into_vector_store(chunks):
    """Insert chunks into ChromaDB using the existing vector store builder."""

    return build_vector_store(
        chunks
    )


def insert_into_memory(conversation_memory, question, answer):
    """Insert a completed user/assistant turn into conversation memory."""

    add_to_memory(
        conversation_memory,
        question,
        answer,
    )


def build_prompt(query, documents):
    """Compatibility helper for callers that imported this stage previously."""

    from core.llm import build_prompt as build_generation_prompt

    return build_generation_prompt(
        query,
        documents,
    )
