from core.llm import generate_answer


def generate_answer_stage(question, documents):
    """Generate an answer with the existing local LLM generation function."""

    answer = generate_answer(
        question,
        documents,
    )

    return answer


def generate(question, documents=None):
    """Backward-compatible alias for the generation stage."""

    return generate_answer_stage(
        question=question,
        documents=documents or [],
    )
