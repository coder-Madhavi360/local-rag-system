from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _doc_text(document: Any) -> str:
    return getattr(
        document,
        "page_content",
        "",
    ).strip()


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )
        if len(token) > 2 and token not in STOPWORDS
    ]


def _token_set(text: str) -> set[str]:
    return set(
        _tokens(
            text
        )
    )


def _cosine_similarity(
    left_text: str,
    right_text: str,
) -> float:
    left_counter = Counter(
        _tokens(
            left_text
        )
    )
    right_counter = Counter(
        _tokens(
            right_text
        )
    )

    if not left_counter or not right_counter:
        return 0.0

    common_tokens = set(left_counter) & set(right_counter)
    dot_product = sum(
        left_counter[token] * right_counter[token]
        for token in common_tokens
    )

    left_norm = math.sqrt(
        sum(
            value * value
            for value in left_counter.values()
        )
    )
    right_norm = math.sqrt(
        sum(
            value * value
            for value in right_counter.values()
        )
    )

    if not left_norm or not right_norm:
        return 0.0

    return dot_product / (
        left_norm * right_norm
    )


def _overlap_ratio(
    source_text: str,
    target_text: str,
) -> float:
    source_tokens = _token_set(
        source_text
    )
    target_tokens = _token_set(
        target_text
    )

    if not source_tokens or not target_tokens:
        return 0.0

    return len(
        source_tokens & target_tokens
    ) / len(
        source_tokens
    )


def _clamp_score(score: float) -> float:
    return max(
        0.0,
        min(
            100.0,
            score * 100.0,
        ),
    )


def _percent(score: float) -> float:
    return round(
        _clamp_score(
            score
        ),
        1,
    )


def _answer_relevancy(
    question: str,
    answer: str,
) -> float:
    question_answer_similarity = _cosine_similarity(
        question,
        answer,
    )
    question_term_coverage = _overlap_ratio(
        question,
        answer,
    )

    return (
        0.65 * question_answer_similarity
        + 0.35 * question_term_coverage
    )


def _context_match(
    answer: str,
    contexts: list[str],
) -> float:
    combined_context = " ".join(
        contexts
    )
    answer_context_similarity = _cosine_similarity(
        answer,
        combined_context,
    )
    answer_term_support = _overlap_ratio(
        answer,
        combined_context,
    )

    return (
        0.45 * answer_context_similarity
        + 0.55 * answer_term_support
    )


def _retrieval_quality(
    question: str,
    contexts: list[str],
) -> float:
    if not contexts:
        return 0.0

    similarities = [
        _cosine_similarity(
            question,
            context,
        )
        for context in contexts
    ]
    overlaps = [
        _overlap_ratio(
            question,
            context,
        )
        for context in contexts
    ]

    max_similarity = max(
        similarities,
        default=0.0,
    )
    average_similarity = sum(
        similarities
    ) / len(
        similarities
    )
    relevant_context_ratio = sum(
        1
        for overlap in overlaps
        if overlap > 0
    ) / len(
        overlaps
    )

    return (
        0.45 * max_similarity
        + 0.35 * average_similarity
        + 0.20 * relevant_context_ratio
    )


def _hallucination_risk(
    answer: str,
    contexts: list[str],
    context_match: float,
) -> float:
    combined_context = " ".join(
        contexts
    )
    unsupported_answer_terms = 1.0 - _overlap_ratio(
        answer,
        combined_context,
    )

    return (
        0.70 * unsupported_answer_terms
        + 0.30 * (1.0 - context_match)
    )


def run_ragas_evaluation(
    question: str,
    answer: str,
    retrieved_docs: list[Any],
) -> tuple[dict[str, float], str | None]:
    contexts = [
        _doc_text(
            document
        )
        for document in retrieved_docs or []
        if _doc_text(
            document
        )
    ]

    answer_relevancy = _answer_relevancy(
        question or "",
        answer or "",
    )
    context_match = _context_match(
        answer or "",
        contexts,
    )
    retrieval_quality = _retrieval_quality(
        question or "",
        contexts,
    )
    hallucination_risk = _hallucination_risk(
        answer or "",
        contexts,
        context_match,
    )

    scores = {
        "Answer Relevancy": _percent(
            answer_relevancy
        ),
        "Context Match": _percent(
            context_match
        ),
        "Retrieval Quality": _percent(
            retrieval_quality
        ),
        "Hallucination Risk": _percent(
            hallucination_risk
        ),
    }

    return scores, None
